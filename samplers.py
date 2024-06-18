import jax
import jax.numpy as jnp
from functools import partial


def z_schedule(i, z0, z1, n_inter):
    """Moves in z follow a linear schedule."""
    return z0 + (z1 - z0) / n_inter * i


def get_n_inter(z0, z1, velocity, dt):
    """Compute the number of intermediate steps."""
    dz = z1 - z0
    abs_dz = jnp.sqrt(jnp.sum(dz ** 2))
    n_inter = jnp.int32(jnp.ceil(abs_dz / (velocity * dt)))
    return n_inter


def accept_step(x0, xnew, log_acc_tot, key):
    """Accept or reject xnew based on log_acc_tot."""
    check_isnan = jnp.any(jnp.isnan(xnew))
    log_acc_tot = jnp.where(check_isnan, -jnp.inf, log_acc_tot)  # always reject
    key, subkey = jax.random.split(key)
    u1 = jax.random.uniform(subkey)
    acceptance = jnp.log(u1) <= log_acc_tot
    xnew = jnp.where(acceptance, xnew, x0)
    return xnew, acceptance, key


def get_key_array(key, n_keys):
    keys = jax.random.split(key, num=n_keys + 1)
    return keys[0], keys[1:]


class GeneralSampler():
    """General sampler class.

    Contains all methods that all different implementations inherit from.
    Specific sampler only needs to specify 'onestep_schedule' and 'body_fn'
    to define procedure to generate move from z0 to z1.
    """
    def __init__(self, energy_fn, energy_partialx_fn, z_sampler, z_sampler_logprob) -> None:
        """z_sampler_logprob(znew,zold)=log prob to propose znew given zold"""
        self.energy_fn = energy_fn
        self.energy_partialx_fn = energy_partialx_fn
        self.z_sampler = z_sampler
        self.z_sampler_logprob = z_sampler_logprob

    # necesseary to mark self as static for jax.jit
    def __hash__(self):
        return hash((self.energy_fn, self.energy_partialx_fn, self.z_sampler, self.z_sampler_logprob))

    def __eq__(self, other):
        return (isinstance(other, GeneralSampler) and
                ((self.energy_fn, self.energy_partialx_fn, self.z_sampler, self.z_sampler_logprob)
                 == (other.energy_fn, other.energy_partialx_fn, other.z_sampler, other.z_sampler_logprob)))

    def get_samples(self, x0, z0, velocity, dt, n_steps, key):
        z_traj = jnp.zeros((n_steps + 1,) + z0.shape)
        accept_traj = jnp.zeros(n_steps, dtype=bool)
        n_inter_traj = jnp.zeros(n_steps, dtype=int)
        x_traj = jnp.zeros((n_steps + 1,) + x0.shape)
        z_traj = z_traj.at[0].set(z0)
        x_traj = x_traj.at[0].set(x0)
        init_vals = (velocity, dt, key, z_traj, x_traj, accept_traj, n_inter_traj)
        velocity, dt, key, z_traj, x_traj, accept_traj, n_inter_traj = jax.lax.fori_loop(0, n_steps,
                                                                                         self.wrapper_body_onestep,
                                                                                         init_vals)
        return z_traj, x_traj, accept_traj.sum() / n_steps, n_inter_traj, key

    def wrapper_body_onestep(self, i, vals):
        velocity, dt, key, z_traj, x_traj, accept_traj, n_inter_traj = vals
        z = z_traj[i]
        x = x_traj[i]
        zprop, key = self.z_sampler(z, key)
        z, x, acceptance, key, log_acc_tot, log_acc_langevin, n_inter = self.onestep_schedule(z, zprop, x, velocity, dt,
                                                                                              key)
        z_traj = z_traj.at[i + 1].set(z)
        x_traj = x_traj.at[i + 1].set(x)
        accept_traj = accept_traj.at[i].set(acceptance)
        n_inter_traj = n_inter_traj.at[i].set(n_inter)
        return (velocity, dt, key, z_traj, x_traj, accept_traj, n_inter_traj)

    def onestep_parallel(self, z0, z1, x0, velocity, dt, n_chains, key):
        key, subkeys = get_key_array(key, n_chains)
        if len(x0.shape) == 2:
            onestep_parallelkeys = jax.vmap(self.onestep_schedule, (None, None, 0, None, None, 0), 0)
            z1_results, xnew_results, acceptance_vals, keys, log_acc_tot, log_acc_langevin, n_inter = onestep_parallelkeys(
                z0, z1, x0, velocity, dt, subkeys)
        else:
            onestep_parallelkeys = jax.vmap(self.onestep_schedule, (None, None, None, None, None, 0), 0)
            z1_results, xnew_results, acceptance_vals, keys, log_acc_tot, log_acc_langevin, n_inter = onestep_parallelkeys(
                z0, z1, x0, velocity, dt, subkeys)
        return z1_results, xnew_results, acceptance_vals, keys[0], log_acc_tot, log_acc_langevin

    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        raise NotImplementedError("Please Implement this method")

    def body_fn(self, i, vals):
        raise NotImplementedError("Please Implement this method")


class AsymSampler(GeneralSampler):
    # jitting here avoids recompilation when calling self.get_samples with different number n_steps
    @partial(jax.jit, static_argnums=0)
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        n_inter = get_n_inter(z0, z1, velocity, dt)
        # z_steps = z0 + dz * jnp.linspace(0, 1., n_inter+1)
        log_acc_zsampler = (self.z_sampler_logprob(z0, z1) - self.z_sampler_logprob(z1, z0))
        log_acc_langevin = 0.  # log(rev) - log(forw)
        init_vals = (log_acc_langevin, key, x0, z0, z1, n_inter, dt)
        (log_acc_langevin, key, xnew, z0, z1, n_inter, dt) = jax.lax.fori_loop(0, n_inter, self.body_fn, init_vals)
        log_acc_ediff = -(self.energy_fn(z1, xnew) - self.energy_fn(z0, x0))
        log_acc_tot = log_acc_zsampler + log_acc_ediff + log_acc_langevin
        xnew, acceptance, key = accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter

    def body_fn(self, i, vals):
        log_acc, key, x, z0, z1, n_inter, dt = vals
        z_old = z_schedule(i, z0, z1, n_inter)
        z_new = z_schedule(i + 1, z0, z1, n_inter)
        neg_grad = -dt * self.energy_partialx_fn(z_new, x)
        key, subkey = jax.random.split(key)
        noise = jnp.sqrt(2 * dt) * jax.random.normal(subkey, shape=x.shape)
        dx = neg_grad + noise
        x += dx
        log_acc += -0.25 / dt * jnp.sum((-dx + dt * self.energy_partialx_fn(z_old, x)) ** 2 - noise ** 2,
                                        axis=-1)  # backward step has different z!
        return (log_acc, key, x, z0, z1, n_inter, dt)


class SymUlaSampler(GeneralSampler):
    @partial(jax.jit, static_argnums=0)
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        n_inter = get_n_inter(z0, z1, velocity, dt)
        log_acc_zsampler = (self.z_sampler_logprob(z0, z1) - self.z_sampler_logprob(z1, z0))
        log_acc_langevin = 0.  # log(rev) - log(forw)
        init_vals = (log_acc_langevin, key, x0, z0, z1, n_inter, dt)
        (log_acc_langevin, key, xnew, z0, z1, n_inter, dt) = jax.lax.fori_loop(0, n_inter + 1, self.body_fn, init_vals)
        log_acc_ediff = -(self.energy_fn(z1, xnew) - self.energy_fn(z0, x0))
        log_acc_tot = log_acc_zsampler + log_acc_ediff + log_acc_langevin
        xnew, acceptance, key = accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter + 1

    def body_fn(self, i, vals):
        log_acc, key, x, z0, z1, n_inter, dt = vals
        z_val = z_schedule(i, z0, z1, n_inter)
        neg_grad = -dt * self.energy_partialx_fn(z_val, x)
        key, subkey = jax.random.split(key)
        noise = jnp.sqrt(2 * dt) * jax.random.normal(subkey, shape=x.shape)
        dx = neg_grad + noise
        x += dx
        log_acc += -0.25 / dt * jnp.sum((-dx + dt * self.energy_partialx_fn(z_val, x)) ** 2 - noise ** 2,
                                        axis=-1)  # backward step has different z!
        return (log_acc, key, x, z0, z1, n_inter, dt)


class SymMalaSampler(GeneralSampler):
    @partial(jax.jit, static_argnums=0)
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        n_inter = get_n_inter(z0, z1, velocity, dt)
        log_acc_zsampler = (self.z_sampler_logprob(z0, z1) - self.z_sampler_logprob(z1, z0))
        log_acc_langevin = 0.  # log(rev) - log(forw)
        init_vals = (log_acc_langevin, key, x0, z0, z1, n_inter, dt)
        (log_acc_langevin, key, xnew, z0, z1, n_inter, dt) = jax.lax.fori_loop(0, n_inter + 1, self.body_fn, init_vals)
        log_acc_tot = log_acc_zsampler + log_acc_langevin
        xnew, acceptance, key = accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter + 1

    def body_fn(self, i, vals):
        log_acc, key, x, z0, z1, n_inter, dt = vals
        z_val = z_schedule(i, z0, z1, n_inter)
        z_val1 = z_schedule(i + 1, z0, z1, n_inter)  # never used if i=n_inter (ok as long as Energy still defined)
        neg_grad = -dt * self.energy_partialx_fn(z_val, x)
        key, subkey = jax.random.split(key)
        noise = jnp.sqrt(2 * dt) * jax.random.normal(subkey, shape=x.shape)
        dx = neg_grad + noise
        xnew = x + dx
        log_acc_mala = (-(self.energy_fn(z_val, xnew) - self.energy_fn(z_val, x))
                        - 0.25 / dt * jnp.sum((-dx + dt * self.energy_partialx_fn(z_val, xnew)) ** 2 - noise ** 2,
                                              axis=-1))
        xnew, acceptance, key = accept_step(x, xnew, log_acc_mala, key)
        log_acc_addition = -(self.energy_fn(z_val1, xnew) - self.energy_fn(z_val, xnew))
        # discount case above (if i=n_inter, z_val1 is meaningless)
        log_acc_addition = jnp.where(i < n_inter, log_acc_addition, 0.)
        log_acc += log_acc_addition
        return (log_acc, key, xnew, z0, z1, n_inter, dt)


class CoinUlaSampler(GeneralSampler):
    @partial(jax.jit, static_argnums=0)
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        n_inter = get_n_inter(z0, z1, velocity, dt)
        key, subkey = jax.random.split(key)
        coin_flip_forw = jax.random.uniform(subkey) <= 0.5
        log_acc_zsampler = (self.z_sampler_logprob(z0, z1) - self.z_sampler_logprob(z1, z0))
        log_acc_langevin = 0.  # log(rev) - log(forw)
        init_vals = (log_acc_langevin, key, x0, z0, z1, n_inter, dt, coin_flip_forw)
        (log_acc_langevin, key, xnew, z0, z1, n_inter, dt, coin_flip_forw) = jax.lax.fori_loop(0, n_inter, self.body_fn,
                                                                                               init_vals)
        log_acc_ediff = -(self.energy_fn(z1, xnew) - self.energy_fn(z0, x0))
        log_acc_tot = log_acc_zsampler + log_acc_ediff + log_acc_langevin
        xnew, acceptance, key = accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter

    def body_fn(self, i, vals):
        log_acc, key, x, z0, z1, n_inter, dt, coin_flip_forw = vals
        z_val0 = z_schedule(i, z0, z1, n_inter)
        z_val1 = z_schedule(i + 1, z0, z1, n_inter)
        z_sample = jnp.where(coin_flip_forw, z_val1, z_val0)
        neg_grad = -dt * self.energy_partialx_fn(z_sample, x)
        key, subkey = jax.random.split(key)
        noise = jnp.sqrt(2 * dt) * jax.random.normal(subkey, shape=x.shape)
        dx = neg_grad + noise
        x += dx
        log_acc += -0.25 / dt * jnp.sum((-dx + dt * self.energy_partialx_fn(z_sample, x)) ** 2 - noise ** 2,
                                        axis=-1)  # backward step has different z!
        return (log_acc, key, x, z0, z1, n_inter, dt, coin_flip_forw)


class CoinMalaSampler(GeneralSampler):
    @partial(jax.jit, static_argnums=0)
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        n_inter = get_n_inter(z0, z1, velocity, dt)
        key, subkey = jax.random.split(key)
        coin_flip_forw = jax.random.uniform(subkey) <= 0.5
        log_acc_zsampler = (self.z_sampler_logprob(z0, z1) - self.z_sampler_logprob(z1, z0))
        log_acc_langevin = 0.  # log(rev) - log(forw)
        init_vals = (log_acc_langevin, key, x0, z0, z1, n_inter, dt, coin_flip_forw)
        (log_acc_langevin, key, xnew, z0, z1, n_inter, dt, coin_flip_forw) = jax.lax.fori_loop(0, n_inter, self.body_fn,
                                                                                               init_vals)
        log_acc_tot = log_acc_zsampler + log_acc_langevin
        xnew, acceptance, key = accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter

    def body_fn(self, i, vals):
        log_acc, key, x, z0, z1, n_inter, dt, coin_flip_forw = vals
        z_val0 = z_schedule(i, z0, z1, n_inter)
        z_val1 = z_schedule(i + 1, z0, z1, n_inter)
        z_sample = jnp.where(coin_flip_forw, z_val1, z_val0)
        neg_grad = -dt * self.energy_partialx_fn(z_sample, x)
        key, subkey = jax.random.split(key)
        noise = jnp.sqrt(2 * dt) * jax.random.normal(subkey, shape=x.shape)
        dx = neg_grad + noise
        xnew = x + dx
        log_acc_mala = (-(self.energy_fn(z_sample, xnew) - self.energy_fn(z_sample, x))
                        - 0.25 / dt * jnp.sum((-dx + dt * self.energy_partialx_fn(z_sample, xnew)) ** 2 - noise ** 2,
                                              axis=-1))
        xnew, acceptance, key = accept_step(x, xnew, log_acc_mala, key)
        x_work = jnp.where(coin_flip_forw, x, xnew)
        log_acc += -(self.energy_fn(z_val1, x_work) - self.energy_fn(z_val0, x_work))
        return (log_acc, key, xnew, z0, z1, n_inter, dt, coin_flip_forw)


