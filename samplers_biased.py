import samplers
import jax
import jax.numpy as jnp
from functools import partial


class NoCoinflipUlaBiasedSampler(samplers.GeneralSampler):
    # Uses CoinUla acceptance without performing the coin flip in the protocol.
    @partial(jax.jit, static_argnums=0)
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        n_inter = samplers.get_n_inter(z0, z1, velocity, dt)
        log_acc_zsampler = (self.z_sampler_logprob(z0, z1) - self.z_sampler_logprob(z1, z0))
        log_acc_langevin = 0.  # log(rev) - log(forw)
        init_vals = (log_acc_langevin, key, x0, z0, z1, n_inter, dt)
        (log_acc_langevin, key, xnew, z0, z1, n_inter, dt) = jax.lax.fori_loop(0, n_inter, self.body_fn, init_vals)
        log_acc_ediff = -(self.energy_fn(z1, xnew) - self.energy_fn(z0, x0))
        log_acc_tot = log_acc_zsampler + log_acc_ediff + log_acc_langevin
        xnew, acceptance, key = samplers.accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter

    def body_fn(self, i, vals):
        log_acc, key, x, z0, z1, n_inter, dt = vals
        z_sample = samplers.z_schedule(i + 1, z0, z1, n_inter)
        neg_grad = -dt * self.energy_partialx_fn(z_sample, x)
        key, subkey = jax.random.split(key)
        noise = jnp.sqrt(2 * dt) * jax.random.normal(subkey, shape=x.shape)
        dx = neg_grad + noise
        x += dx
        log_acc += -0.25 / dt * jnp.sum((-dx + dt * self.energy_partialx_fn(z_sample, x)) ** 2 - noise ** 2,
                                        axis=-1)  # backward step has different z!
        return (log_acc, key, x, z0, z1, n_inter, dt)


class NoCoinflipMalaBiasedSampler(samplers.GeneralSampler):
    # Uses CoinMala acceptance without performing the coin flip in the protocol.
    @partial(jax.jit, static_argnums=0)
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        n_inter = samplers.get_n_inter(z0, z1, velocity, dt)
        log_acc_zsampler = (self.z_sampler_logprob(z0, z1) - self.z_sampler_logprob(z1, z0))
        log_acc_langevin = 0.  # log(rev) - log(forw)
        init_vals = (log_acc_langevin, key, x0, z0, z1, n_inter, dt)
        (log_acc_langevin, key, xnew, z0, z1, n_inter, dt) = jax.lax.fori_loop(0, n_inter, self.body_fn, init_vals)
        log_acc_tot = log_acc_zsampler + log_acc_langevin
        xnew, acceptance, key = samplers.accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter

    def body_fn(self, i, vals):
        log_acc, key, x, z0, z1, n_inter, dt = vals
        z_val0 = samplers.z_schedule(i, z0, z1, n_inter)
        z_val1 = samplers.z_schedule(i + 1, z0, z1, n_inter)
        z_sample = z_val1
        neg_grad = -dt * self.energy_partialx_fn(z_sample, x)
        key, subkey = jax.random.split(key)
        noise = jnp.sqrt(2 * dt) * jax.random.normal(subkey, shape=x.shape)
        dx = neg_grad + noise
        xnew = x + dx
        log_acc_mala = (-(self.energy_fn(z_sample, xnew) - self.energy_fn(z_sample, x))
                        - 0.25 / dt * jnp.sum((-dx + dt * self.energy_partialx_fn(z_sample, xnew)) ** 2 - noise ** 2,
                                              axis=-1))
        xnew, acceptance, key = samplers.accept_step(x, xnew, log_acc_mala, key)
        x_work = x
        log_acc += -(self.energy_fn(z_val1, x_work) - self.energy_fn(z_val0, x_work))
        return (log_acc, key, xnew, z0, z1, n_inter, dt)


class SymUlaBiasedSampler(samplers.GeneralSampler):
    # SymUla sampler using the MALA version energies for acceptance.
    @partial(jax.jit, static_argnums=0)
    def onestep_schedule(self, z0, z1, x0, velocity, dt, key):
        n_inter = samplers.get_n_inter(z0, z1, velocity, dt)
        log_acc_zsampler = (self.z_sampler_logprob(z0, z1) - self.z_sampler_logprob(z1, z0))
        log_acc_langevin = 0.  # log(rev) - log(forw)
        init_vals = (log_acc_langevin, key, x0, z0, z1, n_inter, dt)
        (log_acc_langevin, key, xnew, z0, z1, n_inter, dt) = jax.lax.fori_loop(0, n_inter + 1, self.body_fn, init_vals)
        log_acc_tot = log_acc_zsampler + log_acc_langevin
        xnew, acceptance, key = samplers.accept_step(x0, xnew, log_acc_tot, key)
        znew = jnp.where(acceptance, z1, z0)
        return znew, xnew, acceptance, key, log_acc_tot, log_acc_langevin, n_inter + 1

    def body_fn(self, i, vals):
        log_acc, key, x, z0, z1, n_inter, dt = vals
        z_val = samplers.z_schedule(i, z0, z1, n_inter)
        z_val1 = samplers.z_schedule(i + 1, z0, z1,
                                     n_inter)  # meaningless if i=n_inter (ok as long as Energy still defined)
        neg_grad = -dt * self.energy_partialx_fn(z_val, x)
        key, subkey = jax.random.split(key)
        noise = jnp.sqrt(2 * dt) * jax.random.normal(subkey, shape=x.shape)
        dx = neg_grad + noise
        x += dx
        log_acc_addition = -(self.energy_fn(z_val1, x) - self.energy_fn(z_val, x))
        # discount case above (if i=n_inter, z_val1 is meaningless)
        log_acc_addition = jnp.where(i < n_inter, log_acc_addition, 0.)
        log_acc += log_acc_addition
        return (log_acc, key, x, z0, z1, n_inter, dt)