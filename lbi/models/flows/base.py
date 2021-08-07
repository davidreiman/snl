import jax
import optax
from functools import partial
from .maf import MaskedAffineFlow


def InitializeFlow(model_rng, obs_dim, theta_dim, flow_model=None, n_layers=5):
    """
    Initialize a flow model.

    Args:
        model_rng: a jax random number generator
        obs_dim: dimensionality of the observations
        theta_dim: dimensionality of the simulation parameters
        n_layers: number of affine layers in the flow

    Returns:
        initial_params: a list of parameters
        log_pdf: a function from parameters to log-probability of the observations
        sample: a function from parameters to samples of the parameters

    """

    def loss(params, inputs):
        return -log_pdf(params, inputs).mean()

    if flow_model is None:
        flow_model = MaskedAffineFlow
    if type(model_rng) is int:
        model_rng = jax.random.PRNGKey(model_rng)

    init_fun = flow_model(n_layers)
    initial_params, log_pdf, sample = init_fun(model_rng, obs_dim + theta_dim)
    return (initial_params, loss, log_pdf, sample)


# @jax.jit
@partial(jax.jit, static_argnums=(0,1,))
def train_step(loss, optimizer, params, opt_state, batch):
    def step(params, opt_state, batch):
        batch = jax.numpy.hstack(batch)
        nll, grads = jax.value_and_grad(loss)(params.fast, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)

        return nll, optax.apply_updates(params, updates), opt_state

    return step(params, opt_state, batch)


@partial(jax.jit, static_argnums=(0,))
def valid_step(loss, params, batch):
    def step(params, batch):
        batch = jax.numpy.hstack(batch)
        nll = loss(params.fast, batch)
        return (nll,)

    return step(params, batch)