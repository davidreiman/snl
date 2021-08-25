import jax
import optax
from functools import partial
from .maf import MaskedAffineFlow


def InitializeFlow(
    model_rng,
    obs_dim,
    theta_dim,
    flow_model=None,
    num_layers=5,
    hidden_dim=64,
    **kwargs,
):
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

    def loss(params, inputs, context=None):
        return -log_pdf(params, inputs, context).mean()

    if flow_model is None:
        flow_model = MaskedAffineFlow
    if type(model_rng) is int:
        model_rng = jax.random.PRNGKey(model_rng)

    init_fun = flow_model(num_layers)
    initial_params, log_pdf, sample = init_fun(
        model_rng,
        input_dim=obs_dim,
        context_dim=theta_dim,
        hidden_dim=hidden_dim,
    )
    return initial_params, loss, (log_pdf, sample)
