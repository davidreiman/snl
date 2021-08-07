import jax
import jax.numpy as np
from jax.experimental import stax
import optax
from functools import partial
from .classifier import Classifier


def InitializeClassifier(model_rng, obs_dim, theta_dim, n_layers=5, width=128):
    """
    Initialize a likelihood ratio model.

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

    def loss(params, batch, average=True):
        """binary cross entropy with logits
        taken from jaxchem
        """
        obs, theta, label = batch
        label = label.squeeze()
        # log ratio is the logit of the discriminator
        l_d = logit_d(params, np.hstack([obs, theta])).squeeze()
        max_val = np.clip(-l_d, 0, None)
        L = (
            l_d
            - l_d * label
            + max_val
            + np.log(np.exp(-max_val) + np.exp((-l_d - max_val)))
        )
        if average:
            return np.mean(L)
        return np.sum(L)


    init_random_params, logit_d = Classifier(n_layers=n_layers, width=width)

    if type(model_rng) is int:
        model_rng = jax.random.PRNGKey(model_rng)

    _, initial_params = init_random_params(model_rng, (-1, obs_dim + theta_dim))

    return (initial_params, loss, logit_d)


@partial(jax.jit, static_argnums=(0, 1,))
def train_step(loss, optimizer, params, opt_state, batch):
    def step(params, opt_state, batch):
        nll, grads = jax.value_and_grad(loss)(params.fast, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)

        return nll, optax.apply_updates(params, updates), opt_state

    return step(params, opt_state, batch)


@partial(jax.jit, static_argnums=(0,))
def valid_step(loss, params, batch):
    def step(params, batch):
        nll = loss(params.fast, batch)
        return (nll,)

    return step(params, batch)