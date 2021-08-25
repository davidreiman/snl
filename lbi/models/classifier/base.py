import jax
import jax.numpy as np
from jax.experimental import stax
import optax
from functools import partial
from .classifier import Classifier


def InitializeClassifier(model_rng, obs_dim, theta_dim, hidden_dim=128, num_layers=5, **kwargs):
    """
    Initialize a likelihood ratio model.

    Args:
        model_rng: a jax random number generator
        obs_dim: dimensionality of the observations
        theta_dim: dimensionality of the simulation parameters
        num_layers: number of affine layers in the flow

    Returns:
        initial_params: a list of parameters
        log_pdf: a function from parameters to log-probability of the observations
        sample: a function from parameters to samples of the parameters

    """

    def loss(params, inputs, context, label):
        """binary cross entropy with logits
        taken from jaxchem
        """
        label = label.squeeze()
        # log ratio is the logit of the discriminator
        l_d = logit_d(params, inputs, context).squeeze()
        max_val = np.clip(-l_d, 0, None)
        L = (
            l_d
            - l_d * label
            + max_val
            + np.log(np.exp(-max_val) + np.exp((-l_d - max_val)))
        )
        return np.mean(L)

    init_random_params, logit_d = Classifier(num_layers=num_layers, hidden_dim=hidden_dim)

    if type(model_rng) is int:
        model_rng = jax.random.PRNGKey(model_rng)

    _, initial_params = init_random_params(model_rng, (-1, obs_dim + theta_dim))

    return initial_params, loss, logit_d
