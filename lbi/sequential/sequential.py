import jax
import jax.numpy as np
from ..sampler.hmc import hmc


"""
1. Sample theta from prior
2. Simulate X from theta
3. Assemble dataset
4. Train model (flow or classifier should'nt matter)
5. Repeat at 1 but replace prior with posterior
"""


def _sample_theta(rng, sample, n_theta):
    """
    Sample theta from theta_prior

    rng: jax.random.PRNGKey
    sample: function that samples from theta_prior
    n_samples: number of samples to draw
    """
    return sample(rng, n_theta)


def _simulate_X(rng, simulate, theta, n_samples_per_theta: int = 1):
    """
    Simulate X from theta

    simulator: simulator function
    theta: theta
    n_samples_per_theta: number of samples to draw from simulator per theta
    """
    return simulate(rng, theta, n_samples_per_theta=n_samples_per_theta)


def _assemble_dataset(Theta, X):
    """
    Assemble dataset
    """
    return jax.tree_util.tree_multimap(lambda x, y: (x, y), X, Theta)


def _train_model(trainer, model_params, opt_state, train_dataloader, valid_dataloader):
    """
    Train model
    """
    return trainer(model_params, opt_state, train_dataloader, valid_dataloader)


def _sample_posterior(rng, potential_fn, init_theta, num_samples):
    mcmc = hmc(
        rng,
        potential_fn,
        init_theta,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=True,
        step_size=1e0,
        max_tree_depth=12,
        num_warmup=100,
        num_samples=num_samples,
        num_chains=1,
    )
    mcmc.print_summary()

    return mcmc.get_samples()


# @jax.jit
def _sequential_round(
    rng,
    model_params,
    log_pdf,
    log_prior,
    sample_prior,
    simulate,
    opt_state,
    trainer,
    n_theta=1000,
    n_samples_per_theta=1,
    Theta=None,
    X=None,
):
    if Theta is None:
        Theta = _sample_theta(rng, sample_prior, n_theta)
    if X is None:
        X = _simulate_X(rng, simulate, Theta, n_samples_per_theta)

    print(Theta.shape, X.shape)
    print(Theta, X)

    train_dataloader, valid_dataloader = _assemble_dataset(Theta, X)
    model_params = _train_model(
        trainer, model_params, opt_state, train_dataloader, valid_dataloader
    )

    def potential_fn(model_params, theta):
        return -log_pdf(model_params, theta) - log_prior(theta)

    Theta_post = _sample_posterior(rng, potential_fn, X)
    return model_params, X, Theta, Theta_post


def sequential(
    rng,
    model_params,
    log_pdf,
    log_prior,
    sample_prior,
    simulate,
    opt_state,
    trainer,
    n_round=10,
    n_theta=1000,
    n_samples_per_theta=1,
    Theta=None,
    X=None,
    normalilze=None,
):
    for i in range(n_round):
        model_params, X, Theta, Theta_post = _sequential_round(
            rng,
            model_params,
            log_pdf,
            log_prior,
            sample_prior,
            simulate,
            opt_state,
            trainer,
            n_theta=n_theta,
            n_samples_per_theta=n_samples_per_theta,
            Theta=Theta,
            X=X,
        )

    return model_params, Theta_post
