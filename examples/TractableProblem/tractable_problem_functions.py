import jax
import jax.numpy as np
from lbi.priors import SmoothedBoxPrior



def _calc_vars(theta: np.array):
    """
    Calculate the mean and variance of the posterior.
    """
    # Calculate the mean and variance of the posterior.
    theta_0, theta_1, theta_2, theta_3, theta_4 = theta
    mu = np.stack([theta_0, theta_1])
    s_1 = theta_2 ** 2
    s_2 = theta_3 ** 2
    rho = np.tanh(theta_4)
    Sigma = np.array([[s_1 ** 2, rho * s_1 * s_2], [rho * s_1 * s_2, s_2 ** 2]])
    return mu, Sigma

def log_likelihood(x: np.array, theta: np.array):
    """
    Calculate the log likelihood of the data given the posterior.
    """
    assert x.shape == (4, 2), "x must be a 4x2 array"
    assert theta.ndim == 1 and len(theta) == 5, "theta must be a 1D array of length 5"
    mu, Sigma = _calc_vars(theta)
    
    return -1 / 2 * np.sum(np.log(np.linalg.det(Sigma))) - 1 / 2 * np.sum(
        (x - mu) @ np.linalg.inv(Sigma) @ (x - mu).T
    )

# sample from multi-variate gaussian
def sample(rng, theta: np.array, n_samples: int):
    """
    Sample from the posterior.
    """
    assert theta.ndim == 1 and len(theta) == 5, "theta must be a 1D array of length 5"
    mu, Sigma = _calc_vars(theta)
    samples = jax.random.multivariate_normal(rng, mu, Sigma, shape=(4 * n_samples, 1))
    samples = np.reshape(samples, (n_samples, 4, -1))
    return samples


if __name__ == "__main__":
    import numpy as onp
    import numpyro
    import corner
    import matplotlib.pyplot as plt
    import itertools

    seed = 1234567890
    theta_dim = 5
    rng, _, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)
    
    true_theta = np.array([0.7, -2.9, -1.0, -0.9, 0.6])
    observation = sample(rng, true_theta, n_samples=1)[0]
    
    
    lst = np.array(list(itertools.product([0, 1], repeat=theta_dim)))
    lst = (-1) ** lst
    lst = lst
    num_chains = len(lst)

    
    init_theta = [true_theta*row for i, row in enumerate(lst)]
    init_theta = np.stack(init_theta)
    init_theta = init_theta + 0.1*jax.random.normal(rng, init_theta.shape)

    print("Sample from the posterior w/ hamiltonian monte carlo")

    log_prior = SmoothedBoxPrior(lower=-3., upper=3., sigma=0.01)

    def wrapper_log_posterior(theta):
         log_post = log_likelihood(observation, theta) + log_prior(theta).sum()
         return -log_post

    nuts_kernel = numpyro.infer.NUTS(
        potential_fn=wrapper_log_posterior,
    )
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_samples=15000,
        num_warmup=15000,
        num_chains=num_chains,
    )

    mcmc.run(hmc_rng, init_params=init_theta)
    mcmc.print_summary()
    corner.corner(
        onp.array(mcmc.get_samples()),
        truths=true_theta,
        bins=75,
        range=[(-3, 3) for i in range(init_theta.shape[-1])],
        smooth1d=(1.),
    )
    plt.show()
