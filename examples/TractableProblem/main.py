import jax
import jax.numpy as np
import numpy as onp
import optax
import torch
from lbi.prior import SmoothedBoxPrior
from lbi.sequential.sequential import sequential
from lbi.models.flows import InitializeFlow
from lbi.models.classifier import InitializeClassifier
from lbi.trainer import getTrainer
from tractable_problem_functions import get_simulator

import matplotlib.pyplot as plt
import datetime

# --------------------------
model_type = "classifier"  # "classifier" or "flow"
mssm_model = "cMSSM"

seed = 1234
rng, model_rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)

# Model hyperparameters
n_layers = 4
width = 128

# Optimizer hyperparmeters
max_norm = 1e-3
learning_rate = 1e-1
sync_period = 5
slow_step_size = 0.5


# --------------------------
# Create logger
# from trax.jaxboard import SummaryWriter
# experiment_name = datetime.datetime.now().strftime("%s")
# experiment_name = f"{model_type}_{experiment_name}"
# logger = SummaryWriter("runs/" + experiment_name)
logger = None
# --------------------------
# set up simulation and observables
simulate, obs_dim, theta_dim = get_simulator()

# set up true model for posterior inference test
true_theta = np.array([0.7, -2.9, -1.0, -0.9, 0.6])
x_obs = simulate(rng, true_theta, n_samples_per_theta=1)

# --------------------------
# set up prior
log_prior, sample_prior = SmoothedBoxPrior(
    theta_dim=theta_dim, lower=-3.0, upper=3.0, sigma=0.02
)

# --------------------------
# Create model
if model_type == "classifier":
    initial_params, loss, log_pdf, train_step, valid_step = InitializeClassifier(
        model_rng=model_rng,
        obs_dim=obs_dim,
        theta_dim=theta_dim,
        n_layers=n_layers,
        width=width,
    )
else:
    initial_params, loss, (log_pdf, sample), train_step, valid_step = InitializeFlow(
        model_rng=model_rng,
        obs_dim=obs_dim,
        theta_dim=theta_dim,
        n_layers=n_layers,
    )


# Create optimizer
fast_optimizer = optax.chain(
    # Set the parameters of Adam optimizer
    optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-8),
    optax.adaptive_grad_clip(max_norm),
)
optimizer = optax.lookahead(
    fast_optimizer, sync_period=sync_period, slow_step_size=slow_step_size
)

model_params = optax.LookaheadParams.init_synced(initial_params)
opt_state = optimizer.init(model_params)


# --------------------------
# Create trainer

trainer = getTrainer(
    loss,
    optimizer,
    train_step,
    valid_step=valid_step,
    nsteps=10000,
    eval_interval=100,
    logger=logger,
    train_kwargs=None,
    valid_kwargs=None,
)

# Train model sequentially
model_params, Theta_post = sequential(
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
)
# Run HMC
num_chains = 1
_observation, _theta = TrainSet.transform(
    observation.repeat(TrainSet.theta.shape[0], 1), TrainSet.theta
)
lps = log_pdf(trained_params.slow, np.hstack([_observation, _theta]))
init_theta = np.array(_theta[np.argmax(lps).item()])

print("Sample from the posterior w/ hamiltonian monte carlo")
mcmc = hmc(
    hmc_rng,
    log_pdf,
    trained_params.slow,
    _observation[:1],
    init_theta,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=True,
    step_size=1e0,
    max_tree_depth=12,
    num_warmup=5000,
    num_samples=5000,
    num_chains=num_chains,
    eps_logit=TrainSet.eps_logit,
    prior="smooth_uniform",
    variance=0.01,
)

mcmc.print_summary()
hmc_samples = mcmc.get_samples()

fig = corners.CornerPlots(
    TrainSet=TrainSet,
    hmc_samples=hmc_samples,
    observation=np.array(observation),
    sigma_obs=sigma_obs,
    sigma_cutoff=2.5,
)

logger.plot("corner", plt, close_plot=False)
fig.savefig(f"plots/{mssm_model}_{model_type}_corner.png")


logger.close()
