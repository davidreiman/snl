import jax
import jax.numpy as np
import numpy as onp
import optax
import torch
from mssm.models.jax.flows import (
    InitializeFlow,
)
from mssm.models.jax.classifier import (
    InitializeClassifier,
    train_step,
    valid_step,
)
from mssm.datasets.Hollingsworth import get_dataset, LikelihoodRatioCollate_fn
from mssm.trainer import Train
from mssm.sampler import hmc
from mssm.diagnostics import corners
from trax.jaxboard import SummaryWriter

import matplotlib.pyplot as plt
import datetime

model_type = "classifier"  # "classifier" or "flow"
mssm_model = "cMSSM"
N = 50000
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
experiment_name = datetime.datetime.now().strftime("%s")
experiment_name = f"{model_type}_{experiment_name}"
logger = SummaryWriter("runs/" + experiment_name)

# create observation
observation = torch.tensor([[0.12, 122.0]])
sigma_obs = np.array([0.01, 2.0])

# Load in Hollingsworth dataset
TrainSet, ValidSet = get_dataset(
    N=N,
    sigma_obs=sigma_obs,
    rand_key=rng,
    use_logit=(model_type == "flow"),
    use_minmax=True,
    use_standardization=False,
    mssm_model=mssm_model,
)

obs_dim = TrainSet.obs_dim
theta_dim = TrainSet.theta_dim

train_dataloader = torch.utils.data.DataLoader(
    TrainSet,
    batch_size=512,
    shuffle=True,
    collate_fn=LikelihoodRatioCollate_fn if model_type == "classifier" else None,
)

valid_dataloader = torch.utils.data.DataLoader(
    ValidSet,
    batch_size=4096,
    shuffle=False,
    collate_fn=LikelihoodRatioCollate_fn if model_type == "classifier" else None,
)

# Create model
if model_type == "classifier":
    initial_params, loss, log_pdf = InitializeClassifier(
        model_rng=model_rng,
        obs_dim=obs_dim,
        theta_dim=theta_dim,
        n_layers=n_layers,
        width=width,
    )
    from mssm.models.jax.classifier import train_step, valid_step

else:
    initial_params, loss, log_pdf, sample = InitializeFlow(
        model_rng=model_rng,
        obs_dim=obs_dim,
        theta_dim=theta_dim,
        n_layers=n_layers,
    )
    from mssm.models.jax.flows import train_step, valid_step


# Create optimizer
fast_optimizer = optax.chain(
    # Set the parameters of Adam optimizer
    optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-8),
    optax.adaptive_grad_clip(max_norm),
)
optimizer = optax.lookahead(
    fast_optimizer, sync_period=sync_period, slow_step_size=slow_step_size
)

params = optax.LookaheadParams.init_synced(initial_params)
opt_state = optimizer.init(params)

# Train model with early stopping
trained_params, best_val_loss = Train(
    params,
    loss,
    optimizer,
    opt_state,
    train_dataloader,
    train_step=train_step,
    nsteps=25000,
    eval_interval=500,
    valid_dataloader=valid_dataloader,
    valid_step=valid_step,
    logger=logger if logger is not None else None,
    train_kwargs=None,
    valid_kwargs=None,
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
