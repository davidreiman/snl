import jax
import jax.numpy as np
import numpy as onp
import optax
from lbi.prior import SmoothedBoxPrior
from lbi.dataset import getDataLoaderBuilder
from lbi.sequential.sequential import sequential
from lbi.models.flows import InitializeFlow
from lbi.models.classifier import InitializeClassifier
from lbi.trainer import getTrainer
from lbi.sampler import hmc
from tractable_problem_functions import get_simulator

import corner
import matplotlib.pyplot as plt
import datetime

# --------------------------
model_type = "flow"  # "classifier" or "flow"

seed = 1234
rng, model_rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)

# Model hyperparameters
num_layers = 4
width = 32

# Optimizer hyperparmeters
max_norm = 1e-3
learning_rate = 3e-4
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
X_true = simulate(rng, true_theta, num_samples_per_theta=1)

data_loader_builder = getDataLoaderBuilder(
    sequential_mode=model_type,
    batch_size=512,
    train_split=0.9,
    num_workers=0,
    add_noise=False,
)

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
        num_layers=num_layers,
        width=width,
    )
else:
    initial_params, loss, (log_pdf, sample), train_step, valid_step = InitializeFlow(
        model_rng=model_rng,
        obs_dim=obs_dim,
        theta_dim=theta_dim,
        num_layers=num_layers,
    )


# Create optimizer
fast_optimizer = optax.chain(
    # Set the parameters of Adam optimizer
    optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-8),
    # optax.adaptive_grad_clip(max_norm),
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
    nsteps=100000,
    eval_interval=10,
    logger=logger,
    train_kwargs=None,
    valid_kwargs=None,
)

# Train model sequentially
model_params, Theta_post = sequential(
    rng,
    X_true, 
    model_params,
    log_pdf,
    log_prior,
    sample_prior,
    simulate,
    opt_state,
    trainer,
    data_loader_builder,
    num_round=1,
    num_initial_samples=10000,
    num_samples_per_round=2000,
    num_samples_per_theta=1,
    num_chains=64,
)


# logger.close()
