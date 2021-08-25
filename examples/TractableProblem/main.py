import jax
import jax.numpy as np
import numpy as onp
import optax
from lbi.prior import SmoothedBoxPrior
from lbi.dataset import getDataLoaderBuilder
from lbi.diagnostics import MMD, ROC_AUC, LR_ROC_AUC
from lbi.sequential.sequential import sequential
from lbi.models.base import get_train_step, get_valid_step
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
num_layers = 5
hidden_dim = 512

# Optimizer hyperparmeters
max_norm = 1e-3
learning_rate = 3e-4
weight_decay = 1e-1
sync_period = 5
slow_step_size = 0.5

# Train hyperparameters
nsteps = 250000
patience = 500
eval_interval = 100

# Sequential hyperparameters
num_rounds = 1
num_initial_samples = 100000
num_samples_per_round = 1000
num_chains = 1

# --------------------------
# Create logger
from trax.jaxboard import SummaryWriter

experiment_name = datetime.datetime.now().strftime("%s")
experiment_name = f"{model_type}_{experiment_name}"
logger = SummaryWriter("runs/" + experiment_name)
# logger = None


# --------------------------
# set up simulation and observables
simulate, obs_dim, theta_dim = get_simulator()

# set up true model for posterior inference test
true_theta = np.array([0.7, -2.9, -1.0, -0.9, 0.6])
X_true = simulate(rng, true_theta, num_samples_per_theta=1)

data_loader_builder = getDataLoaderBuilder(
    sequential_mode=model_type,
    batch_size=128,
    train_split=0.95,
    num_workers=0,
    add_noise=False,
)

# --------------------------
# set up prior
log_prior, sample_prior = SmoothedBoxPrior(
    theta_dim=theta_dim, lower=-3.0, upper=3.0, sigma=0.02
)

# TODO: Package model, optimizer, trainer initialization into a function

# --------------------------
# Create model
if model_type == "classifier":
    model_params, loss, log_pdf = InitializeClassifier(
        model_rng=model_rng,
        obs_dim=obs_dim,
        theta_dim=theta_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
else:
    model_params, loss, (log_pdf, sample) = InitializeFlow(
        model_rng=model_rng,
        obs_dim=obs_dim,
        theta_dim=theta_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )

# --------------------------
# Create optimizer
optimizer = optax.chain(
    # Set the parameters of Adam optimizer
    optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    ),
    optax.adaptive_grad_clip(max_norm),
)
# optimizer = optax.lookahead(
#     fast_optimizer, sync_period=sync_period, slow_step_size=slow_step_size
# )

# model_params = optax.LookaheadParams.init_synced(model_params)
opt_state = optimizer.init(model_params)


# --------------------------
# Create trainer

train_step = get_train_step(loss, optimizer)
valid_step = get_valid_step({"valid_loss": loss})

trainer = getTrainer(
    train_step,
    valid_step=valid_step,
    nsteps=nsteps,
    eval_interval=eval_interval,
    patience=patience,
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
    num_rounds=num_rounds,
    num_initial_samples=num_initial_samples,
    num_samples_per_round=num_samples_per_round,
    num_samples_per_theta=1,
    num_chains=num_chains,
    logger=logger,
)


def potential_fn(theta):
    if len(theta.shape) == 1:
        theta = theta[None, :]
    log_post = (
        -log_pdf(
            model_params.fast if hasattr(model_params, "fast") else model_params,
            X_true,
            theta,
        )
        - log_prior(theta)
    )
    return log_post.sum()


num_chains = 128
init_theta = sample_prior(rng, num_samples=num_chains)

mcmc = hmc(
    rng,
    potential_fn,
    init_theta,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=True,
    step_size=1e0,
    max_tree_depth=6,
    num_warmup=2000,
    num_samples=2000,
    num_chains=num_chains,
)
mcmc.print_summary()

theta_samples = mcmc.get_samples(group_by_chain=False).squeeze()

theta_dim = theta_samples.shape[-1]
true_theta = onp.array([0.7, -2.9, -1.0, -0.9, 0.6])

corner.corner(
    onp.array(theta_samples),
    range=[(-3, 3) for i in range(theta_dim)],
    truths=true_theta,
    bins=75,
    smooth=(1.0),
    smooth1d=(1.0),
)

if hasattr(logger, "plot"):
    logger.plot(f"Final Corner Plot", plt, close_plot=True)
else:
    plt.show()

data = simulate(rng, theta_samples, num_samples_per_theta=1)

if model_type == "classifier":
    fpr, tpr, auc = LR_ROC_AUC(
        rng,
        model_params,
        log_pdf,
        data,
        theta_samples,
        data_split=0.05,
    )
else:
    model_samples = sample(rng, model_params, theta_samples)
    fpr, tpr, auc = ROC_AUC(
        rng,
        data,
        model_samples,
)

# Optimal discriminator
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc)
plt.plot(
    np.linspace(0, 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black"
)
plt.legend(loc="lower right")

if hasattr(logger, "plot"):
    logger.plot(f"ROC", plt, close_plot=True)
else:
    plt.show()

logger.close()
