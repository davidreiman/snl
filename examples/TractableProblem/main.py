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
model_type = "classifier"  # "classifier" or "flow"
mssm_model = "cMSSM"

seed = 1234
rng, model_rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)

# Model hyperparameters
num_layers = 4
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
X_true = simulate(rng, true_theta, num_samples_per_theta=1)

data_loader_builder = getDataLoaderBuilder(
    sequential_mode="classifier",
    batch_size=128,
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
    nsteps=1500,
    eval_interval=250,
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
    num_round=3,
    num_initial_samples=1000,
    num_samples_per_round=1000,
    num_samples_per_theta=1,
    num_chains=32,
)

def potential_fn(theta):
    if len(theta.shape) == 1:
        theta = theta[None, :]
        
    model_inputs = np.hstack([X_true, theta])
    log_post = -log_pdf(model_params.slow, model_inputs) - log_prior(theta)
    return log_post.sum()



mcmc = hmc(
    rng,
    potential_fn,
    true_theta,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=True,
    step_size=1e0,
    max_tree_depth=12,
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
)
mcmc.print_summary()
hmc_samples = mcmc.get_samples()

corner.corner(onp.array(hmc_samples), 
              range=[(-3, 3) for i in range(theta_dim)],
              )
plt.show()
# logger.plot("corner", plt, close_plot=False)
# fig.savefig(f"plots/{mssm_model}_{model_type}_corner.png")


# logger.close()
