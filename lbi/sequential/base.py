import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn
import os
from ..utils import is_notebook, Logger, get_gradient_norm, prep_log_dir
from pyro.infer import MCMC, NUTS, Predictive
import pickle


class Sequential():
    def __init__(self,
                 priors, obs_data, param_dim, model, optimizer,
                 simulator=None,
                 param_names=None,
                 num_initial_samples=250,
                 num_samples_per_round=250,
                 summary_interval=50,
                 validation_interval=250,
                 scaler=None,
                 param_scaler=None,
                 obs_truth=None,
                 n_rounds=10,
                 reset_every_round=False,
                 sims_per_model=1,
                 mcmc_walkers=5,
                 mcmc_steps=250,
                 mcmc_discard=50,
                 mcmc_thin=1,
                 max_n_epochs=200,
                 valid_fraction=0.15,
                 batch_size=256,
                 num_workers=0,
                 grad_clip=5.,
                 patience=20,
                 log_dir='./runs/test_run/',
                 settings_path=None,
                 logger=None,
                 hparam_dict=None,
                 metric_dict=None,
                 scalar_funcs=None,
                 progress=False,
                 **kwargs):
        """
        Parameters
            simulator: callable
                A wrapper for the simulator, should consume (*, param_dim)
                arrays, possess a `sims_per_model` argument, and return
                a (*, sims_per_model, data_dim) array
            priors: dict
                Dictionary of priors {name: lbi.inference.priors.Prior}
            obs_data: np.ndarray (*, data_dim)
                Batch of observed data
            model: lbi.models.ConditionalFlow
                Model which will act as the approximate likelihood
            optimizer: torch.optim.Optimizer
                Optimizer for model
            scaler: sklearn.preprocessing.StandardScaler
                Scaler for data if required
            obs_truth: np.ndarray (*, param_dim)
                Batch of observed true params matching obs_data
            n_rounds: int
                Number of SNL rounds
            num_samples_per_round: int
                Number of samples to generate in each SNL round
            summary_interval: int
                Calculate training loss after this many steps
            validation_interval: int
                Calculate validation loss after this many steps
            sims_per_model: int
                Number of simulations to generate per MCMC sample
            mcmc_walkers: int
                Number of independent MCMC walkers
            mcmc_steps: int
                Number of MCMC steps per round per walker
            mcmc_discard: int
                Number of MCMC steps per round per walker to discard
            mcmc_thin: int
                Take every `mcmc_thin` samples from MCMC chain
            max_n_epochs: int
                Number of epochs to train per SNL round
            valid_fraction: float
                Fraction of simulations to hold out for validation
            batch_size: int
                Number of training samples to estimate gradient from
            grad_clip: float
                Value at which to clip the gradient norm during training
            log_dir: str
                Location to store models and logs
            settings_path: str
                Location to read model and snl settings
            logger: ..utils.Logger or torch.utils.tensorboard.SummaryWriter object
                Should work with both basic Logger and tensorboard.SummaryWriter
                Saves training and validation losses
            hparam_dict: dict
                dictionary of model hyperparameters to save with logger
            metric_dict: dict
                dictionary of metric functions evaluated at the end of training
                which take Sequential.model as argument and return scalar
            scalar_funcs: dict
                dictionary of metric functions evaluated every summary interval
                which take Sequential as argument and return scalars
            device: torch.device
                Device to train model on. Default to model.device.
        """
        self.priors = priors
        self.obs_data = obs_data
        self.param_dim = param_dim
        self.model = model
        self.model.eval()  # the only time model should be in .train() is during training
        self.optimizer = optimizer
        self.simulator = simulator
        self.param_names = param_names
        self.data_dim = obs_data.shape[1]
        self.scaler = scaler
        self.param_scaler = param_scaler
        self.obs_truth = obs_truth
        self.n_rounds = n_rounds
        self.reset_every_round = reset_every_round
        if self.reset_every_round:
            self._base_model_params = model.state_dict().copy()
            self._base_optimizer_params = optimizer.state_dict().copy()
        self.num_initial_samples = num_initial_samples
        self.num_samples_per_round = num_samples_per_round
        self.summary_interval = summary_interval
        self.validation_interval = validation_interval
        self.sims_per_model = sims_per_model
        self.mcmc_steps = mcmc_steps
        self.mcmc_discard = mcmc_discard
        self.mcmc_thin = mcmc_thin
        self.mcmc_walkers = mcmc_walkers
        self.max_n_epochs = max_n_epochs
        self.patience = patience
        self.valid_fraction = valid_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grad_clip = grad_clip
        if settings_path is None:
            self.log_dir = None
            self.log_dir = log_dir
        else:
            self.log_dir = prep_log_dir(log_dir=log_dir, settings_path=settings_path)
        # default to bare-bones logger
        self.logger = logger if logger is not None else Logger(log_dir=log_dir)
        self.model_path = os.path.join(log_dir, 'model.pt')
        self.hparam_dict = hparam_dict if hparam_dict is not None else {}
        self.metric_dict = metric_dict if metric_dict is not None else {}
        self.scalar_funcs = scalar_funcs if scalar_funcs is not None else {"Misc/grad_norm": lambda x: get_gradient_norm(x.model.model)}
        self.progress = progress
        self.best_val_loss = np.inf
        self.notebook = is_notebook()
        self.device = model.device

        self.data = {
            'train_data': torch.empty([0, self.data_dim]).cpu(),
            'train_params': torch.empty([0, self.param_dim]).cpu(),
            'valid_data': torch.empty([0, self.data_dim]).cpu(),
            'valid_params': torch.empty([0, self.param_dim]).cpu()}

        if self.scaler is not None:
            with open(f'{self.log_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            obs_data = obs_data.cpu().numpy()
            obs_data = self.scaler.transform(obs_data)
            obs_data = torch.from_numpy(obs_data).float()
        self.x0 = obs_data

    def add_data(self, data, params):
        if self.scaler is not None:
            data = data.cpu().numpy()
            data = self.scaler.transform(data)
            data = torch.from_numpy(data).float()
        if self.param_scaler is not None:  # TODO: make this more general to work with other scalers
            mean = torch.from_numpy(self.param_scaler.mean_).float()
            scale = torch.from_numpy(self.param_scaler.scale_).float()
            params = (params - mean)/scale

        # Select samples for validation
        n = data.shape[0]
        idx = sklearn.utils.shuffle(np.arange(n))
        m = int(self.valid_fraction * n)
        valid_idx = idx[:m]
        train_idx = idx[m:]

        # Store samples in dictionary
        self.data['train_data'] = torch.cat([self.data['train_data'], data[train_idx]], dim=0)
        self.data['train_params'] = torch.cat([self.data['train_params'], params[train_idx]], dim=0)
        self.data['valid_data'] = torch.cat([self.data['valid_data'], data[valid_idx]], dim=0)
        self.data['valid_params'] = torch.cat([self.data['valid_params'], params[valid_idx]], dim=0)

    def simulate(self, params):
        # TODO: Clean up numpy types
        if type(params) is np.ndarray:
            params = torch.from_numpy(params).float()

        params = params.reshape([-1, self.param_dim])
        params = torch.cat(self.sims_per_model * [params])

        data = self.simulator(params, sims_per_model=self.sims_per_model)
        if type(data) is np.ndarray:
            # TODO: Make sure this works with cuda multiprocessing
            data = torch.from_numpy(data)
        data = data.reshape([-1, self.data_dim])
        assert params.shape[0] == data.shape[0], print(params.shape, data.shape)

        return data, params

    def make_loaders(self):
        train_dset = torch.utils.data.TensorDataset(
            self.data['train_data'].float(),
            self.data['train_params'].float())
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

        valid_dset = torch.utils.data.TensorDataset(
            self.data['valid_data'].float(),
            self.data['valid_params'].float())
        valid_loader = torch.utils.data.DataLoader(
            valid_dset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)

        return train_loader, valid_loader

    def loss_func(self, data, params):
        return self.model._loss(data.to(self.device), params.to(self.device))

    def train(self, global_step=0):

        print(f"Training on {self.data['train_data'].shape[0]:,d} samples. "
              f"Validating on {self.data['valid_data'].shape[0]:,d} samples.")
        train_loader, valid_loader = self.make_loaders()

        self.model.train()
        total_loss = 0
        epochs_without_improvement = 0
        # Train
        if self.progress:
            pbar = tqdm(range(self.max_n_epochs))
        else:
            pbar = range(self.max_n_epochs)
        for epoch in pbar:
            for data, params in train_loader:
                self.optimizer.zero_grad()
                loss = self.loss_func(data.to(self.device), params.to(self.device))
                loss.backward()
                total_loss += loss.item()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                global_step += 1
                # Report training loss
                if global_step % self.summary_interval == 0:
                    train_loss = total_loss / float(self.summary_interval)
                    self.logger.add_scalar("Losses/train", train_loss, global_step=global_step)
                    for label, func in self.scalar_funcs.items():
                        self.logger.add_scalar(label, func(self), global_step=global_step)
                    total_loss = 0

                # Evaluate and report validation loss
                if global_step % self.validation_interval == 0:
                    self.model.eval()
                    with torch.no_grad():
                        total_loss = 0
                        i = 0
                        for i, (data, params) in enumerate(valid_loader):
                            loss = self.model._loss(data.to(self.device), params.to(self.device))
                            total_loss += loss.item()
                    val_loss = total_loss / float(1 + i)
                    if val_loss < self.best_val_loss:
                        with open(self.model_path, 'wb') as f:
                            torch.save(self.model.state_dict(), f)
                        self.best_val_loss = val_loss
                    else:
                        epochs_without_improvement += 1
                    if self.progress:
                        pbar.set_description(f"Validation Loss: {val_loss:.3f}")
                    self.logger.add_scalar("Losses/valid", val_loss, global_step=global_step)
                    total_loss = 0
                self.model.train()

            if epochs_without_improvement > self.patience:
                print(f"Early stopped after {epoch} epochs")
                break
        self.model.eval()
        return global_step

    def log_prior(self, params):
        return self.priors.log_prob(params)

    def log_posterior(self, params, prior_only=False):
        if type(params) is np.ndarray:
            params = torch.from_numpy(params).float().to(self.device)

        log_prob = self.log_prior(params)
        if not prior_only:
            params = torch.stack(self.x0.shape[0] * [params])
            log_prob = log_prob + self.model.log_prob(self.x0, params)
        return log_prob

    def hmc(self, num_samples=50, walker_steps=200, burn_in=100):
        def model_wrapper(param_dict):
            if param_dict is not None:
                # TODO: Figure out if there's a way to pass params without dict
                params = param_dict['params']
                log_prob = self.log_prior(params.to(self.device))

                if self.param_scaler is not None:
                    mean = torch.from_numpy(self.param_scaler.mean_).float()
                    scale = torch.from_numpy(self.param_scaler.scale_).float()
                    params = (params - mean)/scale

                log_prob += self.model.log_prob(self.x0, params.to(self.device))
                return -log_prob

        initial_params = self.priors.sample((1,))
        nuts_kernel = NUTS(potential_fn=model_wrapper, adapt_step_size=True)
        mcmc = MCMC(nuts_kernel, num_samples=walker_steps, warmup_steps=burn_in,
                    initial_params={"params": initial_params})
        mcmc.run(self.x0)
        return mcmc.get_samples(num_samples)['params'].view(num_samples, -1)

    def sample_prior(self, num_samples=1000, prior_only=True):
        if prior_only:
            prior_samples = self.priors.sample((num_samples,))
        else:  # sample from nde
            prior_samples = self.hmc(num_samples=num_samples)

        if type(prior_samples) is np.ndarray:
            prior_samples = torch.from_numpy(prior_samples).float().to(self.device)

        return prior_samples

    def sample_posterior(self, num_samples=1000, walker_steps=200, burn_in=100):
        samples = self.hmc(num_samples=num_samples, walker_steps=walker_steps, burn_in=burn_in)
        return samples

    def train_round(self, global_step, show_plots=True):
        # sample from nde after first round
        if global_step == 0:
            prior_samples = self.sample_prior(num_samples=self.num_initial_samples,
                                              prior_only=True)
        else:  # sample posterior
            prior_samples = self.sample_prior(num_samples=self.num_samples_per_round,
                                              prior_only=False)
        if show_plots:
            self.make_plots()

        # Simulate
        sims, prior_samples = self.simulate(prior_samples)
        # Store data
        self.add_data(sims, prior_samples)

        # Train flow on new + old simulations
        try:
            self.best_val_loss = np.inf
            global_step = self.train(global_step=global_step)
        except KeyboardInterrupt:
            pass

        return global_step

    def run(self, show_plots=True):
        """
        metric_dict should be a dictionary containing functions which take the snre class as an argument
        """
        # TODO: think of way to take out simulator from this loop when simulator not included
        snl_start = time.time()
        global_step = 0
        for r in range(self.n_rounds):
            round_start = time.time()
            if self.reset_every_round:
                self.model.load_state_dict(self._base_model_params)
                self.optimizer.load_state_dict(self._base_optimizer_params)

            global_step = self.train_round(global_step)

            t = time.time() - round_start
            total_t = time.time() - snl_start
            print(f"Round {r + 1} complete. Time elapsed: {t // 60:.0f}m {t % 60:.0f}s. "
                  f"Total time elapsed: {total_t // 60:.0f}m {total_t % 60:.0f}s.")
            print("===============================================================")

        for k, f in self.metric_dict.items():
            if callable(f):  # if f is a function
                self.metric_dict.update({k: f(self)})
        self.logger.add_hparams(hparam_dict=self.hparam_dict, metric_dict=self.metric_dict)
        if hasattr(self.logger, "log_asset"):  # comet.ml experiment tracker
            print("file path", f"{self.log_dir}/model.pt")
            self.logger.log_asset(f"{self.log_dir}/model.pt", file_name='model.pt')
            self.logger.log_asset(f"{self.log_dir}/scaler.pkl", file_name='scaler.pkl')
        self.logger.close()

    def make_plots(self):
        pass
