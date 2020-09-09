import os
import time
import torch
import torch.nn as nn
import emcee
import corner
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sbi.utils import is_notebook


class SequentialNeuralLikelihoodRatio:
    """
    A class for likelihood-free inference via Sequential Neural Likelihoods (arxiv.org/abs/1805.07226).
    """
    def __init__(self, simulator, priors, obs_data, model, optimizer,
                 scaler=None, obs_truth=None, n_rounds=10, sims_per_model=1,
                 mcmc_walkers=5, mcmc_steps=250, mcmc_discard=50, mcmc_thin=1,
                 n_epochs=20, valid_fraction=0.05, batch_size=50, grad_clip=5.,
                 log_dir='./', device=None):

        """
        Parameters
            simulator: callable
                A wrapper for the simulator, should consume (*, param_dim)
                arrays, possess a `sims_per_model` argument, and return
                a (*, sims_per_model, data_dim) array
            priors: dict
                Dictionary of priors {name: snl.inference.priors.Prior}
            obs_data: np.ndarray (*, data_dim)
                Batch of observed data
            model: snl.models.ConditionalFlow
                Model which will act as the approximate likelihood
            optimizer: torch.optim.Optimizer
                Optimizer for model
            scaler: sklearn.preprocessing.StandardScaler
                Scaler for data if required
            obs_truth: np.ndarray (*, param_dim)
                Batch of observed true params matching obs_data
            n_rounds: int
                Number of SNL rounds
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
            n_epochs: int
                Number of epochs to train per SNL round
            valid_fraction: float
                Fraction of simulations to hold out for validation
            batch_size: int
                Number of training samples to estimate gradient from
            grad_clip: float
                Value at which to clip the gradient norm during training
            log_dir: str
                Location to store models and logs
            device: torch.device
                Device to train model on
        """

        self.simulator = simulator
        self.priors = priors
        self.obs_data = obs_data
        self.model = model
        self.optimizer = optimizer
        self.param_dim = len(priors)
        self.param_names = list(priors.keys())
        self.data_dim = obs_data.shape[1]
        self.scaler = scaler
        self.obs_truth = obs_truth
        self.n_rounds = n_rounds
        self.sims_per_model = sims_per_model
        self.mcmc_steps = mcmc_steps
        self.mcmc_discard = mcmc_discard
        self.mcmc_thin = mcmc_thin
        self.mcmc_walkers = mcmc_walkers
        self.n_epochs = n_epochs
        self.valid_fraction = valid_fraction
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.log_dir = log_dir
        self.model_path = os.path.join(log_dir, 'snl.pt')
        self.best_val_loss = np.inf
        self.notebook = is_notebook()

        self.data = {
            'train_data': np.empty([0, self.data_dim]),
            'train_params': np.empty([0, self.param_dim]),
            'valid_data': np.empty([0, self.data_dim]),
            'valid_params': np.empty([0, self.param_dim])}

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.scaler is not None:
            obs_data = self.scaler.transform(obs_data)
        self.x0 = torch.from_numpy(obs_data).float().to(self.device)

    def make_loaders(self):
        train_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.data['train_data']).float(),
            torch.from_numpy(self.data['train_params']).float())
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        valid_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.data['valid_data']).float(),
            torch.from_numpy(self.data['valid_params']).float())
        valid_loader = torch.utils.data.DataLoader(
            valid_dset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        return train_loader, valid_loader

    def train(self):
        print(f"Training on {self.data['train_data'].shape[0]:,d} samples. "
              f"Validating on {self.data['valid_data'].shape[0]:,d} samples.")
        train_loader, valid_loader = self.make_loaders()

        self.model.train()
        global_step = 0
        total_loss = 0

        # Train
        pbar = tqdm(range(self.n_epochs))
        for epoch in pbar:
            for data, params in train_loader:
                self.optimizer.zero_grad()
                loss = self.model(data.to(self.device), params.to(self.device))
                loss.backward()
                total_loss += loss.item()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                global_step += 1

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                total_loss = 0
                for i, (data, params) in enumerate(valid_loader):
                    loss = self.model(data.to(self.device), params.to(self.device))
                    total_loss += loss.item()
            val_loss = total_loss / float(i+1)
            if val_loss < self.best_val_loss:
                with open(self.model_path, 'wb') as f:
                    torch.save(self.model.state_dict(), f)
                self.best_val_loss = val_loss
            pbar.set_description(f"Validation Loss: {val_loss:.3f}")
            self.model.train()

    def sample_prior(self, n_samples):
        samples = []
        for p in self.priors.values():
            samples.append(p.sample(n_samples))
        return np.array(samples).T

    def log_prior(self, params):
        params = np.atleast_2d(params).T
        log_prob = np.zeros(params.shape[1])
        for i, p in enumerate(self.priors.values()):
            log_prob = log_prob + p.log_prob(params[i])
        return log_prob

    def log_likelihood(self, params):
        self.model.eval()
        log_prob = []
        with torch.no_grad():
            for context in params:
                context = torch.from_numpy(np.array([context])).float()
                context = context.unsqueeze(0).repeat(self.x0.shape[0], 1).to(self.device)
                log_prob.append(self.model.log_prob(self.x0, context).sum().item())
        log_prob = np.array(log_prob)
        return log_prob

    def log_posterior(self, params, prior_only=False):
        log_prob = self.log_prior(params)
        if not prior_only:
            log_prob = log_prob + self.log_likelihood(params)
        return log_prob

    def mcmc(self, prior_only=False):
        p0 = self.sample_prior(self.mcmc_walkers)
        kwargs = {'prior_only': prior_only}
        sampler = emcee.EnsembleSampler(
            self.mcmc_walkers, self.param_dim, self.log_posterior, kwargs=kwargs)
        print("Running MCMC.")
        start_time = time.time()
        progress = 'notebook' if self.notebook else True
        sampler.run_mcmc(p0, self.mcmc_steps, progress=progress)
        samples = sampler.get_chain()    
        t = time.time() - start_time
        print(f"MCMC complete. Time elapsed: {t//60:.0f}m {t%60:.0f}s.")
        return samples

    def simulate(self, params):
        params = params.reshape([-1, self.param_dim])
        data = self.simulator(params, sims_per_model=self.sims_per_model)
        params = params.repeat(self.sims_per_model, axis=0)
        data = data.reshape([-1, self.data_dim])
        assert params.shape[0] == data.shape[0]

        if self.scaler is not None:
            data = self.scaler.transform(data)

        # Select samples for validation
        n = data.shape[0]
        idx = sklearn.utils.shuffle(np.arange(n))
        m = int(self.valid_fraction * n)
        valid_idx = idx[:m]
        train_idx = idx[m:]

        # Store samples in dictionary
        self.data['train_data'] = np.vstack([self.data['train_data'], data[train_idx]])
        self.data['train_params'] = np.vstack([self.data['train_params'], params[train_idx]])
        self.data['valid_data'] = np.vstack([self.data['valid_data'], data[valid_idx]])
        self.data['valid_params'] = np.vstack([self.data['valid_params'], params[valid_idx]])

    def walker_plot(self, samples):
        fig, axes = plt.subplots(ncols=self.param_dim, nrows=self.mcmc_walkers,
                                 figsize=(self.param_dim, self.mcmc_walkers),
                                 squeeze=False, sharex=True)
        for i in range(self.mcmc_walkers):
            for j in range(self.param_dim):
                s = samples[:, i, j]
                axes[i, j].plot(s)
                if i == 0:
                    axes[i, j].set_title(str(self.param_names[j]))
        plt.show()

    def corner_plot(self, samples):
        fig = corner.corner(
            samples.reshape([-1, self.param_dim]),
            labels=self.param_names,
            smooth=1.0,
            smooth1d=0.25,
            truths=self.obs_truth)
        plt.show()

    def run(self, show_plots=True):
        snl_start = time.time()
        for r in range(self.n_rounds):
            print(f"Round {r+1} of {self.n_rounds}.")
            round_start = time.time()

            # Run MCMC
            prior_only = True if r == 0 else False
            samples = self.mcmc(prior_only)

            # Make walker and corner plots
            if show_plots:
                self.walker_plot(samples)
                self.corner_plot(samples)

            # Discard and thin MCMC samples
            samples = samples[self.mcmc_discard::self.mcmc_thin]

            # Simulate and store data
            self.simulate(samples)

            # Train flow on new + old simulations
            self.train()

            t = time.time() - round_start
            total_t = time.time() - snl_start
            print(f"Round {r+1} complete. Time elapsed: {t//60:.0f}m {t%60:.0f}s. "
                  f"Total time elapsed: {total_t//60:.0f}m {total_t%60:.0f}s.")
            print("===============================================================")
