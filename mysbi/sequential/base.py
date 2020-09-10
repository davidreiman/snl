import emcee
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn
import os
from ..utils import is_notebook

class Sequential():
    def __init__(self, priors, obs_data, model, optimizer,
                 simulator=None, param_names=None, num_samples_per_round=250,
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

        self.priors = priors
        self.obs_data = obs_data
        self.model = model
        self.optimizer = optimizer
        self.simulator = simulator
        self.param_dim = priors.mean.shape[0]
        self.param_names = param_names
        self.data_dim = obs_data.shape[1]
        self.scaler = scaler
        self.obs_truth = obs_truth
        self.n_rounds = n_rounds
        self.num_samples_per_round = num_samples_per_round
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
            'train_data': torch.empty([0, self.data_dim]),
            'train_params': torch.empty([0, self.param_dim]),
            'valid_data': torch.empty([0, self.data_dim]),
            'valid_params': torch.empty([0, self.param_dim])}

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.scaler is not None:
            obs_data = obs_data.cpu().numpy()
            obs_data = self.scaler.transform(obs_data)
            obs_data = torch.from_numpy(obs_data).float().to(self.device)
        self.x0 = obs_data

    def add_data(self, data, params):
        if self.scaler is not None:
            data = data.cpu().numpy()
            data = self.scaler.transform(data)
            data = torch.from_numpy(data).float().to(self.device)

        data.to(self.device)

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
            params = torch.from_numpy(params).float().to(self.device)

        params = params.reshape([-1, self.param_dim])
        data = self.simulator(params, sims_per_model=self.sims_per_model)
        params = torch.cat(self.sims_per_model*[params])
        data = data.reshape([-1, self.data_dim])
        assert params.shape[0] == data.shape[0], print(params.shape, data.shape)

        return data

    def make_loaders(self):
        train_dset = torch.utils.data.TensorDataset(
            self.data['train_data'].float(),
            self.data['train_params'].float())
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        valid_dset = torch.utils.data.TensorDataset(
            self.data['valid_data'].float(),
            self.data['valid_params'].float())
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
                loss = self.model._loss(data.to(self.device), params.to(self.device))
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
                    loss = self.model._loss(data.to(self.device), params.to(self.device))
                    total_loss += loss.item()
            val_loss = total_loss / float(1+len(valid_loader))
            if val_loss < self.best_val_loss:
                with open(self.model_path, 'wb') as f:
                    torch.save(self.model.state_dict(), f)
                self.best_val_loss = val_loss
            pbar.set_description(f"Validation Loss: {val_loss:.3f}")
            self.model.train()

    def log_prior(self, params):
        return self.priors.log_prob(params)

    def log_posterior(self, params, prior_only=False):
        if type(params) is np.ndarray:
            params = torch.from_numpy(params).float().to(self.device)

        log_prob = self.log_prior(params)
        if not prior_only:
            params = torch.stack(self.x0.shape[0]*[params])
            log_prob = log_prob + self.model.log_prob(self.x0, params)
        return log_prob


    def _mcmc_log_posterior(self, params, prior_only=False):
        if type(params) is np.ndarray:
            params = torch.from_numpy(params).float().to(self.device)

        log_prob = self.log_prior(params)
        if not prior_only:
            params = torch.stack(self.x0.shape[0]*[params])
            log_prob = log_prob + self.model.log_prob(self.x0, params).item()
        return log_prob.detach()

    def mcmc(self, prior_only=False, clean=True):
        """
        TODO: Update to pyro? Something better for nns than emcee probably

        :param prior_only:
        :return:
        """
        p0 = self.sample_prior(self.mcmc_walkers)
        kwargs = {'prior_only': prior_only}
        sampler = emcee.EnsembleSampler(
            self.mcmc_walkers, self.param_dim, self._mcmc_log_posterior, kwargs=kwargs)
        print("Running MCMC.")
        start_time = time.time()
        progress = 'notebook' if self.notebook else True
        sampler.run_mcmc(p0, self.mcmc_steps, progress=progress)
        samples = sampler.get_chain()
        t = time.time() - start_time
        print(f"MCMC complete. Time elapsed: {t//60:.0f}m {t%60:.0f}s.")
        if clean:
            return self._clean_mcmc_samples(samples)
        return torch.from_numpy(samples).float().to(self.device)

    def _clean_mcmc_samples(self, samples):
        # Discard and thin MCMC samples
        cleaned_samples = samples[self.mcmc_discard::self.mcmc_thin]
        # TODO: Clean up numpy types
        if type(cleaned_samples) is np.ndarray:
            cleaned_samples = torch.from_numpy(cleaned_samples).float().to(self.device)
        return cleaned_samples.view(torch.tensor(cleaned_samples.shape[:2]).prod(), -1)

    def sample_prior(self, num_samples=1000, prior_only=True):
        if prior_only:
            prior_samples = self.priors.sample((num_samples, ))
        else:  # sample from nde
            prior_samples = self.mcmc(clean=True)
            # shuffle to uncorrelate samples
            prior_samples = prior_samples[torch.randperm(prior_samples.shape[0])][:num_samples]

        if type(prior_samples) is np.ndarray:
            prior_samples = torch.from_numpy(prior_samples).float().to(self.device)

        return prior_samples

    def sample_posterior(self, num_samples=1000):
        samples = self.mcmc(prior_only=False)
        cleaned_samples = self._clean_mcmc_samples(samples)
        if type(cleaned_samples) is np.ndarray:
            cleaned_samples = torch.from_numpy(cleaned_samples).float().to(self.device)

        # shuffle to uncorrelate samples
        prior_samples = cleaned_samples[torch.randperm(cleaned_samples.shape[0])][:num_samples]
        return cleaned_samples

    def run(self, show_plots=True):
        # TODO: think of way to take out simulator from this loop when simulator not included
        snl_start = time.time()
        for r in range(self.n_rounds):
            round_start = time.time()
            # sample from nde after first round
            prior_samples = self.sample_prior(num_samples=self.num_samples_per_round,
                                              prior_only=(r == 0))
            if show_plots:
                self.make_plots()

            # Simulate
            sims = self.simulate(prior_samples)
            # Store data
            print(sims.shape, prior_samples.shape)
            self.add_data(sims, prior_samples)

            # Train flow on new + old simulations
            self.train()

            t = time.time() - round_start
            total_t = time.time() - snl_start
            print(f"Round {r+1} complete. Time elapsed: {t//60:.0f}m {t%60:.0f}s. "
                  f"Total time elapsed: {total_t//60:.0f}m {total_t%60:.0f}s.")
            print("===============================================================")

    def make_plots(self):
        pass
