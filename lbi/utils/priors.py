import numpy as np
import torch
from torch.distributions import Independent, Uniform
from abc import ABC, abstractmethod


class Prior(ABC):
    """
    An abstract class for implementing priors.
    """
    @abstractmethod
    def sample(self, n_samples):
        """
        Sample from prior.
        
        Parameters
            n_samples: int
                Number of samples to draw
        Returns
            samples: np.ndarray
                (n_samples,) array of samples
        """
        pass
    
    @abstractmethod
    def log_prob(self, x):
        """
        Compute log probability of x.
        
        Parameters
            x: np.ndarray
                Array of values to compute the log probability of
        Returns
            log_prob: np.ndarray
                Array of log probabilities (same size as x)
        """
        pass
