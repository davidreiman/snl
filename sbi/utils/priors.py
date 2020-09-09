import numpy as np
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
    

class Uniform(Prior):
    """
    A uniform-distributed prior.
    """
    def __init__(self, lb, ub):
        self.lb = float(lb)
        self.ub = float(ub)
        assert self.ub > self.lb
    
    def sample(self, n_samples):
        return np.random.uniform(self.lb, self.ub, size=[n_samples])
    
    def log_prob(self, x):
        prob = (1. / (self.ub - self.lb)) * np.ones_like(x)
        mask = np.where((x < self.lb) | (x >= self.ub))
        log_prob = np.log(prob)
        log_prob[mask] = -np.inf
        return log_prob