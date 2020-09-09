import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class NeuralDensityEstimator(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def log_prob(self):
        raise NotImplementedError

    @abstractmethod
    def _loss(self):
        raise NotImplementedError

