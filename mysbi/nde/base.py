import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class NeuralDensityEstimator(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        pass

    def log_prob(self, data, context):
        return self.model(data, context)

    @abstractmethod
    def _loss(self, data, context):
        raise NotImplementedError

