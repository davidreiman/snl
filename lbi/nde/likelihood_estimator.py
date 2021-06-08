import torch
from .base import NeuralDensityEstimator


class NeuralLikelihoodEstimator(NeuralDensityEstimator):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        pass

    def log_prob(self, data, context):
        return self.model.log_prob(data.to(self.device), context.to(self.device))

    def sample(self, n_samples, context=None):
        return self.model.sample(n_samples, context=context.to(self.device))

    def _loss(self, data, context):
        loss = -self.log_prob(data.to(self.device), context.to(self.device)).mean()
        return loss
