import torch
from .base import NeuralDensityEstimator


class NeuralLikelihoodEstimator(NeuralDensityEstimator):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        pass

    def log_prob(self, data, context):
        return self.model.log_prob(data, context)

    def sample(self, n_samples, context=None):
        return self.model.sample(n_samples, context=context)

    def _loss(self, data, context):
        loss = -self.log_prob(data, context).mean()
        return loss
