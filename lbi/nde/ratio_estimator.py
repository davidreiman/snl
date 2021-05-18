import torch
from .base import NeuralDensityEstimator


class NeuralRatioEstimator(NeuralDensityEstimator):
    def __init__(self, model):
        """
        Setting criterion as BCEWithLogitsLoss means the network will output logits
            -   i.e. network(x) = log(p/(1 - p)). This means the network will train
                its forward function to be the log_prob of r(x|theta)
        """
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.criterion = torch.nn.BCEWithLogitsLoss()
        pass

    def forward(self, data, context):
        inp = torch.cat([data, context], 1)
        inp = inp.float().to(self.device)
        return self.model(inp).flatten()

    def log_prob(self, data, context):
        """
        With BCEWithLogitsLoss, the log_prob is the forward function
        """
        return self.forward(data, context)

    def _loss(self, data, context):
        def split_in_half(a_list):
            half = len(a_list) // 2
            return a_list[:half], a_list[half:]

        data_a, data_b = split_in_half(data)
        context_a, context_b = split_in_half(context)
        assert (data_a.shape[0] == context_a.shape[0]) & (data_b.shape[0] == context_b.shape[0])

        y_dep_a  = self.forward(data_a, context_a).flatten()
        y_idep_a = self.forward(data_a, context_b).flatten()
        y_dep_b  = self.forward(data_b, context_b).flatten()
        y_idep_b = self.forward(data_b, context_a).flatten()

        loss_a = self.criterion(y_dep_a, torch.ones(y_dep_a.shape[0]).to(self.device)) + \
                 self.criterion(y_idep_a, torch.zeros(y_dep_a.shape[0]).to(self.device))
        loss_b = self.criterion(y_dep_b, torch.ones(y_dep_a.shape[0]).to(self.device)) + \
                 self.criterion(y_idep_b, torch.zeros(y_dep_b.shape[0]).to(self.device))

        loss = loss_a + loss_b

        return loss
