import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import nsflow.nn as nn_
import nsflow.utils as utils
from nsflow.nde import distributions, flows, transforms


class ConditionalFlow(nn.Module):
    """A conditional rational quadratic neural spline flow."""
    def __init__(self, dim, context_dim, transform_type, n_layers, hidden_units,
        n_blocks, dropout, use_batch_norm, tails, tail_bound, n_bins,
        min_bin_height, min_bin_width, min_derivative, unconditional_transform,
        encoder=None):
        super().__init__()

        self.dim = dim
        self.context_dim = context_dim
        self.transform_type = transform_type
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.tails = tails
        self.tail_bound = tail_bound
        self.n_bins = n_bins
        self.min_bin_height = min_bin_height
        self.min_bin_width = min_bin_width
        self.min_derivative = min_derivative
        self.unconditional_transform = unconditional_transform
        self.encoder = encoder

        distribution = distributions.StandardNormal([dim])
        transform = transforms.CompositeTransform([
            self.create_transform(self.transform_type)
            for _ in range(self.n_layers)])
        self.flow = flows.Flow(transform, distribution)

    def create_transform(self, type):
        """Create invertible rational quadratic transformations."""
        linear = transforms.RandomPermutation(features=self.dim)
        if type == 'coupling':
            base = transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=utils.create_mid_split_binary_mask(features=self.dim),
                transform_net_create_fn=lambda in_features, out_features:
                    nn_.ResidualNet(
                        in_features=in_features,
                        out_features=out_features,
                        context_features=self.context_dim,
                        hidden_features=self.hidden_units,
                        num_blocks=self.n_blocks,
                        dropout_probability=self.dropout,
                        use_batch_norm=self.use_batch_norm,
                    ),
                tails=self.tails,
                tail_bound=self.tail_bound,
                num_bins=self.n_bins,
                min_bin_height=self.min_bin_height,
                min_bin_width=self.min_bin_width,
                min_derivative=self.min_derivative,
                apply_unconditional_transform=self.unconditional_transform,
            )
        elif type == 'autoregressive':
            base = transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.dim,
                hidden_features=self.hidden_units,
                context_features=self.context_dim,
                num_bins=self.n_bins,
                tails=self.tails,
                tail_bound=self.tail_bound,
                num_blocks=self.n_blocks,
                use_residual_blocks=True,
                random_mask=False,
                activation=F.relu,
                dropout_probability=self.dropout,
                use_batch_norm=self.use_batch_norm,
        )
        else:
            raise ValueError(f'Transform type {self.transform_type} unavailable.')
        t = transforms.CompositeTransform([linear, base])
        return t

    def log_prob(self, inputs, context=None):
        """Forward pass in density estimation direction.

        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context."""
        if self.encoder is not None and context is not None:
            context = self.encoder(context)
        log_prob = self.flow.log_prob(inputs, context)
        return log_prob

    def forward(self, inputs, context=None):
        """Forward pass to negative log likelihood (NLL).

        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context."""
        log_prob = self.log_prob(inputs, context)
        loss = -torch.mean(log_prob)
        return loss

    def sample(self, n_samples, context=None):
        """Draw samples from the conditional flow.

        Args:
            n_samples (int): Number of samples to draw.
            context (torch.Tensor): [context_dim] tensor of conditioning info."""
        if context is not None:
            context = context.unsqueeze(0)
            if self.encoder is not None:
                context = self.encoder(context)
            context = context.expand(n_samples, -1)
            noise = self.flow._distribution.sample(1, context).squeeze(1)
        else:
            noise = self.flow._distribution.sample(n_samples)
        samples, log_prob = self.flow._transform.inverse(noise, context)
        return samples, log_prob


class MixtureDensityNetwork(nn.Module):
    """A Gaussian mixture density network."""
    def __init__(self, dim, context_dim, n_components,
                n_layers, hidden_units, full_cov, 
                alpha=1e-1, use_batch_norm=True):
        super().__init__()
        
        self.dim = dim
        self.context_dim = context_dim
        self.n_components = n_components
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.full_cov = full_cov
        self.use_batch_norm = use_batch_norm
        self.alpha = alpha
        
        if full_cov:
            self.sizes = [self.n_components, self.dim*self.n_components, 
                         ((self.dim**2 + self.dim) // 2)*self.n_components]
        else:
            self.sizes = [self.n_components, self.dim*self.n_components,
                         self.dim*self.n_components]
            
        self.network = nn.ModuleList()
        self.network.append(self._layer(context_dim, hidden_units))
        self.network.extend([
            self._layer(hidden_units, hidden_units)
            for _ in range(n_layers-1)])
        self.network.append(nn.Linear(hidden_units, sum(self.sizes)))
    
    def _layer(self, in_features, out_features):
        l = nn.Sequential(nn.Linear(in_features, out_features), nn.ELU())
        if self.use_batch_norm:
            l.add_module("batch_norm", nn.BatchNorm1d(out_features))
        return l
    
    def _network(self, x):
        for l in self.network:
            x = l(x)
        return x
    
    def _forward(self, context):
        bsz = context.size(0)
        out = self._network(context)
        
        weights, means, chol = torch.split(out, self.sizes, 1)
        means = means.view(bsz, self.n_components, self.dim)
        chol = chol.view(bsz, self.n_components, -1)
        chol_mat_shape = [bsz, self.n_components, self.dim, self.dim]
        chol_mat = torch.zeros(*chol_mat_shape, device=context.device)
        
        diag = chol[:, :, :self.dim].exp()
        i = torch.arange(self.dim)
        chol_mat[:, :, i, i] = diag
    
        if self.full_cov:
            tril = chol[:, :, self.dim:]
            tril_indices = torch.tril_indices(self.dim, self.dim, -1)
            chol_mat[:, :, tril_indices[0], tril_indices[1]] = tril
            
        return weights, means, chol_mat
    
    def get_params(self, context):
        weights, means, chol_mat = self._forward(context)
        weights = weights.softmax(1)
        cov = torch.matmul(chol_mat, chol_mat.transpose(-2, -1))
        
        m = torch.diagonal(cov, dim1=-1, dim2=-2).sum(-1) / cov.size(-1)
        identity = torch.eye(cov.size(-1))
        target = m[:, :, None, None] * identity

        cov = (1 - self.alpha) * cov + self.alpha * target
        
        diag = torch.diagonal(cov, dim1=-1, dim2=-2).clone()
        i = torch.arange(self.dim)
        cov[:, :, i, i] = torch.clamp(diag, min=1e-3)
        
        return weights, means, cov
    
    def log_prob(self, data, context):
        weights, means, cov = self.get_params(context)
        
        c = D.Categorical(probs=weights)
        n = D.MultivariateNormal(loc=means, covariance_matrix=cov)
        m = D.MixtureSameFamily(c, n)
        
        log_prob = m.log_prob(data)
        return log_prob

    def sample(self, context, n_samples):
        weights, means, cov = self.get_params(context)
        
        c = D.Categorical(probs=weights)
        n = D.MultivariateNormal(loc=means, covariance_matrix=cov)
        m = D.MixtureSameFamily(c, n)
        
        samples = m.sample([n_samples,])
        return samples
    
    def forward(self, data, context):
        log_prob = self.log_prob(data, context)
        return -log_prob.mean()
    

class LowRankMixtureDensityNetwork(nn.Module):
    """A low rank Gaussian mixture density network."""
    def __init__(self, dim, context_dim, n_components,
                 n_layers, hidden_units, rank, 
                 use_batch_norm=True, sigmoid_means=False):
        super().__init__()
        
        self.dim = dim
        self.context_dim = context_dim
        self.n_components = n_components
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.rank = rank
        self.use_batch_norm = use_batch_norm
        self.sigmoid_means = sigmoid_means

        self.sizes = [self.n_components, self.dim*self.n_components, 
                     (self.dim + self.dim*self.rank)*self.n_components]
            
        self.network = nn.ModuleList()
        self.network.append(self._layer(context_dim, hidden_units))
        self.network.extend([
            self._layer(hidden_units, hidden_units)
            for _ in range(n_layers-1)])
        self.network.append(nn.Linear(hidden_units, sum(self.sizes)))
    
    def _layer(self, in_features, out_features):
        l = nn.Sequential(nn.Linear(in_features, out_features), nn.ELU())
        if self.use_batch_norm:
            l.add_module("batch_norm", nn.BatchNorm1d(out_features))
        return l
    
    def _network(self, x):
        for l in self.network:
            x = l(x)
        return x
    
    def _forward(self, context):
        bsz = context.size(0)
        out = self._network(context)
        weights, means, cov = torch.split(out, self.sizes, 1)
        means = means.view(bsz, self.n_components, self.dim)
        if self.sigmoid_means:
            means = torch.sigmoid(means)
        cov = cov.view(bsz, self.n_components, -1)
        cov_diag = cov[:, :, :self.dim].exp() + 1e-5
        cov_factor = cov[:, :, self.dim:]
        cov_factor = cov_factor.view(bsz, -1, self.dim, self.rank)
        return weights, means, cov_diag, cov_factor
    
    def get_params(self, context):
        bsz = context.size(0)
        weights, means, cov_diag, cov_factor = self._forward(context)
        weights = weights.softmax(1)
        cov_diag_mat = torch.zeros(bsz, self.n_components, self.dim, self.dim)
        cov_diag_mat[:, :, torch.arange(self.dim), torch.arange(self.dim)] = cov_diag
        cov = torch.matmul(cov_factor, cov_factor.transpose(-1, -2)) + cov_diag_mat
        return weights, means, cov
    
    def log_prob(self, data, context):
        weights, means, cov_diag, cov_factor = self._forward(context)
        c = D.Categorical(logits=weights)
        n = D.LowRankMultivariateNormal(
            loc=means, cov_factor=cov_factor, cov_diag=cov_diag)
        m = D.MixtureSameFamily(c, n)
        log_prob = m.log_prob(data)
        return log_prob
    
    def forward(self, data, context):
        log_prob = self.log_prob(data, context)
        loss = -log_prob.mean()
        return loss
    
    def sample(self, context, n_samples):
        weights, means, cov_diag, cov_factor = self._forward(context)
        c = D.Categorical(logits=weights)
        n = D.LowRankMultivariateNormal(
            loc=means, cov_factor=cov_factor, cov_diag=cov_diag)
        m = D.MixtureSameFamily(c, n)
        samples = m.sample([n_samples,])
        return samples