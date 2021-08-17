import jax
import jax.numpy as np
import torch


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset class for all datasets.
    """

    def __init__(self, X, Theta, **kwargs):
        super(BaseDataset, self).__init__(**kwargs)
        assert (
            X.shape[0] == Theta.shape[0]
        ), "X and Theta must have the same number of rows"
        self.X = X
        self.Theta = Theta

    def __getitem__(self, index):
        x = self.X[index]
        theta = self.Theta[index]
        return x, theta

    def __len__(self):
        return self.X.shape[0]


class GaussianNoiseDataset(BaseDataset):
    """
    Add gaussian noise on demand

    logged_X_idx: indices of X that are logged. Adding noise to these
                    requires exponentiating the data first


    NOTE: The gaussian noise is added to the transformed data,
            NOT on the original data.
            Make sure to transform sigma accordingly
    """

    def __init__(self, rng, X, Theta, sigma, logged_X_idx=None, **kwargs):
        super(GaussianNoiseDataset, self).__init__(X, Theta, **kwargs)
        if logged_X_idx is None:
            logged_X_idx = np.zeros(X.shape[0])

        self.rng = rng
        self.sigma = sigma
        self.logged_X_idx = logged_X_idx

    def __getitem__(self, index):
        x = self.X[index]
        # exponentiate the logged data
        x[self.logged_X_idx] = np.exp(x[self.logged_X_idx])
        # add noise
        x = x + self.sigma * jax.random.normal(key=self.rng, shape=x.shape)
        # log back
        x[self.logged_X_idx] = np.log(x[self.logged_X_idx])

        theta = self.Theta[index]
        return x, theta

