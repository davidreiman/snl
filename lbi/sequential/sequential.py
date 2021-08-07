import jax 
import jax.numpy as np


"""
1. Sample theta from prior
2. Simulate X from theta
3. Assemble dataset
4. Train model (flow or classifier should'nt matter)
5. Repeat at 1 but replace prior with posterior
"""


def _sample_theta(theta_prior, n_samples):
    """
    Sample theta from theta_prior
    """
    return theta_prior.sample(n_samples)

def _simulate_X(theta, n_samples):
    """
    Simulate X from theta
    """
    return theta.sample(n_samples)

def _assemble_dataset(X, y):
    """
    Assemble dataset
    """
    return jax.tree_util.tree_multimap(lambda x,y: (x,y), X, y)

def _train_model(model, dataset):
    """
    Train model
    """
    return model.fit(dataset)

def _posterior_predictive_predict(model, X):
    """
    Predict from posterior predictive
    """
    return model.predict(X)

