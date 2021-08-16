import jax
import optax 



def get_train_step(loss, optimizer):
    @jax.jit
    def step(params, opt_state, batch):
        nll, grads = jax.value_and_grad(loss)(
            params.fast if hasattr(params, "fast") else params, *batch
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)

        return nll, optax.apply_updates(params, updates), opt_state

    return step




def get_valid_step(metrics: dict=None):
    """
    metrics: dict of metrics whose keys are names of values which are scalar functions
    """
    assert type(metrics) is dict, "metrics must be a dict of scalar functions"
    
    if "valid_loss" not in metrics:
        raise ValueError("metrics must contain valid_loss function")
    
    @jax.jit
    def step(params, batch):
        output = {}
        for k, v in metrics.items():
            output[k] = v(params.fast if hasattr(params, "fast") else params, *batch)
        return output

    return step
