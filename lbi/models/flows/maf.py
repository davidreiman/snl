import jax.numpy as np
import flows
from jax.experimental import stax


def get_masks(input_dim, context_dim=0, hidden_dim=64, num_hidden=1):
    """
    now adapted to take context
    """
    masks = []
    input_degrees = np.arange(input_dim)
    degrees = [input_degrees]

    for n_h in range(num_hidden + 1):
        degrees += [np.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        mask = np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(np.float32)
        mask = np.pad(mask, pad_width=(0, context_dim), constant_values=1)  # to handle context
        masks += [mask]
    return masks


def masked_transform(rng, input_dim, context_dim=0):
    masks = get_masks(input_dim, context_dim=context_dim, hidden_dim=64, num_hidden=1)
    act = stax.Gelu
    init_fun, apply_fun = stax.serial(
        flows.MaskedDense(masks[0]),
        act,
        flows.MaskedDense(masks[1]),
        act,
        flows.MaskedDense(masks[2].tile(2)),
    )
    _, params = init_fun(rng, (input_dim + context_dim,))
    return params, apply_fun


def MaskedAffineFlow(n_layers=5):
    """
    A sequence of affine transformations with a masked affine transform.

    returns init_fun
    """
    return flows.Flow(
        flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * 5),
        flows.Normal(),
    )
