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

    for i, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        mask = np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(
            np.float32
        )
        if i == 0:
            mask = np.pad(
                mask, pad_width=(0, context_dim), constant_values=1
            )  # to handle context
        elif i == 1:
            mask = np.vstack((mask, np.ones((context_dim, mask.shape[-1]))))
        masks += [mask]
    return masks


def masked_transform(rng, input_dim, context_dim=0, hidden_dim=64, num_hidden=1):
    masks = get_masks(
        input_dim, context_dim=context_dim, hidden_dim=hidden_dim, num_hidden=num_hidden
    )
    act = stax.Gelu
    init_fun, apply_fun = stax.serial(
        flows.MaskedDense(masks[0], use_context=True),
        act,
        flows.MaskedDense(masks[1]),
        act,
        flows.MaskedDense(
            masks[2].tile(2)
        ),  # 2x because it parametrizes affine transform
    )
    _, params = init_fun(rng, (input_dim + context_dim,))
    return params, apply_fun


def MaskedAffineFlow(n_layers=5):
    """
    A sequence of affine transformations with a masked affine transform.

    returns init_fun
    """
    return flows.Flow(
        transformation=flows.Serial(
            *(flows.MADE(masked_transform), flows.Reverse()) * n_layers
        ),
        prior=flows.Normal(),
    )


if __name__ == "__main__":
    import jax
    from jax.experimental import stax

    batch = 5
    input_dim = 3
    context_dim = 2
    hidden_dim = 128
    n_layers = 5

    rng = jax.random.PRNGKey(0)
    params, log_pdf, sample = MaskedAffineFlow(n_layers=n_layers)(
        rng,
        input_dim=input_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
    )
    # print(params)
    print(
        log_pdf(
            params, np.zeros((batch, input_dim)), context=np.zeros((batch, context_dim))
        )
    )
    # print(log_pdf(params, np.ones((batch, input_dim))))
