from re import A
import jax.numpy as np
import flows
from jax.experimental import stax


def get_masks(input_dim, context_dim=0, hidden_dim=64, num_hidden=1):
    masks = []
    degrees = [np.arange(input_dim)]
    input_degrees = np.arange(input_dim)

    for n_h in range(num_hidden + 1):
        degrees += [np.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for i, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        mask = np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(
            np.float32
        )
        if i == 0:  # pass in context
            # TODO: This still doesn't pass context to the most-masked element.
            #       Need to figure out how to do that effectively.
            mask = np.vstack((mask, np.ones((context_dim, mask.shape[-1]))))
        masks += [mask]

    return masks


def masked_transform(rng, input_dim, context_dim=0, hidden_dim=64, num_hidden=1):
    masks = get_masks(
        input_dim, context_dim=context_dim, hidden_dim=hidden_dim, num_hidden=num_hidden
    )
    act = stax.Relu
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
            *(
                flows.MADE(masked_transform),
                flows.Reverse(),
                flows.ActNorm(),
            )
            * n_layers
        ),
        prior=flows.Normal(),
    )


if __name__ == "__main__":
    import jax
    import optax
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from jax.experimental import stax
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    def loss(params, inputs, context=None):
        return -log_pdf(params, inputs, context).mean()

    @jax.jit
    def train_step(params, opt_state, batch):
        nll, grads = jax.value_and_grad(loss)(params, *batch)
        updates, opt_state = opt_update(grads, opt_state, params)
        return nll, optax.apply_updates(params, updates), opt_state

    hidden_dim = 32
    n_layers = 4

    batch_size = 128
    seed = 1234
    nsteps = 400

    X, y = make_moons(n_samples=10000, noise=0.05, random_state=seed)
    y = y[:, None]
    input_dim = X.shape[1]
    context_dim = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, stratify=y, random_state=seed
    )

    X_train_s = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_dataloader = DataLoader(
        TensorDataset(X_train_s, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    rng = jax.random.PRNGKey(seed)
    params, log_pdf, sample = MaskedAffineFlow(n_layers=n_layers)(
        rng,
        input_dim=input_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
    )

    learning_rate = 1e-4
    opt_init, opt_update = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.adamw(learning_rate=learning_rate),
    )

    opt_state = opt_init(params)

    iterator = tqdm(range(nsteps))
    try:
        for _ in iterator:
            for batch in train_dataloader:
                batch = [np.array(a) for a in batch]
                nll, params, opt_state = train_step(params, opt_state, batch)
            iterator.set_description("nll = {:.3f}".format(nll))
    except KeyboardInterrupt:
        pass

    plt.scatter(*X_train.T, color="grey", alpha=0.01, marker=".")

    samples_0 = sample(rng, params, context=np.zeros((1000, context_dim)))
    plt.scatter(*samples_0.T, color="red", label="0", marker=".", alpha=0.2)
    samples_1 = sample(rng, params, context=np.ones((1000, context_dim)))
    plt.scatter(*samples_1.T, color="blue", label="1", marker=".", alpha=0.2)

    plt.xlim(-1.5, 2.5)
    plt.ylim(-1, 1.5)
    plt.legend()
    plt.show()
