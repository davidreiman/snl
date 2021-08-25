import jax
import jax.numpy as np
from jax.experimental import stax


def ResidualBlock(hidden_dim, act=None):
    """
    Outputs two features: the embedding and the unchanged inputs.
    """
    if act is None:
        act = stax.Selu
    fully_connected = stax.serial(stax.Dense(hidden_dim), act)
    shortcut = stax.Identity
    return stax.serial(
        stax.FanOut(2), stax.parallel(fully_connected, shortcut), stax.FanInConcat()
    )


def Classifier(num_layers=5, hidden_dim=128, dropout=0.0, use_residual=True, act=None):
    if act is None:
        act = stax.Selu


    if use_residual:
        layers = [ResidualBlock(hidden_dim=hidden_dim) for _ in range(num_layers)]
    else:
        layers = [lyr for _ in range(num_layers) for lyr in (stax.Dense(hidden_dim), act)]
    # append a final linear layer for binary classification
    layers += [stax.Dense(1)]

    init_random_params, _logit_d = stax.serial(*layers)
    
    def logit_d(params, *args):
        return _logit_d(params, np.hstack(args))
    
    return init_random_params, logit_d


if __name__ == "__main__":
    import optax
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from functools import partial
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from tqdm.auto import tqdm

    def loss(params, inputs, targets, eps=1e-4):
        """binary cross entropy with logits
        taken from jaxchem
        """
        # log ratio is the logit of the discriminator
        l_d = logit_d(params, inputs).squeeze()
        max_val = np.clip(-l_d, 0, None)
        L = (
            l_d
            - l_d * targets
            + max_val
            + np.log(np.exp(-max_val) + np.exp((-l_d - max_val)))
        )

        return np.sum(L)

    @jax.jit
    def train_step(params, opt_state, batch):
        def step(params, opt_state, batch):
            inputs, labels = batch
            nll, grads = jax.value_and_grad(loss)(params, inputs, labels)
            updates, opt_state = opt_update(grads, opt_state, params)

            return nll, optax.apply_updates(params, updates), opt_state

        return step(params, opt_state, batch)

    nsteps = 100
    batch_size = 64
    seed = 1234

    X, y = load_breast_cancer(return_X_y=True)
    num_feat = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_s = torch.tensor(X_train_s, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_dataloader = DataLoader(
        TensorDataset(X_train_s, y_train), batch_size=batch_size, shuffle=True
    )

    init_random_params, logit_d = Classifier(num_layers=3, hidden_dim=128)
    _, params = init_random_params(jax.random.PRNGKey(seed), (-1, X_train_s.shape[-1]))

    learning_rate = 0.01
    opt_init, opt_update = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-learning_rate),
    )

    opt_state = opt_init(params)

    print(
        "Test accuracy: {:.3f}".format(
            jax.numpy.mean(
                (jax.nn.sigmoid(logit_d(params, np.array(X_test_s))) > 0.5).flatten()
                == y_test
            )
        )
    )

    iterator = tqdm(range(nsteps))
    for _ in iterator:
        for batch in train_dataloader:
            batch = [np.array(a) for a in batch]
            nll, params, opt_state = train_step(params, opt_state, batch)
        iterator.set_description("nll = {:.3f}".format(nll))

    print()
    print(
        "Test accuracy: {:.3f}".format(
            jax.numpy.mean(
                (jax.nn.sigmoid(logit_d(params, np.array(X_test_s))) > 0.5).flatten()
                == y_test
            )
        )
    )
    print(
        classification_report(
            y_test,
            (jax.nn.sigmoid(logit_d(params, np.array(X_test_s))) > 0.5).flatten(),
        )
    )
