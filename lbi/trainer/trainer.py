import jax.numpy as np
from tqdm.auto import tqdm


def Train(
    params,
    loss,
    optimizer,
    opt_state,
    train_dataloader,
    train_step,
    nsteps=10000,
    eval_interval=100,
    valid_dataloader=None,
    valid_step=None,
    logger=None,
    train_kwargs=None,
    valid_kwargs=None,
):
    if train_kwargs is None:
        train_kwargs = {}
    if valid_kwargs is None:
        valid_kwargs = {}

    iterator = tqdm(range(nsteps))
    best_valid_loss = np.inf
    best_params = params  # purposefully not a copy

    try:
        for _step_num in iterator:
            batch = [np.array(a) for a in next(iter(train_dataloader))]
            nll, params, opt_state = train_step(
                loss,
                optimizer,
                params,
                opt_state,
                batch,
            )
            if np.isnan(nll):
                print("We've hit nan-ville. Stopping early.")
                break
            
            if logger is not None:
                logger.scalar("train loss", nll, step=_step_num)

            if _step_num % eval_interval == 0 and valid_step is not None:
                assert valid_dataloader is not None, "valid_dataloader is None"
                batch = [np.array(a) for a in next(iter(valid_dataloader))]

                # assumes first valid metric is the validation loss
                valid_metrics = valid_step(loss, params, batch)
                if valid_metrics[0] < best_valid_loss:
                    best_valid_loss = valid_metrics[0]
                    best_params = params
                elif np.isnan(valid_metrics[0]) or np.isinf(valid_metrics[0]):
                    print("We've hit nan-ville. Stopping early.")
                    break
                if logger is not None:
                    for i, valid_metric in enumerate(valid_metrics):
                        logger.scalar(f"valid metric {i}", valid_metric, step=_step_num)
                iterator.set_description(f"Valid loss: {valid_metrics[0]:.4f}")
    except KeyboardInterrupt:
        print("Keyboard interrupted. Stopping early")
        pass
    if valid_step is None:
        best_params = params

    return best_params, best_valid_loss
