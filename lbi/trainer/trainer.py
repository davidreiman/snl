import jax.numpy as np
from tqdm.auto import tqdm


def getTrainer(
    optimizer,
    train_step,
    nsteps=10000,
    eval_interval=100,
    valid_step=None,
    logger=None,
    train_kwargs=None,
    valid_kwargs=None,
):
    if train_kwargs is None:
        train_kwargs = {}
    if valid_kwargs is None:
        valid_kwargs = {}

    def trainer(
        params,
        opt_state,
        train_dataloader,
        valid_dataloader=None,
        num_round=0,
    ):

        iterator = tqdm(range(nsteps))
        best_valid_loss = np.inf
        best_params = params  # purposefully not a copy

        try:
            for _step_num in iterator:
                batch = [np.array(a) for a in next(iter(train_dataloader))]
                nll, params, opt_state = train_step(
                    params,
                    opt_state,
                    batch,
                )
                if np.isnan(nll):
                    print("We've hit nan-ville. Stopping early.")
                    break

                if hasattr(logger, "scalar"):
                    logger.scalar("train loss", nll, step=(num_round*nsteps)+_step_num)

                if _step_num % eval_interval == 0 and valid_step is not None:
                    assert valid_dataloader is not None, "valid_dataloader is None"
                    batch = [np.array(a) for a in next(iter(valid_dataloader))]

                    # assumes first valid metric is the validation loss
                    valid_metrics = valid_step(params, batch)
                    if valid_metrics['valid_loss'] < best_valid_loss:
                        best_valid_loss = valid_metrics['valid_loss']
                        best_params = params
                    elif np.isnan(valid_metrics['valid_loss']) or np.isinf(valid_metrics['valid_loss']):
                        print("We've hit nan-ville. Stopping early.")
                        break
                    if hasattr(logger, "scalar"):
                        for key, val in valid_metrics.items():
                            logger.scalar(
                                f"{key}", val, step=(num_round*nsteps)+_step_num
                            )
                    iterator.set_description(f"Valid loss: {valid_metrics['valid_loss']:.4f}")
        except KeyboardInterrupt:
            print("Keyboard interrupted. Stopping early")
            pass
        if valid_step is None:
            best_params = params

        return best_params

    return trainer
