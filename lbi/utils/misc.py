import os
import time
import numpy as np
import pathlib
import shutil


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False


def get_n_params(model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in trainable])
    return n_params


def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (0.5)


def prep_log_path(log_path='./runs/test_run/', settings_path='./settings.json'):
    """
    return: str
        Absolute path to log_path
    """
    # make nested directory if needed
    path = pathlib.Path(log_path)
    path.mkdir(parents=True, exist_ok=True)
    # copy settings folder into new directory
    # TODO: Clean this up?
    shutil.copy(settings_path, str(path.absolute())+'/settings.json')
    # no need for the entire Path (at least on linux). Might need it for windows?
    return str(path.absolute())

