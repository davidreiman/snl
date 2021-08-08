import torch
from .base import BaseDataset
from .collate_fns import FlowCollate_fn, LikelihoodRatioCollate_fn

def getDataLoaders(
    X, Theta, sequential_mode, batch_size, train_split=0.9, num_workers=0, **kwargs
):
    """
    Get data loaders for a dataset.
    """
    from sklearn.model_selection import train_test_split

    assert sequential_mode in [
        "flow",
        "likelihood_ratio",
    ], "sequential_mode must be 'flow' or 'likelihood_ratio'"
    
    if sequential_mode == "flow":
        collate_fn = FlowCollate_fn
    elif sequential_mode == "likelihood_ratio":
        collate_fn = LikelihoodRatioCollate_fn

    train_X, valid_X, train_Theta, valid_Theta = train_test_split(
        X, Theta, train_size=train_split
    )

    trainDataset = BaseDataset(train_X, train_Theta, **kwargs)
    validDataset = BaseDataset(valid_X, valid_Theta, **kwargs)

    trainLoader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    validLoader = torch.utils.data.DataLoader(
        validDataset,
        batch_size=batch_size * 10,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return trainLoader, validLoader
