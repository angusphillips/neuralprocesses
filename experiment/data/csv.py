import os
import torch
from neuralprocesses.data.wrappers import CSVGenerator
from .util import register_data

__all__ = []


def setup(cfg, train_ds, eval_ds, test_ds, device):

    gen_train = CSVGenerator(
        dataloader=train_ds,
        dtype=torch.float32,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        device=device
    )

    gen_eval = CSVGenerator(
        dataloader=eval_ds,
        dtype=torch.float32,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        device=device
    )

    gen_test = CSVGenerator(
        dataloader=test_ds,
        dtype=torch.float32,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        device=device
    )

    return gen_train, gen_eval, gen_test, None

register_data("csv", setup)
