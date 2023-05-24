import torch

from neuralprocesses.data.wrappers import GPPosteriorGenerator
from .util import register_data

from hydra.utils import instantiate

__all__ = []


def setup(cfg, train_ds, eval_ds, test_ds, device):

    gen_train = GPPosteriorGenerator(
        dataloader=train_ds,
        dtypee=torch.float32,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_batches=cfg.num_batches_train,
        loglik_eval=False,
        device=device
    )

    gen_eval = GPPosteriorGenerator(
        dataloader=eval_ds,
        dtypee=torch.float32,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_batches=cfg.num_batches_cv,
        loglik_eval=False,
        device=device
    )

    gen_test = GPPosteriorGenerator(
        dataloader=test_ds,
        dtypee=torch.float32,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_batches=cfg.num_batches_test,
        loglik_eval=False,
        device=device
    )

    return gen_train, gen_eval, gen_test, None

register_data("gpposterior", setup)
