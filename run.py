import argparse
import logging
import jax
from tqdm import trange, tqdm
import os
import sys
import time
import warnings
from functools import partial
from data.tensordataset import DataLoader, TensorDataset, random_split

import experiment as exp
import lab as B
import neuralprocesses.torch as nps
import numpy as np
import torch
from matrix.util import ToDenseWarning

from jax.config import config

from neuralprocesses.utils import ModelWrapper
config.update("jax_enable_x64", True)

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from logging.loggers_pl import LoggerCollection
from evaluate import MetricsCollection
from evaluate.plotting import basic_plots

__all__ = ["main"]

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ToDenseWarning)

@hydra.main(config_path="config", config_name="main")
def run(cfg):
    def train_one_epoch(state, model, opt, objective, gen, *, fix_noise):
        """Train for an epoch."""
        vals = []
        for batch in gen.epoch():
            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
                fix_noise=fix_noise,
            )
            vals.append(B.to_numpy(obj))
            # Be sure to negate the output of `objective`.
            val = -B.mean(obj)
            opt.zero_grad(set_to_none=True)
            val.backward()
            opt.step()

        vals = B.concat(*vals)
        # out.kv("Loglik (T)", exp.with_err(vals, and_lower=True))
        return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))

    def eval(state, model, objective, gen, obj_name=None, stage=None, epoch=None, logger=None):
        """Perform evaluation."""
        with torch.no_grad():
            vals, kls, kls_diag = [], [], []
            for batch in gen.epoch():
                state, obj = objective(
                    state,
                    model,
                    batch["contexts"],
                    batch["xt"],
                    batch["yt"],
                )

                # Save numbers.
                n = nps.num_data(batch["xt"], batch["yt"])
                vals.append(B.to_numpy(obj))
                if "pred_logpdf" in batch:
                    kls.append(B.to_numpy(batch["pred_logpdf"] / n - obj))
                if "pred_logpdf_diag" in batch:
                    kls_diag.append(B.to_numpy(batch["pred_logpdf_diag"] / n - obj))

            # Report numbers.
            vals = B.concat(*vals)
            # print("Loglik (V)", exp.with_err(vals, and_lower=True))
            # if kls:
            #     out.kv("KL (full)", exp.with_err(B.concat(*kls), and_upper=True))
            # if kls_diag:
            #     out.kv("KL (diag)", exp.with_err(B.concat(*kls_diag), and_upper=True))

            if logger is not None:
                logger.log_metrics({f"{stage}/{obj_name}": B.mean(vals)}, epoch)

            return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))

    os.environ["GEOMSTATS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["WANDB_START_METHOD"] = "thread"

    #Converting 'None' to None - presumably there is a better way of doing this?
    cfg = OmegaConf.to_container(cfg)
    for k in cfg:
        if cfg[k] == 'None':
            cfg[k] = None
    cfg = OmegaConf.create(cfg)

    log.info(f'run path: {os.getcwd()}')
    wd = os.getcwd()

    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    metrics = [instantiate(metric) for metric in cfg.metric.values()]
    metrics = MetricsCollection(metrics)

    # Determine which device to use. Try to use a GPU if one is available.
    if cfg.device:
        device = cfg.device
    elif cfg.gpu is not None:
        device = f"cuda:{cfg.gpu}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    B.set_global_device(device)
    # Maintain an explicit random state through the execution.
    state = B.create_random_state(torch.float32, seed=cfg.seed)

    # General config.
    config = {
        "default": {
            "epochs": cfg.epochs,
            "rate": cfg.rate,
        },
        "epsilon": 1e-8,
        "epsilon_start": 1e-2,
        "cholesky_retry_factor": 1e6,
        "fix_noise": None,
        "fix_noise_epochs": 3,
        "width": cfg.width,
        "dim_embedding": cfg.dim_embedding,
        "enc_same": False,
        "num_heads": 8,
        'num_enc_layers': cfg.num_enc_layers,
        'num_dec_layers': cfg.num_dec_layers,
        "unet_channels": (64,) * 6,
        "unet_strides": (1,) + (2,) * 5,
        "conv_channels": 64,
        "encoder_scales": None,
        "fullconvgnp_kernel_factor": 2,
        "mean_diff": cfg.mean_diff,
        "num_basis_functions": 64,
        "eeg_mode": cfg.eeg_mode,
        'dim_x': cfg.dim_x,
        'dim_y': cfg.dim_y,
        "transform": None
    }


    log.info("Stage : Instantiate dataset")
    rng = jax.random.PRNGKey(cfg.seed)
    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, log=log)

    if isinstance(dataset, TensorDataset):
        # split and wrap dataset into dataloaders
        train_ds, eval_ds, test_ds = random_split(
            dataset, lengths=cfg.splits, seed=0
        )
    else:
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    train_ds, eval_ds, test_ds = (
        DataLoader(train_ds, batch_dims=cfg.batch_size, rng=next_rng, shuffle=True),
        DataLoader(eval_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
        DataLoader(test_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
    )
    log.info(
        f"Train size: {len(train_ds.dataset)}. Val size: {len(eval_ds.dataset)}. Test size: {len(test_ds.dataset)}"
    )

    # Setup data generators for training and for evaluation.
    gen_train, gen_eval, gen_test, gen_lik = exp.data[cfg.data]["setup"](
        cfg,
        train_ds,
        eval_ds,
        test_ds,
        device=device,
    )

    # Set the regularisation based on the experiment settings.
    B.epsilon = config["epsilon"]
    B.cholesky_retry_factor = config["cholesky_retry_factor"]

    if "model" in config:
        # See if the experiment constructed the particular flavour of the model already.
        model = config["model"]
    else:
        # Construct the model.
        # if cfg.model == "cnp":
        #     model = nps.construct_gnp(
        #         dim_x=config["dim_x"],
        #         dim_yc=(1,) * config["dim_y"],
        #         dim_yt=config["dim_y"],
        #         dim_embedding=config["dim_embedding"],
        #         enc_same=config["enc_same"],
        #         num_dec_layers=config["num_layers"],
        #         width=config["width"],
        #         likelihood="het",
        #         transform=config["transform"],
        #     )
        if cfg.model == "gnp":
            model = nps.construct_gnp(
                dim_x=cfg.dim_x,
                dim_y=cfg.dim_y,
                # dim_yc=(1,) * cfg.dim_y,
                # dim_yt=cfg.dim_y,
                dim_embedding=cfg.dim_embedding,
                enc_same=cfg.enc_same,
                num_enc_layers=cfg.num_enc_layers,
                num_dec_layers=cfg.num_dec_layers,
                width=cfg.width,
                likelihood="lowrank",
                num_basis_functions=cfg.num_basis_functions,
                transform=cfg.transform,
            )
        elif cfg.model == "np":
            model = nps.construct_gnp(
                dim_x=cfg.dim_x,
                dim_y=cfg.dim_y,
                # dim_yc=(1,) * cfg.dim_y,
                # dim_yt=cfg.dim_y,
                dim_embedding=cfg.dim_embedding,
                enc_same=cfg.enc_same,
                num_enc_layers=cfg.num_enc_layers,
                num_stoch_enc_layers=cfg.num_stoch_enc_layers,
                num_dec_layers=cfg.num_dec_layers,
                width=cfg.width,
                likelihood="het",
                dim_lv=cfg.dim_embedding,
                transform=cfg.transform,
            )
        elif cfg.model == "agnp":
            model = nps.construct_agnp(
                dim_x=cfg.dim_x,
                dim_y=cfg.dim_y,
                # dim_yc=(1,) * cfg.dim_y,
                # dim_yt=cfg.dim_y,
                dim_embedding=cfg.dim_embedding,
                enc_same=cfg.enc_same,
                num_heads=cfg.num_heads,
                num_dec_layers=cfg.num_dec_layers,
                width=cfg.width,
                likelihood="lowrank",
                num_basis_functions=cfg.num_basis_functions,
                transform=cfg.transform,
            )
        elif cfg.model == "anp":
            model = nps.construct_agnp(
                dim_x=cfg.dim_x,
                dim_yc=(1,) * cfg.dim_y,
                dim_yt=cfg.dim_y,
                dim_embedding=cfg.dim_embedding,
                enc_same=cfg.enc_same,
                num_enc_layers=cfg.num_enc_layers,
                num_stoch_enc_layers=cfg.num_stoch_enc_layers,
                num_heads=cfg.num_heads,
                num_dec_layers=cfg.num_dec_layers,
                width=cfg.width,
                likelihood="het",
                num_basis_functions=cfg.num_basis_functions,
                dim_lv=cfg.dim_embedding,
                transform=cfg.transform,
            )
        # elif cfg.model == "acnp":
        #     model = nps.construct_agnp(
        #         dim_x=config["dim_x"],
        #         dim_yc=(1,) * config["dim_y"],
        #         dim_yt=config["dim_y"],
        #         dim_embedding=config["dim_embedding"],
        #         enc_same=config["enc_same"],
        #         num_heads=config["num_heads"],
        #         num_dec_layers=config["num_layers"],
        #         width=config["width"],
        #         likelihood="het",
        #         transform=config["transform"],
        #     )
        # elif cfg.model == "convcnp":
        #     model = nps.construct_convgnp(
        #         points_per_unit=config["points_per_unit"],
        #         dim_x=config["dim_x"],
        #         dim_yc=(1,) * config["dim_y"],
        #         dim_yt=config["dim_y"],
        #         likelihood="het",
        #         conv_arch=cfg.arch,
        #         unet_channels=config["unet_channels"],
        #         unet_strides=config["unet_strides"],
        #         conv_channels=config["conv_channels"],
        #         conv_layers=config["num_layers"],
        #         conv_receptive_field=config["conv_receptive_field"],
        #         margin=config["margin"],
        #         encoder_scales=config["encoder_scales"],
        #         transform=config["transform"],
        #     )

        elif cfg.model == "convgnp":
            model = nps.construct_convgnp(
                points_per_unit=4,
                dim_x=cfg.dim_x,
                dim_y=cfg.dim_y,
                likelihood="lowrank",
                conv_arch="unet",
                unet_channels=(64,) * 7,
                unet_strides=(1,) + (2,) * 6,
                conv_channels=64,
                conv_layers=6,
                conv_receptive_field=100,
                num_basis_functions=64,
                margin=1,
                encoder_scales=None,
                transform="softplus",
            )
        elif cfg.model == "convnp":
            model = nps.construct_convgnp(
                points_per_unit=4,
                dim_x=cfg.dim_x,
                dim_y=cfg.dim_y,
                likelihood="het",
                conv_arch="unet",
                unet_channels=(64,) * 7,
                unet_strides=(1,) + (2,) * 6,
                conv_channels=64,
                conv_layers=6,
                conv_receptive_field=100,
                dim_lv=16,
                margin=1,
                encoder_scales=None,
                transform="softplus",
            )
        elif cfg.model == "fullconvgnp":
            model = nps.construct_fullconvgnp(
                points_per_unit=4,
                dim_x=cfg.dim_x,
                dim_y=cfg.dim_y,
                conv_arch="unet",
                unet_channels=(64,) * 7,
                unet_strides=(1,) + (2,) * 6,
                conv_channels=64,
                conv_layers=6,
                conv_receptive_field=100,
                kernel_factor=2,
                margin=1,
                encoder_scales=None,
                transform="softplus",
            )
        else:
            raise ValueError(f'Invalid model "{cfg.model}".')

    # Settings specific for the model:
    if config["fix_noise"] is None:
        if cfg.model in {"np", "anp", "convnp"}:
            config["fix_noise"] = True
        else:
            config["fix_noise"] = False

    # Ensure that the model is on the GPU and print the setup.
    model = model.to(device)
    if not cfg.load:
        log.info(f"Number of parameters: {nps.num_params(model)}")
        logger.log_metrics({'Number of parameters': nps.num_params(model)}, step=0)

    # Setup training objective.
    if cfg.objective == "loglik":
        objective = partial(
            nps.loglik,
            num_samples=cfg.num_samples,
            normalise=not cfg.unnormalised,
        )
        objective_cv = partial(
            nps.loglik,
            num_samples=cfg.num_samples,
            normalise=not cfg.unnormalised,
        )
    elif cfg.objective == "elbo":
        objective = partial(
            nps.elbo,
            num_samples=cfg.num_samples,
            subsume_context=True,
            normalise=not cfg.unnormalised,
        )
        objective_cv = partial(
            nps.elbo,
            num_samples=cfg.num_samples,
            subsume_context=False,  # Lower bound the right quantity.
            normalise=not cfg.unnormalised,
        )
    else:
        raise RuntimeError(f'Invalid objective "{cfg.objective}".')

    if cfg.mode == 'test':
        name = "model-best.torch"
        model.load_state_dict(
            torch.load(wd.file(name), map_location=device)["weights"]
        )
    elif cfg.resume:
        start = cfg.resume_at_epoch - 1
        d_last = torch.load(wd.file("model-last.torch"), map_location=device)
        d_best = torch.load(wd.file("model-best.torch"), map_location=device)
        model.load_state_dict(d_last["weights"])
        best_eval_lik = d_best["objective"]
    else:
        best_eval_lik = -np.inf

    if cfg.mode == 'train' or cfg.mode == 'all':
        # Setup training loop.
        opt = torch.optim.Adam(model.parameters(), cfg.rate)

        # Set regularisation high for the first epochs.
        original_epsilon = B.epsilon
        B.epsilon = config["epsilon_start"]

        t = tqdm(
            range(start, cfg.epochs),
            total=cfg.epochs - start,
            bar_format="{desc}{bar}{r_bar}",
            mininterval=1,
        )

        for i in t:
            # Set regularisation to normal after the first epoch.
            if i > 0:
                B.epsilon = original_epsilon

            # Perform an epoch.
            if config["fix_noise"] and i < config["fix_noise_epochs"]:
                fix_noise = 1e-4
            else:
                fix_noise = None
            state, _ = train_one_epoch(
                state,
                model,
                opt,
                objective,
                gen_train,
                fix_noise=fix_noise,
            )

            # The epoch is done. Now evaluate.
            state, val = eval(state, model, objective_cv, gen_eval)
            logger.log_metrics({'objective': val}, step=i)
            t.set_description(f"Objective: {val:.3f}")

            # Save current model.
            torch.save(
                {
                    "weights": model.state_dict(),
                    "objective": val,
                    "epoch": i + 1,
                },
                wd.file(f"model-last.torch"),
            )

            # Check if the model is the new best. If so, save it.
            if val > best_eval_lik:
                best_eval_lik = val
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "objective": val,
                        "epoch": i + 1,
                    },
                    wd.file(f"model-best.torch"),
                )
        success = True

    name = "model-best.torch"
    model.load_state_dict(
        torch.load(wd.file(name), map_location=device)["weights"]
    )

    model_wrapper = ModelWrapper(model)

    if cfg.mode == "test" or (cfg.mode == "all" and success):
        log.info("Stage : Test")
        if cfg.test_val:
            metrics.get_and_log_metrics(model_wrapper, eval_ds, cfg, log, logger, 'val', cfg.epochs)
        if cfg.test_test:
            metrics.get_and_log_metrics(model_wrapper, test_ds, cfg, log, logger, 'test', cfg.epochs)
        if cfg.test_plot:
            basic_plots(model_wrapper, test_ds, cfg, logger, 'test', cfg.epochs)
        success = True
    logger.save()
    logger.finalize("success" if success else "failure")