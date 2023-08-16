"""
Main file for split mnist experiment offline SVGP model i.e. the model has access to the whole data set.
"""
import logging
import os
import pickle

import gpflow.models
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate, call
import hydra

import sys

sys.path.append("..")

from exp_utils import get_hydra_output_dir
from mnist_utils import setup_wandb

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="offline_mnist_experiment")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    train_data, test_data = call(cfg.dataset.dataloader)
    # Set up inducing variables
    inducing_variable = train_data[0][:cfg.n_inducing_variable].copy().tolist()
    cfg.model.inducing_variable = inducing_variable
    cfg.model.num_data = train_data[0].shape[0]

    if cfg.wandb.username is not None:
        setup_wandb(cfg)

    offline_model = instantiate(cfg.model)
    log.info("Model initialized; Optimization started!!!")

    log.info(f"---------------------------------------------")
    log.info(f"Starting mnist experiment with seed={cfg.seed}")
    log.info(f"---------------------------------------------")

    _, nlpd, acc = call(cfg.optimize)(model=offline_model, train_data=train_data,
                                      test_data=test_data, optimizer=instantiate(cfg.optimizer))

    logging.info(f"NLPD after the task is {nlpd[-1]}")
    logging.info(f"Accuracy after the task is {acc[-1]}\n\n")

    log.info(f"---------------------------------------------")

    np.savez(os.path.join(output_dir, "training_statistics.npz"), nlpd=nlpd,
             acc=acc)
    parameters = gpflow.utilities.parameter_dict(offline_model)
    with open(os.path.join(output_dir, "model_offline.pkl"), "wb") as f:
        pickle.dump(parameters, f)


if __name__ == '__main__':
    run_experiment()
