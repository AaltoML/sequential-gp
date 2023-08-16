"""
Main file for UCI regression tasks offline model.
"""
import logging
import os
import pickle

import gpflow.models
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate, call
import hydra
import wandb

import sys
sys.path.append("..")

from exp_utils import get_hydra_output_dir
from uci_utils import load_model_parameters, setup_wandb

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="offline_experiment")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    all_train_data, all_test_data = call(cfg.dataset.dataloader)

    log.info(f"---------------------------------------------")
    log.info(f"Dataset : {cfg.dataset}")
    log.info(f"---------------------------------------------")

    if len(all_train_data) > 1:
        log.info(f"Cross validation starting with {cfg.dataset.dataloader.n_k_folds} K-folds!!!")

    k_fold_nlpds = []
    k_fold_eval = []
    k_fold_id = 0
    for train_data, test_data in zip(all_train_data, all_test_data):
        cfg.model.num_data = train_data[0].shape[0]
        log.info(f"---------------------------------------------")
        log.info(f"Starting with set {k_fold_id}")
        log.info(f"---------------------------------------------")
        log.info(f"Train data are {train_data[0].shape[0]} and test data are {test_data[0].shape[0]}")

        if cfg.wandb.username is not None:
            setup_wandb(cfg)

        # Set up inducing variables
        inducing_variable = train_data[0][:cfg.n_inducing_variable].copy().tolist()
        cfg.model.inducing_variable = inducing_variable

        model = instantiate(cfg.model)
        log.info("Model initialized; Optimization started!!!")
        if cfg.load_model_path is not None:
            with open(cfg.load_model_path, "rb") as f:
                dict_params = pickle.load(f)
            load_model_parameters(model, dict_params)
            log.info(f"Model parameters loaded from {cfg.load_model_path}")

        elbo_vals, nlpd_vals, eval_vals = call(cfg.optimize)(model=model, train_data=train_data, test_data=test_data,
                                                             optimizer=instantiate(cfg.optimizer))
        if len(nlpd_vals) > 0:
            log.info(f"Final ELBO: {elbo_vals[-1]}")
            log.info(f"Test NLPD: {nlpd_vals[-1]}")
            log.info(f"Test RMSE/Acc: {eval_vals[-1]}")

        log.info("Optimization successfully done!!!")

        if cfg.wandb.username is not None:
            plt.clf()
            plt.plot(elbo_vals)
            plt.title("ELBO")
            wandb.log({"optim_elbo_vals": plt})

            plt.clf()
            plt.plot(nlpd_vals)
            plt.title("NLPD")
            wandb.log({"optim_nlpd_vals": plt})

            plt.clf()
            plt.plot(eval_vals)
            plt.title("Eval.")
            wandb.log({"optim_eval_vals": plt})

        np.savez(os.path.join(output_dir, "training_statistics_" + str(k_fold_id) + ".npz"), elbo=elbo_vals,
                 nlpd=nlpd_vals, eval=eval_vals)
        parameters = gpflow.utilities.parameter_dict(model)
        with open(os.path.join(output_dir, "model_" + str(k_fold_id) + ".pkl"), "wb") as f:
            pickle.dump(parameters, f)

        k_fold_id += 1
        k_fold_nlpds.append(nlpd_vals[-1])
        k_fold_eval.append(eval_vals[-1])
        log.info(f"---------------------------------------------")

    if len(k_fold_nlpds) > 1:
        log.info(f"Mean NLPD over k-folds = {np.mean(k_fold_nlpds)}")
        log.info(f"Std NLPD over k-folds = {np.std(k_fold_nlpds)}")

        log.info(f"Mean eval over k-folds = {np.mean(k_fold_eval)}")
        log.info(f"Std eval over k-folds = {np.std(k_fold_eval)}")


if __name__ == '__main__':
    run_experiment()
