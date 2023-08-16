"""
Main file for the proposed model on UCI regression tasks.
"""
import logging
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call
import hydra
import wandb
import gpflow

import sys

sys.path.append("../../")
sys.path.append("..")

from src.models.utils import fixed_Z, memory_picker, random_picker
from exp_utils import get_hydra_output_dir, convert_data_to_online
from uci_utils import setup_wandb

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="online_experiment")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    all_train_data, all_test_data = call(cfg.dataset.dataloader)

    if len(all_train_data) > 1:
        log.info(f"Cross validation starting with {cfg.dataset.dataloader.n_k_folds} K-folds!!!")

    k_fold_nlpds = []
    k_fold_eval = []
    k_fold_time = []
    k_fold_id = 0

    for train_data, test_data in zip(all_train_data, all_test_data):

        log.info(f"---------------------------------------------")
        log.info(f"Starting with set {k_fold_id}")
        log.info(f"---------------------------------------------")

        log.info(f"Train data are {train_data[0].shape[0]} and test data are {test_data[0].shape[0]}")
        log.info("Splitting data into sets...")

        online_data = convert_data_to_online(train_data, cfg.n_sets, cfg.sort_data)
        log.info(f"Data splitted successfully into {cfg.n_sets} sets!!!")

        np.savez(os.path.join(output_dir, "splitted_dataset" + str(k_fold_id) + ".npz"), data=online_data)

        if cfg.wandb.username is not None:
            setup_wandb(cfg)

        # Set up inducing variables
        inducing_variable = train_data[0][:cfg.n_inducing_variable].copy().tolist()
        cfg.model.inducing_variable = inducing_variable

        model = instantiate(cfg.model)
        if cfg.load_model_path is not None:
            with open(cfg.load_model_path, "rb") as f:
                dict_params = pickle.load(f)

            model.inducing_variable.Z = dict_params['.inducing_variable.Z']
            model.kernel.lengthscales = dict_params['.kernel.lengthscales']
            model.kernel.variance = dict_params['.kernel.variance']
            model.likelihood.variance = dict_params['.likelihood.variance']

            log.info(f"Model parameters loaded from {cfg.load_model_path}")

        memory = (online_data[0][0][:1], online_data[0][1][:1])
        if cfg.online_gp.memory_picker == "random":
            memory_picker = random_picker
        else:
            memory_picker = memory_picker

        online_gp = instantiate(cfg.online_gp)(model=model, memory=memory, opt_hypers=instantiate(cfg.optimizer),
                                               Z_picker=fixed_Z, memory_picker=memory_picker)
        start_time = time.time()
        nlpd_vals, eval_vals, _ = call(cfg.optimize)(online_gp=online_gp, train_data=online_data, test_data=test_data,
                                                     debug=True)
        end_time = time.time()

        log.info(f"Test NLPD: {nlpd_vals[-1]}")
        log.info(f"Test RMSE/Acc: {eval_vals[-1]}")
        log.info(f"Time (s): {end_time - start_time}")

        log.info("Optimization successfully done!!!")

        if cfg.wandb.username is not None:
            plt.clf()
            plt.plot(nlpd_vals)
            plt.title("NLPD")
            wandb.log({"optim_nlpd_vals": plt})

            plt.clf()
            plt.plot(eval_vals)
            plt.title("RMSE/Acc")
            wandb.log({"optim_eval_vals": plt})

        np.savez(os.path.join(output_dir, "training_statistics_" + str(k_fold_id) + ".npz"), nlpd=nlpd_vals,
                 eval=eval_vals)
        parameters = gpflow.utilities.parameter_dict(model)
        with open(os.path.join(output_dir, "model_" + str(k_fold_id) + ".pkl"), "wb") as f:
            pickle.dump(parameters, f)

        k_fold_id += 1
        k_fold_nlpds.append(nlpd_vals[-1])
        k_fold_eval.append(eval_vals[-1])
        k_fold_time.append(end_time - start_time)
        log.info(f"---------------------------------------------")

    if len(k_fold_nlpds) > 1:
        log.info(f"Mean NLPD over k-folds = {np.mean(k_fold_nlpds)}")
        log.info(f"Std NLPD over k-folds = {np.std(k_fold_nlpds)}")

        log.info(f"Mean RMSE/Acc over k-folds = {np.mean(k_fold_eval)}")
        log.info(f"Std RMSE/Acc over k-folds = {np.std(k_fold_eval)}")

        log.info(f"Mean time over k-folds = {np.mean(k_fold_time)}")
        log.info(f"Std time over k-folds = {np.std(k_fold_time)}")


if __name__ == '__main__':
    run_experiment()
