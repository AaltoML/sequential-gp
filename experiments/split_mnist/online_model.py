"""
Main file for split mnist experiment online model.
"""

import pickle
import os
import logging
import wandb

import gpflow
import tensorflow as tf
import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate, call

import sys

sys.path.append("..")
sys.path.append("../..")

from exp_utils import get_hydra_output_dir
from src.models.tsvgp_cont import piv_chol
from mnist_utils import setup_wandb

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="online_mnist_experiment")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    if cfg.wandb.username is not None:
        setup_wandb(cfg)

    all_train_data, all_test_data = call(cfg.dataset.dataloader)

    # Set up inducing variables
    inducing_variable = all_train_data[0][0][:cfg.n_inducing_variable].copy().tolist()
    cfg.model.inducing_variable = inducing_variable

    model = instantiate(cfg.model)
    gpflow.utilities.set_trainable(model.inducing_variable.Z, False)

    memory = (all_train_data[0][0][:1], all_train_data[0][1][:1])
    online_gp = instantiate(cfg.online_gp)(model=model, memory=memory, opt_hypers=instantiate(cfg.optimizer),
                                           Z_picker=piv_chol, memory_picker=call(cfg.memory_picker))

    log.info(f"---------------------------------------------")
    log.info(f"Starting split mnist experiment with seed={cfg.seed}")
    log.info(f"---------------------------------------------")

    nlpd_vals = []
    acc_vals = []
    task_break_pnts = []
    task_id = 0

    previous_tasks = None
    for train_data, test_data in zip(all_train_data, all_test_data):

        log.info(f"---------------------------------------------")
        log.info(f"Task {task_id}")
        log.info(f"Train data are {train_data[0].shape[0]} and test data are {test_data[0].shape[0]}")
        log.info("Splitting data into sets...")

        if previous_tasks is None:
            previous_tasks = test_data
        else:
            previous_tasks = (np.concatenate([previous_tasks[0], test_data[0].copy()], axis=0),
                              np.concatenate([previous_tasks[1], test_data[1].copy()], axis=0))

        # Calculating Accuracy and NLPD before the model is trained
        nlpd_init = -1 * tf.reduce_mean(online_gp.model.predict_log_density(previous_tasks)).numpy().item()

        pred_m, _ = online_gp.model.predict_y(previous_tasks[0])
        pred_argmax = tf.reshape(tf.argmax(pred_m, axis=1), (-1, 1))
        acc_init = np.mean(pred_argmax == previous_tasks[1])

        nlpd, acc = call(cfg.optimize)(model=online_gp, train_data=train_data, test_data=previous_tasks)

        if cfg.wandb.username is not None:
            wandb.log({"Accuracy": acc[-1]})

        # Add init acc and nlpd
        acc = [acc_init] + acc
        nlpd = [nlpd_init] + nlpd

        nlpd_vals += nlpd
        acc_vals += acc

        task_break_pnts.append(len(nlpd_vals))

        logging.info(f"NLPD after the task is {nlpd[-1]}")
        logging.info(f"Accuracy after the task is {acc[-1]}\n\n")

        logging.info("NLPD on all tasks:\n")
        for i in range(task_id, -1, -1):
            nlpd = -1 * tf.reduce_mean(online_gp.model.predict_log_density(all_test_data[i]))

            pred_m, _ = online_gp.model.predict_y(all_test_data[i][0])
            pred_argmax = tf.reshape(tf.argmax(pred_m, axis=1), (-1, 1))
            acc = np.mean(pred_argmax == all_test_data[i][1])

            logging.info(f"NLPD on task {i} is {nlpd}")
            logging.info(f"Accuracy on task {i} is {acc}\n\n")

            # Save model memory and inducing variables
            Z = online_gp.model.inducing_variable.Z.numpy().copy()
            mem = online_gp.memory[0].copy()
            np.savez(os.path.join(output_dir, f"memory_and_Z_{task_id}.npz"), mem=mem, Z=Z)

        log.info(f"---------------------------------------------")
        task_id += 1

    np.savez(os.path.join(output_dir, "training_statistics.npz"), nlpd=nlpd_vals,
             acc=acc_vals, task_break_pnts=task_break_pnts)
    parameters = gpflow.utilities.parameter_dict(online_gp.model)
    with open(os.path.join(output_dir, "model_online.pkl"), "wb") as f:
        pickle.dump(parameters, f)


if __name__ == '__main__':
    run_experiment()
