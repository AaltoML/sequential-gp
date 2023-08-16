import logging
import os
import pickle

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

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="streaming_mnist_experiment")
def run_experiment(cfg: DictConfig):
    """
    Initialize and run the experiment.
    """
    output_dir = get_hydra_output_dir()

    all_train_data, all_test_data = call(cfg.dataset.dataloader)

    log.info(f"---------------------------------------------")
    log.info(f"Starting mnist experiment with seed={cfg.seed}")
    log.info(f"---------------------------------------------")

    # Set up inducing variables
    inducing_variable = all_train_data[0][0][:cfg.n_inducing_variable].copy().tolist()
    cfg.model.inducing_variable = inducing_variable

    cfg.model.num_data = all_train_data[0][0].shape[0]
    cfg.model.num_latent_gps = cfg.num_classes
    model = instantiate(cfg.model)

    log.info("Model initialized; Optimization started!!!")

    optimizer = instantiate(cfg.optimizer)

    nlpd_vals = []
    acc_vals = []
    task_break_pnts = []
    task_id = 0
    first_init = True

    previous_tasks = None
    mu, Su, Kaa, Zopt = None, None, None, None

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
        f_mean, f_var = model.predict_f(previous_tasks[0])
        if len(f_var.shape) == 1:
            f_var = f_var[..., None]
        nlpd = model.likelihood.predict_log_density(f_mean, f_var, previous_tasks[1])
        nlpd_init = -1 * tf.reduce_mean(nlpd).numpy().item()

        pred_m, _ = model.predict_y(previous_tasks[0])
        pred_argmax = tf.reshape(tf.argmax(pred_m, axis=1), (-1, 1))
        acc_init = np.mean(pred_argmax == previous_tasks[1])

        nlpd, acc, mu, Su, Kaa, Zopt, model = call(cfg.optimize)(optimizer=optimizer, model=model,
                                                                 train_data=train_data,
                                                                 test_data=previous_tasks, mu=mu, Su=Su,
                                                                 Kaa=Kaa, Zopt=Zopt,
                                                                 first_init=first_init)
        first_init = False
        logging.info(f"NLPD after the task is {nlpd[-1]}")
        logging.info(f"Accuracy after the task is {acc[-1]}\n\n")

        # Add init acc and nlpd
        acc = [acc_init] + acc
        nlpd = [nlpd_init] + nlpd

        nlpd_vals += nlpd
        acc_vals += acc

        task_break_pnts.append(len(nlpd_vals))

        logging.info("NLPD on previous tasks:\n")
        for i in range(task_id, -1, -1):
            f_mean, f_var = model.predict_f(all_test_data[i][0])
            if len(f_var.shape) == 1:
                f_var = f_var[..., None]
            nlpd = model.likelihood.predict_log_density(f_mean, f_var, all_test_data[i][1])
            nlpd = -1 * tf.reduce_mean(nlpd)

            pred_m, _ = model.predict_y(all_test_data[i][0])
            pred_argmax = tf.reshape(tf.argmax(pred_m, axis=1), (-1, 1))
            acc = np.mean(pred_argmax == all_test_data[i][1])

            logging.info(f"NLPD on task {i} is {nlpd}")
            logging.info(f"Accuracy on task {i} is {acc}\n\n")

        log.info(f"---------------------------------------------")
        task_id += 1

    np.savez(os.path.join(output_dir, "training_statistics.npz"), nlpd=nlpd_vals,
             acc=acc_vals, task_break_pnts=task_break_pnts)
    parameters = gpflow.utilities.parameter_dict(model)
    with open(os.path.join(output_dir, "model_streaming.pkl"), "wb") as f:
        pickle.dump(parameters, f)


if __name__ == '__main__':
    run_experiment()
