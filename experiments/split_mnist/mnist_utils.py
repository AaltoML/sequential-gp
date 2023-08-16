"""
The file contains utility functions for split-mnist expeirment.
"""
from typing import Tuple

import tensorflow as tf
import numpy as np
import gpflow
import wandb
from omegaconf import OmegaConf

import sys

sys.path.append("../../..")

from src.models.tsvgp_cont import OnlineGP
from src.streaming_sparse_gp.osvgpc import OSVGPC


def setup_wandb(cfg):
    """
    Set up wandb.
    """
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(project="MNIST", entity=cfg.wandb.username, config=wandb_cfg)


def load_mnist(seed: int = None, train_split_percentage: float = 0.80) -> (list, list):
    """
    Load MNIST data set.

    seed: if seed needs to be fixed, by default it is None.
    train_split_percentage: float value between (0, 1), governing the split of data into train and test set.
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    x, y = mnist_train
    x = tf.reshape(x, [x.shape[0], -1]).numpy()
    x = x.astype(np.float64) / 255
    y = np.reshape(y, (-1, 1))
    y = np.int64(y)

    xt, yt = mnist_test
    xt = tf.reshape(xt, [xt.shape[0], -1]).numpy()
    xt = xt.astype(np.float64) / 255
    yt = np.reshape(yt, (-1, 1))
    yt = np.int64(yt)

    # merge train and test into one
    X = np.concatenate([x, xt], axis=0)
    Y = np.concatenate([y, yt], axis=0)

    all_data = np.concatenate([X, Y], axis=1)

    n_train = int(all_data.shape[0] * train_split_percentage)

    np.random.shuffle(all_data)
    train_data = all_data[:n_train]
    test_data = all_data[n_train:]

    train_tasks = (train_data[:, :-1], train_data[:, -1:].astype(np.int64))
    test_tasks = (test_data[:, :-1], test_data[:, -1:].astype(np.int64))

    return train_tasks, test_tasks


def load_split_mnist(seed: int = None, train_split_percentage: float = 0.80) -> (list, list):
    """
    Load split-mnist data set.

    seed: if seed needs to be fixed, by default it is None.
    train_split_percentage: float value between (0, 1), governing the split of data into train and test set.
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    x, y = mnist_train
    x = tf.reshape(x, [x.shape[0], -1]).numpy()
    x = x.astype(np.float64) / 255
    y = np.reshape(y, (-1, 1))
    y = np.int64(y)

    xt, yt = mnist_test
    xt = tf.reshape(xt, [xt.shape[0], -1]).numpy()
    xt = xt.astype(np.float64) / 255
    yt = np.reshape(yt, (-1, 1))
    yt = np.int64(yt)

    # merge train and test into one
    X = np.concatenate([x, xt], axis=0)
    Y = np.concatenate([y, yt], axis=0)

    train_tasks = []
    test_tasks = []

    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    # Create specific tasks
    for t in tasks:
        idx, _ = np.where((Y == t[0]) | (Y == t[1]))
        np.random.shuffle(idx)

        x_task = X[idx]
        y_task = Y[idx]

        n_task = int(x_task.shape[0] * train_split_percentage)

        train_tasks.append((x_task[:n_task], y_task[:n_task]))
        test_tasks.append((x_task[n_task:], y_task[n_task:]))

    return train_tasks, test_tasks


def get_mini_batches(data: [np.ndarray, np.ndarray], minibatch_size: int = 1000) -> list:
    """
    Make mini-batches of data.
    """
    num_batches = int(data[0].shape[0] / minibatch_size)
    batched_data = []
    for n in range(num_batches):
        tmp_data = (data[0][n * minibatch_size:(n + 1) * minibatch_size],
                    data[1][n * minibatch_size:(n + 1) * minibatch_size])
        batched_data.append(tmp_data)
    # Last batch data
    if data[0].shape[0] % minibatch_size != 0:
        tmp_data = (data[0][num_batches * minibatch_size:],
                    data[1][num_batches * minibatch_size:])
        batched_data.append(tmp_data)

    return batched_data


def optimize_online_model_minibatch(model: OnlineGP, train_data: [np.ndarray, np.ndarray],
                                    test_data: [np.ndarray, np.ndarray],
                                    minibatch_size: int = 100, train_hyper: bool = True, train_mem: bool = True,
                                    n_hyp_opt_steps: int = 20) -> (list, list):
    """
    Optimize the Online GP model

    model: the OnlineGP model.
    train_data: A tuple of training data.
    test_data: A tuple of test data.
    minibatch_size: An integer value corresponding to the minibatch size. Defaults to 100
    train_hyper: A boolean variable for training the hyperparameters or not. Defaults to True.
    train_mem:  A boolean variable for training the memory or not. Defaults to True.
    n_hyp_opt_steps: An integer value corresponding to the number of hyperparameter optimization steps. Defaults to 20.

    returns: a list of NLPD and accuracy values.
    """
    batched_data = get_mini_batches(train_data, minibatch_size)

    nlpd_vals = []
    acc_vals = []
    for batch_data in batched_data:
        for var in model.optimizer.variables():
            var.assign(tf.zeros_like(var))
        model.update_with_new_batch(new_data=batch_data, train_hyps=train_hyper, n_hyp_opt_steps=n_hyp_opt_steps,
                                    train_mem=train_mem, remove_memory=False)

        nlpd = -1 * tf.reduce_mean(model.model.predict_log_density(test_data))
        nlpd_vals.append(nlpd)

        pred_m, _ = model.model.predict_y(test_data[0])
        pred_argmax = tf.reshape(tf.argmax(pred_m, axis=1), (-1, 1))
        acc = np.mean(pred_argmax == test_data[1])
        acc_vals.append(acc)

    return nlpd_vals, acc_vals


def optimize_streaming_model_minibatch(optimizer, model, train_data: Tuple[np.ndarray, np.ndarray],
                                       test_data: Tuple[np.ndarray, np.ndarray], iterations: int = 100,
                                       minibatch_size: int = 100, mu=None, Su=None, Kaa=None, Zopt=None,
                                       first_init=True):
    """
    Optimize the streaming model of Bui et al. 2017.

    The code is based on the official implementation: https://github.com/thangbui/streaming_sparse_gp
    """
    def optimization_step_adam():
        optimizer.minimize(model.training_loss, model.trainable_variables)

    def optimization_step_scipy():
        optimizer.minimize(model.training_loss, model.trainable_variables, options={'maxiter': iterations})

    def optimization_step():
        if isinstance(optimizer, gpflow.optimizers.Scipy):
            optimization_step_scipy()
        else:
            for _ in range(iterations):
                optimization_step_adam()

    def init_Z(cur_Z, new_X, use_old_Z=True):
        if use_old_Z:
            Z = np.copy(cur_Z)
        else:
            M = cur_Z.shape[0]
            M_old = int(0.7 * M)
            M_new = M - M_old
            old_Z = cur_Z[np.random.permutation(M)[0:M_old], :]
            new_Z = new_X[np.random.permutation(new_X.shape[0])[0:M_new], :]
            Z = np.vstack((old_Z, new_Z))
        return Z

    use_old_z = False
    nlpd_vals = []
    acc_vals = []

    batched_data = get_mini_batches(train_data, minibatch_size)
    for i, new_data in enumerate(batched_data):
        X, y = (new_data[0], new_data[1])

        if first_init:
            if isinstance(optimizer, gpflow.optimizers.Scipy):
                gpflow.optimizers.Scipy().minimize(
                    model.training_loss_closure((X, y)), model.trainable_variables, options={'maxiter': iterations})
            else:
                for _ in range(iterations):
                    optimizer.minimize(model.training_loss_closure((X, y)), model.trainable_variables)
            first_init = False
        else:
            Zinit = init_Z(Zopt, X, use_old_z)
            model = OSVGPC((X, y), gpflow.kernels.Matern52(), gpflow.likelihoods.Softmax(num_classes=10), mu, Su, Kaa,
                           Zopt, Zinit, num_latent_gps=10)
            optimization_step()

        Zopt = model.inducing_variable.Z.numpy()
        mu, Su = model.predict_f(Zopt, full_cov=True)
        if len(Su.shape) == 3:
            Su = Su[0, :, :] + 1e-4 * np.eye(mu.shape[0])
        Kaa = model.kernel(model.inducing_variable.Z)

        # NLPD calculation
        f_mean, f_var = model.predict_f(test_data[0])
        if len(f_var.shape) == 1:
            f_var = f_var[..., None]
        nlpd = model.likelihood.predict_log_density(f_mean, f_var, test_data[1])
        nlpd = -1 * tf.reduce_mean(nlpd)
        nlpd_vals.append(nlpd)

        # acc
        pred_m, _ = model.predict_y(test_data[0])
        pred_argmax = tf.reshape(tf.argmax(pred_m, axis=1), (-1, 1))
        acc = np.mean(pred_argmax == test_data[1])
        acc_vals.append(acc)

    return nlpd_vals, acc_vals, mu, Su, Kaa, Zopt, model
