"""
Utility function for UCI datasets.
"""
import time
from typing import Tuple
import os

import gpflow.models
import tensorflow as tf
import pandas as pd
import numpy as np
from gpflow.likelihoods import Bernoulli
from sklearn.preprocessing import StandardScaler
from gpflow.models.svgp import SVGP
from sklearn.model_selection import KFold
import wandb
from omegaconf import OmegaConf

import sys

sys.path.append("../..")
sys.path.append("..")

from src.streaming_sparse_gp.osgpr import OSGPR_VFE
from src.streaming_sparse_gp.osvgpc import OSVGPC


def setup_wandb(cfg):
    """
    Set up wandb.
    """
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(project="UCI", entity=cfg.wandb.username, config=wandb_cfg)


def load_data(data_path: str, train_split_percentage: float = 0.8, normalize: bool = False,
              seed: int = None, n_k_folds: int = None, random_state: int = None,
              dataset_type: str = "regression") -> (Tuple[np.ndarray, np.ndarray],
                                                    Tuple[np.ndarray, np.ndarray]):
    """
    Load UCI dataset on the basis of data name.

    If k_folds is passed then a list of several folds are returned.

    returns a list of set of (X, Y) as Tuple as train_data and test_data.
    """
    if seed is not None:
        np.random.seed(seed)

    if not os.path.exists(data_path):
        raise Exception("Data path does not exist ")

    df = pd.read_csv(data_path)
    X = df.to_numpy()[:, :-1]
    Y = df.to_numpy()[:, -1].reshape((-1, 1))

    if normalize:
        x_scaler = StandardScaler().fit(X)
        X = x_scaler.transform(X)

        if dataset_type == "regression":
            y_scaler = StandardScaler().fit(Y)
            Y = y_scaler.transform(Y)

    if n_k_folds is None:
        data_dim = X.shape[-1]
        n = Y.shape[0]

        # combine X and Y and shuffle
        XY = np.concatenate([X, Y], axis=1)
        np.random.shuffle(XY)

        n_train = int(np.floor(n * train_split_percentage))

        x_train = XY[:n_train, :data_dim]
        y_train = XY[:n_train, data_dim:]

        x_test = XY[n_train:, :data_dim]
        y_test = XY[n_train:, data_dim:]

        if dataset_type == "classification":
            y_train = y_train.astype(np.int64)
            y_test = y_test.astype(np.int64)

        train_data = [(x_train, y_train)]
        test_data = [(x_test, y_test)]
    else:
        train_data, test_data = get_cross_validation_sets((X, Y), k_folds=n_k_folds, random_state=random_state)

    return train_data, test_data


def load_model_parameters(model, params: dict):
    """Loads the parameters from dictionary to the model"""
    gpflow.utilities.multiple_assign(model, params)


def get_cross_validation_sets(data: Tuple[np.ndarray, np.ndarray], k_folds=5, random_state: int = None):
    """
    Split the dataset for K-Fold validation.
    """

    kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)

    train_k_folds_set = []
    test_k_folds_set = []
    for train_idx, test_idx in kf.split(data[0]):
        train_k_folds_set.append((data[0][train_idx], data[1][train_idx]))
        test_k_folds_set.append((data[0][test_idx], data[1][test_idx]))

    return train_k_folds_set, test_k_folds_set


def optimize_streaming_model(optimizer, model, train_data: Tuple[np.ndarray, np.ndarray],
                             test_data: Tuple[np.ndarray, np.ndarray], task: str, iterations: int = 100,
                             use_old_z=False, fast_conditioning=False):
    """
    Optimize the streaming model of Bui et al. 2017.

    The code is based on the official implementation: https://github.com/thangbui/streaming_sparse_gp
    """

    @tf.function
    def optimization_step_adam():
        for _ in range(iterations):
            optimizer.minimize(model.training_loss, model.trainable_variables)

    # @tf.function
    def optimization_step_adam_classification(loss, variables):
        for _ in range(iterations):
            optimizer.minimize(loss, variables)

    def optimization_step_scipy():
        optimizer.minimize(model.training_loss, model.trainable_variables, options={'maxiter': iterations})

    def optimization_step():
        if isinstance(optimizer, gpflow.optimizers.Scipy):
            optimization_step_scipy()
        else:
            optimization_step_adam()

    def get_model_prediction():
        Zopt = model.inducing_variable.Z.numpy()
        mu, Su = model.predict_f(Zopt, full_cov=True)
        if len(Su.shape) == 3:
            Su = Su[0, :, :]

        return mu, Su, Zopt

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

    n_sets = len(train_data)

    # NLPD calculation
    f_mean, f_var = model.predict_f(test_data[0])
    if len(f_var.shape) == 1:
        f_var = f_var[..., None]
    nlpd = model.likelihood.predict_log_density(f_mean, f_var, test_data[1])
    nlpd = -1 * tf.reduce_mean(nlpd)
    print(f"Initial NLPD: {nlpd}")

    nlpd_vals = []
    evaluation_vals = []
    time_vals = []
    for n in range(n_sets):
        new_data = train_data[n]
        X, y = (new_data[0], new_data[1])

        start_time = time.time()
        if task == "regression":
            if n == 0:
                optimization_step()

                mu, Su, Zopt = get_model_prediction()
            else:
                Kaa1 = model.kernel(model.inducing_variable.Z)

                Zinit = init_Z(Zopt, X, use_old_z)
                var = model.likelihood.variance
                if isinstance(model.kernel, gpflow.kernels.Matern52):
                    kernel = gpflow.kernels.Matern52(variance=model.kernel.variance,
                                                     lengthscales=model.kernel.lengthscales)
                else:  # For running Magnetometer.
                    kernel = gpflow.kernels.Sum([gpflow.kernels.Constant(model.kernel.kernels[0].variance),
                                                 gpflow.kernels.Matern52(
                                                     lengthscales=model.kernel.kernels[1].lengthscales,
                                                     variance=model.kernel.kernels[1].variance)])

                model = OSGPR_VFE((X, y), kernel, mu, Su, Kaa1, Zopt, Zinit)
                model.likelihood.variance.assign(var)

                optimization_step()

                mu, Su, Zopt = get_model_prediction()
        else:
            if n == 0:
                if isinstance(optimizer, gpflow.optimizers.Scipy):
                    gpflow.optimizers.Scipy().minimize(
                        model.training_loss_closure((X, y)), model.trainable_variables, options={'maxiter': iterations})
                else:
                    for _ in range(iterations):
                        optimizer.minimize(model.training_loss_closure((X, y)), model.trainable_variables)
            else:
                Zinit = init_Z(Zopt, X, use_old_z)
                if fast_conditioning:
                    kernel = model.kernel
                else:
                    kernel = gpflow.kernels.Matern52()
                model = OSVGPC((X, y), kernel, gpflow.likelihoods.Bernoulli(), mu, Su, Kaa, Zopt,
                               Zinit)
                optimization_step_adam_classification(model.training_loss, model.trainable_variables)

            Zopt = model.inducing_variable.Z.numpy()
            mu, Su = model.predict_f(Zopt, full_cov=True)
            if len(Su.shape) == 3:
                Su = Su[0, :, :] + 1e-4 * np.eye(mu.shape[0])
            Kaa = model.kernel(model.inducing_variable.Z)

        time_vals.append(time.time() - start_time)

        # NLPD calculation
        f_mean, f_var = model.predict_f(test_data[0])
        if len(f_var.shape) == 1:
            f_var = f_var[..., None]
        nlpd = model.likelihood.predict_log_density(f_mean, f_var, test_data[1])
        nlpd = -1 * tf.reduce_mean(nlpd)
        nlpd_vals.append(nlpd)

        # RMSE calculation
        if task == "regression":
            y_pred, _ = model.likelihood.predict_mean_and_var(f_mean, f_var)
            rmse = np.sqrt(np.mean(np.square(y_pred - test_data[1])))
            evaluation_vals.append(rmse)
        else:
            pred_mean, _ = model.likelihood.predict_mean_and_var(f_mean, f_var)
            pred_mean = pred_mean.numpy()
            pred_mean[pred_mean >= 0.5] = 1
            pred_mean[pred_mean < 0.5] = 0
            correct_prediction = np.sum(pred_mean == test_data[1])
            acc = correct_prediction / test_data[0].shape[0]
            evaluation_vals.append(acc)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Set {n}")
        print(f"NLPD = {nlpd_vals[-1]}")
        print(f"Eval. metric (RMSE/Acc.) = {evaluation_vals[-1]}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return nlpd_vals, evaluation_vals, time_vals
