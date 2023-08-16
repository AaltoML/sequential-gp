"""
Module for continual learning t-SVGP model with whitened parameterization.
"""
from __future__ import annotations

import gpflow.optimizers
from gpflow import default_float, default_jitter
from gpflow.covariances import Kuf, Kuu
from gpflow.models import GPModel
from gpflow.models.model import RegressionData
from gpflow.models.training_mixins import InputData
from gpflow.models.util import inducingpoint_wrapper

from src.sites import DenseSites
from src.util import (
    conditional_from_precision_sites_white,
    kl_from_precision_sites_white,
)

from src.models.utils import (
    piv_chol,
    update_lambda_Z_move,
    memory_picker,
)

from src.models.tsvgp import base_SVGP
from typing import Union

import numpy as np
import tensorflow as tf


class t_SVGP_cont(base_SVGP):
    """
    Class for the continual learning t-SVGP model. 
    N.B. We us the natural parameterization without the Kzz matrix so natural_param_2 = Kzz^{-1} * lambda2 * Kzz^{-1}.

    Args:
        kernel: GPflow kernel object
        likelihood: GPflow likelihood object
        inducing_variable: GPflow inducing variable object
        mean_function: GPflow mean function object, defaults to None
        num_latent_gps: int, the number of latent processes to use, defaults to 1
        lambda_1: float, the first variational parameter, defaults to None (num_latent_gps, num_inducing)
        lambda_2: float, the second variational parameter, defaults to None (num_latent_gps, num_inducing, num_inducing)
        num_data: int, the total number of observations, defaults to None
        num_laten_gps: int, the number of latent processes, defaults to 1

    Methods:
        __init__: initializes the t_SVGP_cont object
        _init_variational_parameters: initializes the variational parameters
        predict_f: returns the mean and variance of the latent function at the new data points
        prior_kl: returns the KL divergence between the prior and variational distribution
        online_update: updates the posterior based on some new data
        elbo: returns the evidence lower bound (ELBO) of the model
    """

    def __init__(
            self,
            kernel,
            likelihood,
            inducing_variable,
            *,
            mean_function=None,
            num_latent_gps: int = 1,
            lambda_1=None,
            lambda_2=None,
            num_data=None,
    ):
        GPModel.__init__(self, kernel, likelihood, mean_function, num_latent_gps)

        self.num_data = num_data
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        self.num_inducing = self.inducing_variable.num_inducing

        self._init_variational_parameters(self.num_inducing, lambda_1, lambda_2)

    def _init_variational_parameters(self, num_inducing, lambda_1, lambda_2):
        lambda_1 = np.zeros((num_inducing, self.num_latent_gps)) if lambda_1 is None else lambda_1

        if lambda_2 is None:
            lambda_2 = [
                tf.eye(num_inducing, dtype=default_float()) * default_jitter()
                for _ in range(self.num_latent_gps)
            ]
            lambda_2 = np.array(lambda_2)
        else:
            assert lambda_2.ndim == 3
            self.num_latent_gps = lambda_2.shape[0]

        self.sites = DenseSites(lambda_1=lambda_1, lambda_2=lambda_2)

    @property
    def lambda_1(self):
        return self.sites.lambda_1

    @property
    def lambda_2(self):
        return self.sites.lambda_2

    def prior_kl(self) -> tf.Tensor:
        K_uu = Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  
        return kl_from_precision_sites_white(K_uu, self.lambda_1, L2=self.lambda_2)

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> Union[tf.Tensor, tf.Tensor]:
        K_uu = Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        ) 
        K_uf = Kuf(self.inducing_variable, self.kernel, Xnew)
        K_ff = self.kernel.K_diag(Xnew)[..., None]

        mu, var = conditional_from_precision_sites_white(
            K_uu, K_ff, K_uf, self.lambda_1, L2=self.lambda_2)

        return mu + self.mean_function(Xnew), var

    def online_update(self, lambda_1_t, lambda_2_t, extra_data=RegressionData, lr=1.0,
                      jitter=default_jitter()) -> Union[tf.Tensor, tf.Tensor]:

        grad_mu = self.grad_varexp_natural_params(extra_data) 

        lambda_1_t_new = (1.0 - lr) * lambda_1_t + lr * grad_mu[0]
        lambda_2_t_new = (1.0 - lr) * lambda_2_t + lr * (-2) * grad_mu[1]

        return lambda_1_t_new, lambda_2_t_new

    def elbo(self, data: RegressionData, memory: RegressionData, scale) -> tf.Tensor:
        kl = self.prior_kl()
        X, Y = data
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)

        if memory is not None: # adds memory term to the ELBO
            X_m, Y_m = memory
            f_mean_m, f_var_m = self.predict_f(X_m, full_cov=False, full_output_cov=False)
            var_exp_m = self.likelihood.variational_expectations(f_mean_m, f_var_m, Y_m)
        else:
            var_exp_m = tf.cast(0.0, kl.dtype)

        return tf.reduce_sum(var_exp) + scale*var_exp_m - kl


    def grad_varexp_natural_params(self, data, jitter=1e-9, nat_params=None):
        X, Y = data

        mean, var = self.predict_f(X)

        with tf.GradientTape(persistent=True) as g:
            g.watch(mean)
            g.watch(var)
            ve = self.likelihood.variational_expectations(mean, var, Y)
        d_exp_dm = g.gradient(ve, mean)
        d_exp_dv = g.gradient(ve, var)
        del g

        eps = 1e-8
        d_exp_dv = tf.minimum(d_exp_dv, -eps * tf.ones_like(d_exp_dv))

        grad_nat_1 = (d_exp_dm - 2.0 * (d_exp_dv * mean))
        grad_nat_2 = d_exp_dv

        K_uf = Kuf(self.inducing_variable, self.kernel, X)

        grad_sparse_1 = K_uf @ grad_nat_1

        grad_sparse_2 = K_uf @ tf.linalg.diag(tf.transpose(grad_nat_2)) @ tf.transpose(K_uf)

        return grad_sparse_1, grad_sparse_2

class OnlineGP:
    """
    A wrapper class for t_SVGP_cont to allow online updates.

    Args:
        model: a t_SVGP_cont object representing the GP model
        opt_hypers: a GPflow optimizer object for optimizing the hyperparameters, defaults to None
        n_steps: an integer, the number of steps to use for updating the dual parameters, defaults to 2
        lambda_lr: a float, the learning rate for updating the dual parameters, defaults to 1.0
        Z_picker: a function for selecting the inducing points, defaults to piv_chol
        num: an integer, the number of inducing points to use, defaults to 0
        memory: a list of tuples representing the memory of the model, defaults to None
        Z_y: a numpy array of shape (num_data, num_latent_gps), the function values at the inducing points, defaults to None
        memory_picker: a function for selecting the memory points, defaults to memory_picker
        num_mem: an integer, the number of memory points to use, defaults to 5

    Methods:
        update_with_new_batch: performs the main update for the model hyps, Lambda and Z.
        _move_Z: moves the inducing points to new locations based on the input data
        _update_Lambda: updates the dual parameters using the input data and learning rate
        _update_memory: updates the memory of the model
        _optimization_step: defines the optimization step for the model hyps
    """
    def __init__(self, model: t_SVGP_cont, opt_hypers=None, n_steps=2, lambda_lr=1.0,
                 Z_picker=piv_chol, num=0, memory=None,
                 memory_picker=memory_picker, num_mem=5):

        self.model = model
        self.optimizer = opt_hypers
        self.lam_steps = n_steps
        self.lam_lr = lambda_lr
        self.Z_picker = Z_picker
        self.num_p = num
        self.memory_picker = memory_picker
        self.mem_n = num_mem
        self.memory = memory
        self.lamb1_mem = tf.Variable(np.zeros((model.num_inducing, model.num_latent_gps)))
        self.lamb2_mem = tf.Variable(np.zeros((model.num_latent_gps, model.num_inducing, model.num_inducing)))

    def _move_Z(self, data=None):
        x, _ = data
        M = self.model.num_inducing
        Z_old = self.model.inducing_variable.Z.numpy()
        Z_n, ind = self.Z_picker(x, Z_old, self.model, M)
        new_l1, new_l2 = update_lambda_Z_move(self.model, Z_n, Z_old)
        self.model.lambda_1.assign(new_l1)
        self.model.lambda_2.assign(new_l2)
        self.model.inducing_variable.Z.assign(Z_n)
        return Z_old, Z_n

    def _update_Lambda(self, data=None, l_lr=1.0, n_steps=None):
        model = self.model
        lambda_1_t = np.zeros((model.num_inducing, model.num_latent_gps))
        lambda_2_t = np.zeros((model.num_latent_gps, model.num_inducing, model.num_inducing))

        for i in range(n_steps):
            lambda_1_t_new, lambda_2_t_new = self.model.online_update(lambda_1_t, lambda_2_t, data, lr=l_lr)

            lambda_1_new = model.lambda_1 - lambda_1_t + lambda_1_t_new
            lambda_2_new = model.lambda_2 - lambda_2_t + lambda_2_t_new

            model.lambda_1.assign(lambda_1_new)
            model.lambda_2.assign(lambda_2_new)
            lambda_1_t, lambda_2_t = lambda_1_t_new, lambda_2_t_new
        return lambda_1_t_new, lambda_2_t_new

    def _update_memory(self, new_data):

        x_new, y_new = new_data
        _, ind = self.memory_picker(new_data, self.model, self.mem_n)
        
        if self.memory is None:
            new_mem_x = x_new[ind]
            new_mem_y = y_new[ind]
        else:
            x_m, y_m = self.memory

            new_mem_x = np.concatenate([x_m, x_new[ind]], axis=0)
            new_mem_y = np.concatenate([y_m, y_new[ind]], axis=0)

        self.memory = (new_mem_x, new_mem_y)

    def update_with_new_batch(self, new_data, n_hyp_opt_steps=20, train_hyps=True, train_mem=True,
                              remove_memory: bool = True, return_kernel_params=True):
        """
        Updates the model with new data.

        This method performs the main update for the model hyperparameters, dual parameters, and inducing points
        using the new data. It first updates the inducing points using the new data, then updates the dual parameters
        using the updated inducing points, and finally updates the model hyperparameters using the updated dual parameters.

        Args:
            new_data: a tuple containing (x_new,y_new), the new data to update the model with.
            n_hyp_opt_steps: an integer, the number of optimization steps to perform for the model hyperparameters.
            train_hyps: a boolean, whether to train the model hyperparameters.
            train_mem: a boolean, whether to train lambdas using the memory.
            remove_memory: a boolean, whether to remove memory before update.
            _update_n: an integer, the number of total datapoints seen
        Returns:
            kernel_hyperparam: Used for plotting the kernel hyperparameters over time.
        """
        x, _ = new_data
        kernel_hyperparam = []

        optimizer = self.optimizer
        if self.mem_n > 0:
            scale = self.num_p/self.mem_n
        else:
            scale = 0

        Z_old, Z_new = self._move_Z(new_data)
        self._update_Lambda(new_data, l_lr=self.lam_lr, n_steps=self.lam_steps)

        if self.memory is not None:
            if remove_memory:
                self._remove_memory(Z_old, Z_new)
            lamb1_mem, lamb2_mem = self._update_Lambda(self.memory, l_lr=self.lam_lr, n_steps=self.lam_steps)
            self.lamb1_mem.assign(lamb1_mem)
            self.lamb2_mem.assign(lamb2_mem)

        if train_hyps:
            if isinstance(optimizer, gpflow.optimizers.Scipy):
                def training_objective():
                    return -self.model.elbo(new_data, self.memory, scale)
                optimizer.minimize(training_objective, self.model.trainable_variables,
                                   options={'maxiter': n_hyp_opt_steps})
            else:
                for i in range(n_hyp_opt_steps):
                    self._optimization_step(optimizer, new_data, self.memory, scale)
                    if return_kernel_params:
                        kernel_hyperparam.append((self.model.kernel.lengthscales.numpy().item(),
                                                  self.model.kernel.variance.numpy().item()))

        n = x.shape[0]
        self._update_n(n)

        if train_mem:
            self._update_memory(new_data)

        return kernel_hyperparam

    @tf.function
    def _optimization_step(self, optimizer=None, batch=None, memory=None, scale=None):

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.model.trainable_variables)
            objective = - self.model.elbo(batch, memory, scale)

        grads = tape.gradient(objective, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return objective

    def _update_n(self, n):
        self.num_p = self.num_p + n

    def _remove_memory(self, z_old, z_new):
        model = self.model

        old_l1 = self.lamb1_mem
        old_l2 = self.lamb2_mem

        K_zf = model.kernel(z_old, z_new)
        A_p = tf.linalg.solve(model.kernel(z_old), K_zf)

        new_l1 = tf.transpose(A_p) @ old_l1
        new_l2 = tf.transpose(A_p) @ old_l2 @ A_p

        lambda_1_new = model.lambda_1 - new_l1
        lambda_2_new = model.lambda_2 - new_l2

        model.lambda_1.assign(lambda_1_new)
        model.lambda_2.assign(lambda_2_new)

        return
