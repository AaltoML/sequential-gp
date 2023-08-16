"""
Module for the t-SVGP models with individual sites per data point.
"""
from typing import Optional

import numpy as np
import tensorflow as tf
from gpflow import default_jitter, kullback_leiblers
from gpflow.conditionals import conditional
from gpflow.covariances import Kuf, Kuu
from gpflow.models import GPModel
from gpflow.models.training_mixins import InputData, RegressionData
from gpflow.models.util import inducingpoint_wrapper
#from gpflow.types import MeanAndVariance

from src.sites import DiagSites
from src.util import posterior_from_dense_site_white, project_diag_sites


class t_SVGP_sites(GPModel):
    """
    Class for the t-SVGP model with sites
    """

    def __init__(
        self,
        data: RegressionData,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        lambda_1=None,
        lambda_2=None,
        num_latent: Optional[int] = 1
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        GPModel.__init__(self, kernel, likelihood, mean_function, num_latent_gps)
        x_data, y_data = data
        num_data = x_data.shape[0]
        self.num_data = num_data
        self.num_latent = num_latent or y_data.shape[1]
        self.data = data
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        self.num_inducing = self.inducing_variable.num_inducing
        self._init_variational_parameters(self.num_data, lambda_1, lambda_2)
        self.whiten = False

    def _init_variational_parameters(self, num_inducing, lambda_1, lambda_2):
        """
        Constructs the site parameters λ₁, Λ₂.
        for site t(u) = exp(uᵀλ₁ - ½ uᵀΛ₂u)

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically referred to as M.
        :param lambda_1: np.array or None
            First order natural parameter of the variational site.
        :param lambda_2: np.array or None
            Second order natural parameter of the variational site.
        """

        lambda_1 = np.zeros((num_inducing, self.num_latent_gps)) if lambda_1 is None else lambda_1
        if lambda_2 is None:
            lambda_2 = (
                np.ones((num_inducing, self.num_latent_gps)) * 1e-6
                if lambda_2 is None
                else lambda_2
            )
        else:
            assert lambda_2.ndim == 2
            self.num_latent_gps = lambda_2.shape[-1]

        self.sites = DiagSites(lambda_1, lambda_2)

    @property
    def lambda_1(self):
        """first natural parameter"""
        return self.sites.lambda_1

    @property
    def lambda_2(self):
        """second natural parameter"""
        return self.sites.lambda_2

    def get_mean_chol_cov_inducing_posterior(self):
        """
        Computes the mean and cholesky factor of the posterior
        on the inducing variables q(u) = 𝓝(u; m, S)
        S = (K⁻¹ + Λ₂)⁻¹ = (K⁻¹ + L₂L₂ᵀ)⁻¹ = K - KL₂W⁻¹L₂ᵀK , W = (I + L₂ᵀKL₂)⁻¹
        m = S λ₁
        """
        X, _ = self.data
        K_uu = Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [P, M, M] or [M, M]
        K_uf = Kuf(self.inducing_variable, self.kernel, X)  # [P, M, M] or [M, M]
        lambda_1, lambda_2 = project_diag_sites(K_uf, self.lambda_1, self.lambda_2, cholesky=False)
        return posterior_from_dense_site_white(K_uu, lambda_1, lambda_2)

    def natgrad_step(self, lr=0.1):
        """Takes natural gradient step in Variational parameters in the local parameters
        λₜ = rₜ▽[Var_exp] + (1-rₜ)λₜ₋₁
        Input:
        :param: X : N x D
        :param: Y:  N x 1
        :param: lr: Scalar

        Output:
        Updates the params
        """
        X, Y = self.data
        mean, var = self.predict_f(X)

        with tf.GradientTape() as g:
            g.watch([mean, var])
            ve = self.likelihood.variational_expectations(mean, var, Y)
        grads = g.gradient(ve, [mean, var])

        grads = grads[0] - 2.0 * grads[1] * mean, grads[1]

        # compute update in natural form
        lambda_2 = -0.5 * self.lambda_2
        lambda_1 = self.lambda_1

        lambda_1 = (1 - lr) * lambda_1 + lr * grads[0]
        lambda_2 = (1 - lr) * lambda_2 + lr * grads[1]

        eps = 1e-8
        # crop hack, can't instantiate negative sites nats2 but optim might take you there
        lambda_2 = tf.minimum(lambda_2, -eps * tf.ones_like(lambda_2))

        # To match SVGP you need to eliminate this jitter for minibatching
        self.lambda_1.assign(lambda_1)
        self.lambda_2.assign(-2.0 * lambda_2)

    def prior_kl(self) -> tf.Tensor:
        """Returns the KL divergence KL[q(u)|p(u)]"""
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, q_mu, q_sqrt, whiten=self.whiten
        )

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """The variational lower bound"""
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = self.data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)

        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> None:
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        return mu + self.mean_function(Xnew), var
