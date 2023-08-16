"""
Module for the t-VGP model class
"""
from typing import Optional

import numpy as np
import tensorflow as tf
from gpflow.config import default_float, default_jitter
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import InternalDataTrainingLossMixin

from src.sites import DiagSites


class t_VGP(GPModel, InternalDataTrainingLossMixin):
    r"""
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior.

    The key reference is:
      Khan, M., & Lin, W. (2017). Conjugate-Computation Variational Inference:
      Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models.
      In Artificial Intelligence and Statistics (pp. 878-887).

    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent: Optional[int] = 1,
    ):
        """
        X is a data matrix, size [N, D]
        Y is a data matrix, size [N, R]
        kernel, likelihood, mean_function are appropriate GPflow objects

        """
        super().__init__(kernel, likelihood, mean_function, num_latent)

        x_data, y_data = data
        num_data = x_data.shape[0]
        self.num_data = num_data
        self.num_latent = num_latent or y_data.shape[1]
        self.data = data

        lambda_1 = np.zeros((num_data, self.num_latent))
        lambda_2 = 1e-6 * np.ones((num_data, self.num_latent))
        self.sites = DiagSites(lambda_1, lambda_2)

    @property
    def lambda_1(self):
        """first natural parameter"""
        return self.sites.lambda_1

    @property
    def lambda_2(self):
        """second natural parameter"""
        return self.sites.lambda_2

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        x_data, y_data = self.data
        pseudo_y = self.lambda_1 / self.lambda_2
        sW = tf.sqrt(tf.abs(self.lambda_2))

        # Computes conversion λ₁, λ₂ → m, V by using q(f) ≃ t(f)p(f)
        K = self.kernel(x_data) + tf.eye(self.num_data, dtype=default_float()) * default_jitter()
        # L = chol(I  + √λ₂ᵀ K √λ₂ᵀ)
        L = tf.linalg.cholesky(
            tf.eye(self.num_data, dtype=tf.float64) + (sW @ tf.transpose(sW)) * K
        )
        # T = L⁻¹ λ₂ K
        T = tf.linalg.solve(L, tf.tile(sW, (1, self.num_data)) * K)
        # Σ = (K⁻¹ + λ₂)⁻¹  = K - K √λ₂ (I  + √λ₂ᵀ K √λ₂ᵀ)⁻¹ √λ₂ᵀ K =  K - K √λ₂L⁻ᵀL⁻¹√λ₂ᵀ K
        post_v = tf.reshape(
            tf.linalg.diag_part(K) - tf.reduce_sum(T * T, axis=0), (self.num_data, 1)
        )
        # Σ = (K⁻¹ + λ₂)⁻¹ = (K⁻¹(I + λ₂K))⁻¹ = K (I + λ₂K)⁻¹ = K L⁻ᵀL⁻¹
        # μ =  Σ λ₁ = K L⁻ᵀL⁻¹ λ₂ (λ₂⁻¹λ₁) = K α
        alpha = sW * tf.linalg.solve(tf.transpose(L), tf.linalg.solve(L, sW * pseudo_y))
        post_m = K @ alpha
        # Store alpha for prediction
        self.q_alpha = alpha

        # Get variational expectations.
        # ELBO = E_q log(p(y,f)/q(t)) = E_q log(p(y|f)p(f))/Z⁻¹ p(f)t(f))
        # = log(Z) - E_q log t(f) + E_q log p(y|f)
        # log_Z = \int p(f)t(f)df
        E_q_log_lik = tf.reduce_sum(
            self.likelihood.variational_expectations(post_m, post_v, y_data)
        )
        E_q_log_t = -tf.reduce_sum(0.5 * (self.lambda_2) * ((pseudo_y - post_m) ** 2 + post_v))
        log_Z = -tf.transpose(pseudo_y) @ alpha / 2.0 - tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(L))
        )
        elbo = log_Z - E_q_log_t + E_q_log_lik
        return elbo

    def update_variational_parameters(self, beta=0.05) -> tf.Tensor:
        """Takes natural gradient step in Variational parameters in the local parameters
        λₜ = rₜ▽[Var_exp] + (1-rₜ)λₜ₋₁
        Input:
        :param: X : N x D
        :param: Y:  N x 1
        :param: lr: Scalar

        Output:
        Updates the params
        """

        x_data, y_data = self.data
        pseudo_y = self.lambda_1 / self.lambda_2
        sW = tf.sqrt(tf.abs(self.lambda_2))

        # Computes conversion λ₁, λ₂ → m, V by using q(f) ≃ t(f)p(f)
        K = self.kernel(x_data) + tf.eye(self.num_data, dtype=default_float()) * default_jitter()
        L = tf.linalg.cholesky(
            tf.eye(self.num_data, dtype=tf.float64) + (sW @ tf.transpose(sW)) * K
        )
        T = tf.linalg.solve(L, tf.tile(sW, (1, self.num_data)) * K)
        post_v = tf.reshape(
            tf.linalg.diag_part(K) - tf.reduce_sum(T * T, axis=0), (self.num_data, 1)
        )
        alpha = sW * tf.linalg.solve(tf.transpose(L), tf.linalg.solve(L, sW * pseudo_y))
        post_m = K @ alpha

        # Keep alphas updated
        self.q_alpha = alpha

        # Get variational expectations derivatives.
        with tf.GradientTape(persistent=True) as g:
            g.watch(post_m)
            g.watch(post_v)
            var_exp = self.likelihood.variational_expectations(post_m, post_v, y_data)

        d_exp_dm = g.gradient(var_exp, post_m)
        d_exp_dv = g.gradient(var_exp, post_v)
        del g

        # Take the tVGP step and transform to be ▽μ[Var_exp]
        lambda_1 = (1.0 - beta) * self.lambda_1 + beta * (d_exp_dm - 2.0 * (d_exp_dv * post_m))
        lambda_2 = (1.0 - beta) * self.lambda_2 + beta * (-2.0 * d_exp_dv)

        self.lambda_1.assign(lambda_1)
        self.lambda_2.assign(lambda_2)

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        The posterior variance of F is given by
            q(f) = N(f | K alpha + mean, [K⁻¹ + diag(lambda²)]⁻¹)
        Here we project this to F*, the values of the GP at Xnew which is given
        by
           q(F*) = N ( F* | K_{*F} alpha + mean, K_{**} - K_{*f}[K_{ff} +
                                           diag(lambda⁻²)]⁻¹ K_{f*} )

        """
        assert full_output_cov is False
        x_data, _y_data = self.data

        # Evaluate the kernel
        Kx = self.kernel(x_data, Xnew)
        K = self.kernel(x_data)

        # Predictive mean
        f_mean = tf.linalg.matmul(Kx, self.q_alpha, transpose_a=True) + self.mean_function(Xnew)

        # Predictive var
        A = K + tf.linalg.diag(tf.transpose(1.0 / self.lambda_2))
        L = tf.linalg.cholesky(A)
        Kx_tiled = tf.tile(Kx[None, ...], [self.num_latent, 1, 1])
        LiKx = tf.linalg.solve(L, Kx_tiled)
        if full_cov:
            f_var = self.kernel(Xnew) - tf.linalg.matmul(LiKx, LiKx, transpose_a=True)
        else:
            f_var = self.kernel(Xnew, full_cov=False) - tf.reduce_sum(tf.square(LiKx), 1)
        return f_mean, tf.transpose(f_var)
