# Copyright (c) 2023 AaltoML group
# Author: github.com/st--

from typing import Union
from copy import deepcopy

import torch

from botorch.models.gpytorch import GPyTorchModel
from botorch import settings

from gpytorch import lazify
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import (
    DiagLazyTensor,
    CholLazyTensor,
    TriangularLazyTensor,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)

from volatilitygp.utils import pivoted_cholesky_init
from volatilitygp.models.single_task_variational_gp import _update_caches


class _OurSingleTaskVariationalGP(ApproximateGP):
    def __init__(
        self,
        init_points=None,
        likelihood=None,
        learn_inducing_locations=True,
        covar_module=None,
        mean_module=None,
        use_piv_chol_init=True,
        num_inducing=None,
        use_whitened_var_strat=True,
        init_targets=None,
        train_inputs=None,
        train_targets=None,
    ):

        if covar_module is None:
            covar_module = ScaleKernel(RBFKernel())

        if use_piv_chol_init:
            if num_inducing is None:
                num_inducing = int(init_points.shape[-2] / 2)

            if num_inducing < init_points.shape[-2]:
                covar_module = covar_module.to(init_points)

                covariance = covar_module(init_points)
                if init_targets is not None and init_targets.shape[-1] == 1:
                    init_targets = init_targets.squeeze(-1)
                if likelihood is not None and not isinstance(
                    likelihood, GaussianLikelihood
                ):
                    _ = likelihood.newton_iteration(
                        init_points, init_targets, model=None, covar=covariance
                    )
                    if likelihood.has_diag_hessian:
                        hessian_sqrt = likelihood.expected_hessian().sqrt()
                    else:
                        hessian_sqrt = (
                            lazify(likelihood.expected_hessian())
                            .root_decomposition()
                            .root
                        )
                    covariance = hessian_sqrt.matmul(covariance).matmul(
                        hessian_sqrt.transpose(-1, -2)
                    )
                inducing_points = pivoted_cholesky_init(
                    init_points, covariance.evaluate(), num_inducing
                )
            else:
                inducing_points = init_points.detach().clone()
        else:
            inducing_points = init_points.detach().clone()

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.shape[-2]
        )
        if use_whitened_var_strat:
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
            )
        else:
            variational_strategy = UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
            )
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean() if mean_module is None else mean_module
        self.mean_module.to(init_points)
        self.covar_module = covar_module

        self.likelihood = GaussianLikelihood() if likelihood is None else likelihood
        self.likelihood.to(init_points)
        self.train_inputs = [train_inputs] if train_inputs is not None else [init_points]
        self.train_targets = train_targets if train_targets is not None else init_targets

        self.condition_into_exact = True

        self.to(init_points)
        
        self.num_online_updates = 1
        self.lr = 1.0

    def make_copy(self):
        with torch.no_grad():
            inducing_points = self.variational_strategy.inducing_points.detach().clone()

            if hasattr(self, "input_transform"):
                [p.detach_() for p in self.input_transform.buffers()]

            new_covar_module = deepcopy(self.covar_module)

            new_model = self.__class__(
                init_points=inducing_points,
                likelihood = deepcopy(self.likelihood),
                use_piv_chol_init=False,
                mean_module = deepcopy(self.mean_module),
                covar_module=deepcopy(self.covar_module),
                input_transform=deepcopy(self.input_transform)
                if hasattr(self, "input_transform")
                else None,
                outcome_transform=deepcopy(self.outcome_transform)
                if hasattr(self, "outcome_transform")
                else None,
            )
            
            var_dist = self.variational_strategy._variational_distribution
            mean = var_dist.variational_mean.detach().clone()
            cov_root = var_dist.chol_variational_covar.detach().clone()

            new_var_dist = new_model.variational_strategy._variational_distribution
            with torch.no_grad():
                new_var_dist.variational_mean.set_(mean)
                new_var_dist.chol_variational_covar.set_(cov_root)
            
            new_model.variational_strategy.variational_params_initialized.fill_(1)

        return new_model
    
    def get_fantasy_model(
        self,
        inputs,
        targets,
        noise=None,
        condition_into_sgpr=False,
        targets_are_gaussian=True,
        **kwargs,
    ):
        X = inputs
        Y = targets
        
        # make copy of self
        fantasy_model = self.make_copy()
        inducing_points = self.variational_strategy.inducing_points

        # mean and cov from prediction strategy
        var_cov_root = TriangularLazyTensor(
            self.variational_strategy._variational_distribution.chol_variational_covar
        )
        var_cov = CholLazyTensor(var_cov_root)
        var_mean = (
            self.variational_strategy.variational_distribution.mean
        )  # .unsqueeze(-1)
        if var_mean.shape[-1] != 1:  # TODO: won't work for M=1 ...
            var_mean = var_mean.unsqueeze(-1)
            
        # GPyTorch's way of computing Kuf:
        #full_inputs = torch.cat([inducing_points, X], dim=-2)
        full_inputs = torch.cat([torch.tile(inducing_points, X.shape[:-2] + (1,1)), X], dim=-2)
        full_covar = self.covar_module(full_inputs)

        # Covariance terms
        num_induc = inducing_points.size(-2)
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        
        K_uf = induc_data_covar

        #Kuu = self.covar_module(inducing_points)
        Kuu = induc_induc_covar
        
        chol_Kuu = Kuu[0].cholesky() # = L
        prior_mean = self.mean_module(inducing_points).unsqueeze(-1) # = µ
        # u = L v + µ
        # mean(u) = L mean(v) + µ
        # cov(u) = L cov(v) L^T
        unwhitened_var_mean = chol_Kuu.matmul(var_mean) + prior_mean
        unwhitened_var_cov = chol_Kuu.matmul(var_cov).matmul(chol_Kuu.t())
        lambda_1, lambda_2 = mean_cov_to_natural_param(unwhitened_var_mean, unwhitened_var_cov, Kuu)

        lambda_1_t = torch.zeros_like(lambda_1)
        lambda_2_t = torch.zeros_like(lambda_2)
        
        # online_update
        for _ in range(self.num_online_updates):
            # grad_varexp_natural_params
            with torch.no_grad():
                Xt = torch.tile(X, Y.shape[:-2] + (1,1,1))
#                 if Y.shape[-1] == 1:
#                     Xt.unsqueeze_(-1)
                pred = fantasy_model(Xt)
                mean = pred.mean
                var = pred.variance
            mean.requires_grad_()
            var.requires_grad_()

            # variational expectations
            f_dist = MultivariateNormal(mean, DiagLazyTensor(var))
            ve_terms = fantasy_model.likelihood.expected_log_prob(Y, f_dist)
            ve = ve_terms.sum()  # in principle we ought to be careful to check whether we should divide by num_data here; but we only add one point at a time, so in our use-case it does not matter
            
            ve.backward(inputs=[mean, var])
            d_exp_dm = mean.grad  # [batch, N]
            d_exp_dv = var.grad  # [batch, N]

            eps = 1e-8
            d_exp_dv.clamp_(max=-eps)

            grad_nat_1 = (d_exp_dm - 2.0 * (d_exp_dv * mean))
            grad_nat_2 = d_exp_dv

            grad_mu_1 = K_uf.matmul(grad_nat_1[..., None])

            grad_mu_2 = K_uf.matmul(DiagLazyTensor(grad_nat_2).matmul(K_uf.swapdims(-1,-2)))

            lr = self.lr
            scale = 1.0

            lambda_1_t_new = (1.0 - lr) * lambda_1_t + lr * scale * grad_mu_1
            lambda_2_t_new = (1.0 - lr) * lambda_2_t + lr * scale * (-2) * grad_mu_2
            
            lambda_1_new = lambda_1 - lambda_1_t + lambda_1_t_new
            lambda_2_new = lambda_2 - lambda_2_t + lambda_2_t_new
            
            unwhitened_mean, unwhitened_cov = conditional_from_precision_sites_white_full(
                Kuu, lambda_1_new, lambda_2_new,
                jitter=getattr(self, "tsvgp_jitter", 0.0)
            )
            unwhitened_mean = unwhitened_mean
            
            # u = L v + µ
            # L^{-1} (mean(u) - µ) = mean(v)
            # L^{-1} cov(u) L^{-T} = cov(v)
            # A = L^{-1} cov(u)
            # A L^{-T} = cov(v)
            # cov(v) = L^{-1} (L^{-1} cov(u))^T
            
            new_mean = chol_Kuu.inv_matmul(unwhitened_mean - prior_mean).squeeze(-1)
            new_cov = chol_Kuu.inv_matmul(chol_Kuu.inv_matmul(unwhitened_cov).swapdims(-1,-2))
            new_cov_root = new_cov.cholesky()
 
            fantasy_var_dist = fantasy_model.variational_strategy._variational_distribution
            with torch.no_grad():
                fantasy_var_dist.variational_mean.set_(new_mean)
                fantasy_var_dist.chol_variational_covar.set_(new_cov_root)
            
            lambda_1 = lambda_1_new
            lambda_2 = lambda_2_new
            lambda_1_t = lambda_1_t_new
            lambda_2_t = lambda_2_t_new
        
        return fantasy_model
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateNormal(mean_x, covar_x)
        return latent_pred

    def to(self, *args, **kwargs):
        _update_caches(self, *args, **kwargs)
        self.variational_strategy = self.variational_strategy.to(*args, **kwargs)
        _update_caches(self.variational_strategy, *args, **kwargs)
        return super().to(*args, **kwargs)


def mean_cov_to_natural_param(mu, Su, K_uu):
    """
    Transforms (m,S) to (λ₁,P) tsvgp_white parameterization
    """
    lamb1 = K_uu.matmul(Su.inv_matmul(mu))
    lamb2 = K_uu.matmul(Su.inv_matmul(K_uu.evaluate())) - K_uu.evaluate()

    return lamb1, lamb2


def conditional_from_precision_sites_white_full(
    Kuu,
    lambda1,
    Lambda2,
    jitter=1e-9,
):
    """
    Given a g₁ and g2, and distribution p and q such that
      p(g₂) = N(g₂; 0, Kuu)
      p(g₁) = N(g₁; 0, Kff)
      p(g₁ | g₂) = N(g₁; Kfu (Kuu⁻¹) g₂, Kff - Kfu (Kuu⁻¹) Kuf)
    And  q(g₂) = N(g₂; μ, Σ) such that
        Σ⁻¹  = Kuu⁻¹  + Kuu⁻¹LLᵀKuu⁻¹
        Σ⁻¹μ = Kuu⁻¹l
    This method computes the mean and (co)variance of
      q(g₁) = ∫ q(g₂) p(g₁ | g₂) dg₂ = N(g₂; μ*, Σ**)
    with
    Σ** = k** - kfu Kuu⁻¹ kuf - kfu Kuu⁻¹ Σ Kuu⁻¹ kuf
        = k** - kfu Kuu⁻¹kuf + kfu (Kuu + LLᵀ)⁻¹ kuf
    μ* = k*u Kuu⁻¹ m
       = k*u Kuu⁻¹ Λ⁻¹ Kuu⁻¹ l
       = k*u (Kuu + LLᵀ)⁻¹ l
    Inputs:
    :param Kuu: tensor M x M
    :param l: tensor M x 1
    :param L: tensor M x M
    """
    #TODO: rewrite this

    R = (Lambda2 + Kuu).add_jitter(jitter)
    
    mean = Kuu.matmul(R.inv_matmul(lambda1))
    cov = Kuu.matmul(R.inv_matmul(Kuu.evaluate()))  # TODO: symmetrise?
    return mean, cov

class OurSingleTaskVariationalGP(_OurSingleTaskVariationalGP, GPyTorchModel):
    def __init__(
        self,
        init_points=None,
        likelihood=None,
        learn_inducing_locations=True,
        covar_module=None,
        mean_module=None,
        use_piv_chol_init=True,
        num_inducing=None,
        use_whitened_var_strat=True,
        init_targets=None,
        train_inputs=None,
        train_targets=None,
        outcome_transform=None,
        input_transform=None,
    ):
        if outcome_transform is not None:
            is_gaussian_likelihood = (
                isinstance(likelihood, GaussianLikelihood) or likelihood is None
            )
            if train_targets is not None and is_gaussian_likelihood:
                if train_targets.ndim == 1:
                    train_targets = train_targets.unsqueeze(-1)
                train_targets, _ = outcome_transform(train_targets)

            if init_targets is not None and is_gaussian_likelihood:
                init_targets, _ = outcome_transform(init_targets)
                init_targets = init_targets.squeeze(-1)

        if train_targets is not None:
            train_targets = train_targets.squeeze(-1)

        # unlike in the exact gp case we need to use the input transform to pre-define the inducing pts
        if input_transform is not None:
            if init_points is not None:
                init_points = input_transform(init_points)

        _OurSingleTaskVariationalGP.__init__(
            self,
            init_points=init_points,
            likelihood=likelihood,
            learn_inducing_locations=learn_inducing_locations,
            covar_module=covar_module,
            mean_module=mean_module,
            use_piv_chol_init=use_piv_chol_init,
            num_inducing=num_inducing,
            use_whitened_var_strat=use_whitened_var_strat,
            init_targets=init_targets,
            train_inputs=train_inputs,
            train_targets=train_targets,
        )

        if input_transform is not None:
            self.input_transform = input_transform.to(
                self.variational_strategy.inducing_points
            )

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform.to(
                self.variational_strategy.inducing_points
            )

    def forward(self, x):
        x = self.transform_inputs(x)
        return super().forward(x)

    @property
    def num_outputs(self) -> int:
        # we should only be able to have one output without a multitask variational strategy here
        return 1

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: Union[bool, torch.Tensor] = False,
        **kwargs,
    ):
        if observation_noise and not isinstance(self.likelihood, _GaussianLikelihoodBase):
            raise NotImplementedError
            #noiseless_posterior = super().posterior(
            #    X=X, observation_noise=False, **kwargs
            #)
            #noiseless_mvn = noiseless_posterior.mvn
            #neg_hessian_f = self.likelihood.neg_hessian_f(noiseless_mvn.mean)
            #try:
            #    likelihood_cov = neg_hessian_f.inverse()
            #except:
            #    eye_like_hessian = torch.eye(
            #        neg_hessian_f.shape[-2],
            #        device=neg_hessian_f.device,
            #        dtype=neg_hessian_f.dtype,
            #    )
            #    likelihood_cov = lazify(neg_hessian_f).inv_matmul(eye_like_hessian)

            #noisy_mvn = type(noiseless_mvn)(
            #    noiseless_mvn.mean, noiseless_mvn.lazy_covariance_matrix + likelihood_cov
            #)
            #return GPyTorchPosterior(mvn=noisy_mvn)

        return super().posterior(X=X, observation_noise=observation_noise, **kwargs)

    def fantasize(self,
            X,
            sampler,
            observation_noise: bool = True,
            **kwargs,
    ):
        assert observation_noise is True
        propagate_grads = kwargs.pop("propagate_grads", False)
        with settings.propagate_grads(propagate_grads):
            post_X = self.posterior(X, observation_noise=False)
        f_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
        Y_fantasized = self.likelihood(f_fantasized).sample()
        return self.condition_on_observations(X=X, Y=Y_fantasized, **kwargs)
