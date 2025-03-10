#!/usr/bin/env python3

"""
This module provides functions for robust Gaussian Process (GP) modeling using Bayesian and frequentist approaches.

Functions:
    bayesian_robust_gp(sampmods, model0, bounds, delta_max=0.05, tau=0.01, rho_max=None, conservatism=False):
        Constructs a robust GP model using Bayesian methods.
        
        Parameters:
            sampmods (list): List of sample models.
            model0 (GPModel): Initial GP model.
            bounds (Tensor): Tensor specifying the bounds for the input space.
            delta_max (float, optional): Maximum allowable deviation. Default is 0.05.
            tau (float, optional): Step size for discretizing the input space. Default is 0.01.
            rho_max (float, optional): Maximum allowable rho value. Default is None.
            conservatism (bool, optional): Flag to indicate if conservatism should be applied. Default is False.
        
        Returns:
            robustmodel (GPModel): The robust GP model.
            sqrtbeta (Tensor): The square root of the beta parameter.
            covar_set (optional): Covariance set if conservatism is True.
            totals (optional): Totals if conservatism is True.

    beta_bayes(bounds=torch.arange(2).unsqueeze(-1), tau=0.01, delta_max=0.05):
        Computes the beta parameter for Bayesian robust GP.
        
        Parameters:
            bounds (Tensor, optional): Tensor specifying the bounds for the input space. Default is torch.arange(2).unsqueeze(-1).
            tau (float, optional): Step size for discretizing the input space. Default is 0.01.
            delta_max (float, optional): Maximum allowable deviation. Default is 0.05.
        
        Returns:
            beta (Tensor): The beta parameter.

    beta_freq(gp, B, delta_max=0.05, uncertainty=0.1):
        Computes the beta parameter for frequentist robust GP.
        
        Parameters:
            gp (GPModel): The GP model.
            B (float): A parameter for the frequentist approach.
            delta_max (float, optional): Maximum allowable deviation. Default is 0.05.
            uncertainty (float, optional): Uncertainty parameter. Default is 0.1.
        
        Returns:
            sqrt_beta (Tensor): The square root of the beta parameter.

    lambda_(gp, uncertainty):
        Computes the lambda parameter for frequentist robust GP.
        
        Parameters:
            gp (GPModel): The GP model.
            uncertainty (float): Uncertainty parameter.
        
        Returns:
            lambda (Tensor): The lambda parameter.
"""



import torch
from utils.task_gamma import get_barbeta
from copy import deepcopy
from math import sqrt, log
from torch import Tensor


def bayesian_robust_gp(
    sampmods,
    model0,
    bounds,
    delta_max: float = 0.05,
    tau: float = 0.01,
    rho_max = None,
    conservatism: bool = False,
):
    if rho_max is None:
        rho_max = delta_max
    maxsqrtbeta = beta_bayes(bounds, tau, rho_max).sqrt()
    gamma, lambda_, sigprime, total, covar_set, totals = get_barbeta(
        model0, sampmods, maxsqrtbeta, delta_max
    )
    sigmaf = model0.covar_module.outputscale.detach()
    print(f"Correlation Matrix: {sigprime@sigprime.T}")
    print(f"Uncertainty at supplementary task: {total*sigmaf}")
    model0.task_covar_module._set_covar_factor(sigprime)
    robustmodel = deepcopy(model0)
    print(f"lambda:{lambda_}")
    print(f"gamma: {gamma}")
    sqrtbeta = gamma * maxsqrtbeta + lambda_
    print(f"sqrtbeta: {sqrtbeta}")
    if conservatism:
        return robustmodel, sqrtbeta, covar_set, totals
    else:
        return robustmodel, sqrtbeta


def beta_bayes(
    bounds: Tensor = torch.arange(2).unsqueeze(-1),
    tau: float = 0.01,
    delta_max: float = 0.05,
):
    M = torch.hstack([torch.ceil((bu - bl) / tau)-1 for bl, bu in bounds.T])
    m = M.prod()
    beta = 2 * torch.log(m / delta_max)
    return beta


def beta_freq(gp, B, delta_max: float = 0.05, uncertainty: float = 0.1):
    n = gp.train_targets.size(0)
    fact1 = sqrt(n + 2 * n * sqrt(log(1 / delta_max)) + 2 * log(1 / delta_max))
    print(f"fact1: {fact1}")
    sqrt_beta = lambda_(gp, uncertainty) * B + fact1
    return sqrt_beta


def lambda_(gp, uncertainty) -> Tensor:
    covar = gp.task_covar_module._eval_covar_matrix()
    n = covar.size(-1)
    matofdiag = torch.ones(n, n) - torch.eye(n)
    covar_u = torch.minimum(
        covar + uncertainty * matofdiag, torch.ones_like(covar) - 1e-1 * matofdiag
    )
    covar_l = torch.maximum(covar - uncertainty * matofdiag, torch.zeros_like(covar))
    eig_u = torch.linalg.eigvals(torch.linalg.solve(covar_u, covar)).real.max()
    eig_l = torch.linalg.eigvals(torch.linalg.solve(covar_l, covar)).real.max()
    return torch.maximum(eig_u, eig_l).sqrt()

