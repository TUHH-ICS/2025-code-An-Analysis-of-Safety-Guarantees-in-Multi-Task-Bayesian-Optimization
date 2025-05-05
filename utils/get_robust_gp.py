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
            delta_max (float, optional): Failure probability for the discretized set. Default is 0.05.
            tau (float, optional): Step size for discretizing the input space. Default is 0.01.
            rho_max (float, optional): Failure probability for the uncertainty set C_rho. Default is None.
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
    
    Further functions:
    get_barbeta(model0, sampmods, maxsqrtbeta, rho_max: float):
        Computes the lambda and gamma values for the given model and sample models.
    get_gamma(sampmods):
        Computes the scaling factor induced by the standard deviation.
    get_nu(sampmods):
        Computes the scaling factor induced by the mean.
    post_mean_SE(sampmods):
        Computes the squared error of the posterior means for the different covariance matrices.
    plot_sampmods_det(sampmods):
        Plots the histogram of the determinants of the covariance matrices from the sample models.
"""



import torch
from copy import deepcopy
from torch import Tensor
import torch
from math import floor
from matplotlib import pyplot as plt


def bayesian_robust_gp(
    sampmods,
    model0,
    bounds,
    delta_max: float = 0.05,
    tau: float = 0.01,
    rho_max = None,
):
    if rho_max is None:
        rho_max = delta_max
    robustmodel = deepcopy(model0)

    sqrtbeta = beta_bayes(bounds, tau, delta_max).sqrt()
    gamma, nu, sigprime = get_barbeta(
        model0, sampmods, sqrtbeta, rho_max
    )
    print(f"nu:{nu}")
    print(f"gamma: {gamma}")
    sqrtbetabar = gamma * sqrtbeta + nu
    print(f"sqrtbetabar: {sqrtbetabar}")
    sigmaprime = sigprime@sigprime.T
    print(f"Correlation Matrix: {sigmaprime}")
    if sqrtbeta <= sqrtbetabar*sigmaprime.det():
        robustmodel.task_covar_module._set_covar_factor(torch.eye(sigmaprime.size(0)))
        print("Using identity covariance matrix")
        return robustmodel, sqrtbeta
    else:
        robustmodel.task_covar_module._set_covar_factor(sigprime)
        print("Using correlation matrix")
        return robustmodel, sqrtbetabar


def beta_bayes(
    bounds: Tensor = torch.arange(2).unsqueeze(-1),
    tau: float = 0.01,
    delta_max: float = 0.05,
):
    M = torch.hstack([torch.ceil((bu - bl) / (2*tau)) + 1 for bl, bu in bounds.T])
    m = M.prod()
    beta = 2 * torch.log(m / delta_max)
    return beta


def get_barbeta(model0, sampmods, maxsqrtbeta, rho_max: float):
    noise = model0.likelihood.noise.detach()
    sigmaf = model0.covar_module.outputscale.detach()
    covar = sampmods.task_covar_module._eval_covar_matrix()
    indmax = floor(covar.size(0) * (1 - rho_max))
    chol_covar = sampmods.task_covar_module.covar_factor.detach()
    dets = (torch.linalg.det(sigmaf*covar)+sigmaf*noise)/(sigmaf+noise)
    plot_sampmods_det(sampmods)
    with torch.no_grad():
        gamma = get_gamma(sampmods)
        # nu2 = get_nu_comp(sampmods)
        nu = get_nu(sampmods)
    total = maxsqrtbeta*gamma + nu
    total *= dets.sqrt().view(-1,1)
    total,inds = total.sort(dim=-1)
    Id = total[:,indmax].argmin()
    sigprime = chol_covar[Id]
    gamma = gamma[Id,inds[Id,indmax]].detach()
    nu = nu[Id,inds[Id,indmax]].detach()
    return gamma, nu, sigprime


def get_gamma(sampmods):
    covar = sampmods.task_covar_module._eval_covar_matrix()
    sigmaf = sampmods.covar_module.outputscale.detach()
    noise = sampmods.likelihood.noise.detach()
    gamma = torch.linalg.eigvals(torch.linalg.solve((covar+noise/sigmaf*torch.eye(covar.size(-1)).unsqueeze(0)).unsqueeze(1),(covar+noise/sigmaf*torch.eye(covar.size(-1)).unsqueeze(0)).unsqueeze(0))).real.max(dim=-1)[0]
    return gamma.sqrt()


def get_covar_factors(sampmods):
    covar = sampmods.task_covar_module._eval_covar_matrix().detach()
    return torch.linalg.cholesky(covar, upper=False).squeeze()


def get_nu_comp(sampmods):
    sampmods.train()
    covar_factors = get_covar_factors(sampmods)
    L_norm = torch.linalg.norm(torch.linalg.solve(covar_factors.unsqueeze(1), covar_factors.unsqueeze(0)),dim=(-1,-2),ord=2)**2
    train_inputs = sampmods.train_inputs[0]
    train_targets = sampmods.train_targets
    noise = sampmods.likelihood.noise.detach()
    K = sampmods(train_inputs).covariance_matrix
    Kd = K + noise * torch.eye(K.size(-1)).unsqueeze(0)
    alpha = torch.linalg.solve(Kd, train_targets.unsqueeze(-1))
    term1 = alpha.transpose(-1,-2).matmul(K).matmul(alpha).squeeze()
    term2 = alpha.unsqueeze(1).transpose(-1,-2).matmul(K).matmul(alpha).squeeze()
    norm_diff = term1.unsqueeze(1) - 2*term2 + L_norm.mul(term1.unsqueeze(0))
    mean_se = post_mean_SE(sampmods)
    return norm_diff.add(mean_se).maximum(torch.tensor(1e-12)).sqrt()


def get_nu(sampmods):
    sampmods.train()
    covar_factors = sampmods.task_covar_module._eval_covar_matrix().detach()
    L = covar_factors.unsqueeze(0).matmul(torch.linalg.solve(covar_factors.unsqueeze(1), covar_factors.unsqueeze(0)))
    train_inputs = sampmods.train_inputs[0]
    train_targets = sampmods.train_targets
    train_tasks = train_inputs[0,:,-1].to(dtype=torch.int32)
    noise = sampmods.likelihood.noise.detach()
    K = sampmods(train_inputs).covariance_matrix
    Kd = K + noise * torch.eye(K.size(-1)).unsqueeze(0)
    alpha = torch.linalg.solve(Kd, train_targets.unsqueeze(-1))
    k1 = sampmods.covar_module(train_inputs[:,:,:-1]).evaluate()
    task_covs = L[...,train_tasks,train_tasks.unsqueeze(-1)]
    K2 = k1.mul(task_covs)
    term1 = alpha.transpose(-1,-2).matmul(K).matmul(alpha)
    term2 = alpha.unsqueeze(1).transpose(-1,-2).matmul(K).matmul(alpha)
    term3 = alpha.transpose(-1,-2).matmul(K2).matmul(alpha)
    norm_diff = (term1.unsqueeze(1) -2*term2 + term3).squeeze()
    mean_se = post_mean_SE(sampmods)
    return norm_diff.add(mean_se).maximum(torch.tensor(1e-12)).sqrt()


def post_mean_SE(sampmods):
    noise =sampmods.likelihood.noise.detach()
    (train_inputs,) = sampmods.train_inputs
    sampmods.eval()
    mu1 = sampmods(train_inputs).mean
    mu0 = mu1
    normdiff = torch.norm(mu0.unsqueeze(1) - mu1.unsqueeze(0), 2,dim=-1) ** 2 / noise
    return normdiff  


def plot_sampmods_det(sampmods):
    covar = sampmods.task_covar_module._eval_covar_matrix()
    dets = torch.linalg.det(covar)
    _, ax = plt.subplots()
    ax.hist(dets.detach().numpy())
    ax.set_title("Determinants of Covariance Matrices")
    ax.set_xlabel("Determinant")
    ax.set_ylabel("Frequency")
    plt.show()