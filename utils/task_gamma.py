#!/usr/bin/env python3

"""
This module provides utility functions for working with task covariance matrices in the context of 
Bayesian optimization using Gaussian Processes.
Functions:
    get_barbeta(model0, sampmods, maxsqrtbeta, delta_max: float):
        Computes the lambda and gamma values for the given model and sample models.
    get_gamma(candidate_dict, sampmods):
        Computes the gamma values for the given candidates.
    get_covar_candidates(sampmods):
        Retrieves the covariance matrix candidates from the sample models. One can set an optional threshold for the number of candidates.
    get_lambda(model0, candidate_dict, sampmods):
        Computes the lambda values for the given candidates.
    post_mean_MSE(candidate_dict, sampmods):
        Computes the mean squared error of the posterior means for the given candidates
    plot_sampmods_det(sampmods):
        Plots the histogram of the determinants of the covariance matrices from the sample models.
"""


import torch
from math import floor
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot as plt


def get_barbeta(model0, sampmods, maxsqrtbeta, delta_max: float):
    noise = model0.likelihood.noise.detach()
    sigmaf = model0.covar_module.outputscale.detach()
    covar = sampmods.task_covar_module._eval_covar_matrix()
    chol_covar_all = sampmods.task_covar_module.covar_factor.detach()
    indmax = floor(covar.size(0) * (1 - delta_max))
    candidate_dict = get_covar_candidates(sampmods)
    chol_covar = sampmods.task_covar_module.covar_factor.detach()[candidate_dict["inds"]]
    dets = (torch.linalg.det(sigmaf*candidate_dict['covar'])+sigmaf*noise)/(sigmaf+noise)
    with torch.no_grad():
        gamma = get_gamma(candidate_dict,sampmods).sqrt()
        lambda_ = get_lambda(model0,candidate_dict, sampmods).sqrt()
    total = maxsqrtbeta*gamma + lambda_
    total2 = total.clone() 
    total *= dets.sqrt().view(-1,1)
    total,inds = total.sort(dim=-1)
    Id = total[:,indmax].argmin()
    thprime = chol_covar[Id]
    gamma = gamma[Id,inds[Id,indmax]].detach()
    lambda_ = lambda_[Id,inds[Id,indmax]].detach()
    covar_set = chol_covar_all[inds[Id,:indmax].squeeze()]
    return gamma, lambda_, thprime, total[Id,indmax], covar_set, total2[torch.arange(0,total2.size(0)),inds[:,indmax].squeeze()]

def get_gamma(candidate_dict,sampmods):
    covar = sampmods.task_covar_module._eval_covar_matrix()
    sigmaf = sampmods.covar_module.outputscale.detach()
    noise = sampmods.likelihood.noise.detach()
    gamma = torch.linalg.eigvals(torch.linalg.solve((candidate_dict["covar"]+noise/sigmaf*torch.eye(covar.size(-1)).unsqueeze(0)).unsqueeze(1),(covar+noise/sigmaf*torch.eye(covar.size(-1)).unsqueeze(0)).unsqueeze(0))).real.max(dim=-1)[0]
    return gamma

def get_covar_candidates(sampmods):
    covar = sampmods.task_covar_module._eval_covar_matrix()
    num_samps = covar.size(0)
    _,inds = torch.linalg.det(covar).sort(descending=False)
    candiate_dict = {"covar":covar[inds[:round(1.*num_samps)]], "inds":inds[:round(1.*num_samps)]}
    return candiate_dict


def get_lambda(model0, candidate_dict, sampmods):
    (train_inputs,) = model0.train_inputs 
    train_tasks = train_inputs[:,-1].unsqueeze(-1).to(dtype=torch.int32)
    train_targets = deepcopy(model0.train_targets).unsqueeze(-1)
    covarMats = sampmods.task_covar_module._eval_covar_matrix()
    num_samps = covarMats.size(0)
    num_cands = candidate_dict["inds"].size(0)
    inds = candidate_dict["inds"]
    num_tsk = covarMats.size(-1)
    candidate_chol = sampmods.task_covar_module.covar_factor.detach()[candidate_dict["inds"]]
    mu = [model0.mean_module.base_means[i].constant.detach() for i in range(num_tsk)]   
    train_means = torch.empty_like(train_targets)
    for i in range(num_tsk):
        train_means[train_tasks==i] = mu[i]
    train_targets -= train_means

    sampmods1 = deepcopy(sampmods)
    covar_chol_2 = torch.linalg.solve((candidate_chol+1e-2*torch.eye(num_tsk)).unsqueeze(1),covarMats.unsqueeze(0)).permute(0,1,-1,-2)
    K1 = sampmods.forward(train_inputs.repeat(num_samps,1,1)).covariance_matrix
    K1td = sampmods.likelihood(sampmods.forward(train_inputs.repeat(num_samps,1,1))).covariance_matrix
    K1tdInv = torch.linalg.solve(K1td,torch.eye(K1td.size(-1)).unsqueeze(0))
    alpha_2 = K1tdInv@train_targets
    norms = torch.empty(num_cands,num_samps)
    for i in tqdm(range(num_cands)):
        alpha_1 = alpha_2[inds[i]].view(1,-1,1)
        sampmods1.task_covar_module._set_covar_factor(covar_chol_2[i,:,:,:])
        K2 = sampmods1.forward(train_inputs.repeat(num_samps,1,1)).covariance_matrix
        norms[i,:] = (alpha_2.permute(0,-1,-2)@K2@alpha_2+alpha_1.permute(0,-1,-2)@K1[inds[i]]@alpha_1-2*alpha_1.permute(0,-1,-2)@K1@alpha_2).squeeze()
    norms += post_mean_MSE(candidate_dict,sampmods)
    norms = norms.maximum(torch.tensor(1e-8))
    return norms
    

def post_mean_MSE(candidate_dict, sampmods):
    inds = candidate_dict["inds"]
    noise =sampmods.likelihood.noise.detach()
    # outputscale = sampmods.covar_module.outputscale.detach()
    (train_inputs,) = sampmods.train_inputs
    sampmods.eval(), sampmods.eval()
    mu1 = sampmods(train_inputs).mean
    mu0 = mu1[inds]
    normdiff = torch.norm(mu0.unsqueeze(1) - mu1.unsqueeze(0), 2,dim=-1) ** 2 / noise
    print(f"Norm difference of posterior means: {normdiff.max()}")
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