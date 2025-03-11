#!/usr/bin/env python3


"""
This script reproduces the data used for the illustrations in the paper. It applies the safe-UCB algorithm on a given function.

Usage:
    python3 -m run_SafeUCB

Arguments:
    function_name (str): The name of the function to optimize. Should be one of the following:
        - MTBranin_ST
        - LbSync_ST
        - MTPowell_ST

Main Script:
    - Sets default tensor type to float64.
    - Reads the function name from command line arguments.
    - Initializes parameters and directories.
    - Loads initial data points and thresholds.
    - For each initial point:
        - Configures the objective function and its parameters based on the function name.
        - Normalizes initial points and evaluates them for all tasks.
        - Samples supplementary tasks and concatenates data.
        - Standardizes training targets.
        - Initializes Bayesian Optimization.
        - Builds a Gaussian Process (GP) model.
        - Runs optimization loop for a specified number of runs.
    - Saves the final data to a file.
"""

import torch
from botorch.utils.transforms import normalize, unnormalize
import utils.utils
from utils.utils import sample_from_task,concat_data
from utils.mcmc_samples import get_mcmc_samples
import utils.get_robust_gp
from bo.bo_loop import BayesianOptimization
from numpy import load
from utils.optim import optimize_gp
from utils.functions import LbSync, MTBranin, MTPowell
import sys, os
import pickle
from utils.utils import standardize
from math import ceil

torch.set_default_dtype(torch.float64)

function_name = "MTBranin_ST" # Change function that should be optimized here

nruns = 100 # number of main task evaluations
delta_max = 0.05 # failure probability
tau = 0.001 # discretization parameter

if function_name == "MTBranin_ST":
    num_tsks = 2
    obj = MTBranin(num_tsks=num_tsks)
    d = obj.dim
    bounds = obj.bounds
if function_name =="LbSync_ST":
    num_tsks = 2
    num_lasers = 5
    KType = "PI"
    obj = LbSync(Ktyp=KType,num_lasers=num_lasers,num_tsks = num_tsks)
    d = obj.dim
    bounds = obj.bounds
if function_name == "MTPowell_ST":
    num_tsks = 2
    obj = MTPowell(num_tsks=num_tsks)
    d = obj.dim
    bounds = obj.bounds

folder = "Bayes_ST"
if not os.path.exists(f"data/{folder}"): os.mkdir(f"data/{folder}")

norm_bounds = torch.tensor([[0.0], [1.0]]).repeat(1,d)
tasks = list(range(num_tsks))

data = load(f"data/X_init_"+function_name.split("_")[0]+".npy", allow_pickle=True).item()
X_init = data["X_init"]
T = data["threshold"]
x0 = torch.tensor(X_init[0,...]).unsqueeze(0)
norm_bounds = torch.vstack((torch.zeros(1,d),torch.ones(1,d)))

num_sup_task_samples = 1
num_acq_samps = [1]
for _ in range(num_tsks-1): num_acq_samps.append(num_sup_task_samples)

# optimize for every initial point
data_sets=[]
bests = []
X_init = torch.tensor(X_init)
for i in range(X_init.size(0)):
    print("Round: ", i+1)
    x0 = X_init[i,...].view(1,bounds.size(-1))
    norm_bounds = torch.vstack((torch.zeros(1,d),torch.ones(1,d)))

    norm_x0 = normalize(x0,bounds)

    # evalaute initial point for all tasks
    train_targets = torch.zeros(num_tsks,1)
    for j in range(num_tsks):
        train_targets[j,...] = obj.f(norm_x0,j)
    train_tasks = torch.arange(num_tsks).unsqueeze(-1)
    norm_train_inputs = norm_x0.repeat(num_tsks,1)

    # evalaute supplementary tasks
    for k in range(1,num_tsks):
        x, t, y = sample_from_task(obj,[k],norm_bounds,n=2*num_sup_task_samples)
        norm_train_inputs, train_tasks, train_targets = concat_data((x, t, y),(norm_train_inputs,train_tasks,train_targets))
    
    norm_train_targets, mu, std = standardize(train_targets,train_task=train_tasks)
    T_stdizd, _, _ = standardize(T, mu, std)
    bo = BayesianOptimization(obj,tasks,norm_bounds,T_stdizd,(mu,std),num_acq_samps) 
    gp = utils.utils.build_mtgp((norm_train_inputs,train_tasks),norm_train_targets)
    for _ in range(nruns):
        gp,_,_ = optimize_gp(gp,mode=1,max_iter=200) # get MAP estimate
        mu = [gp.mean_module.base_means[r].constant.detach() for r in range(num_tsks)]
        robust_gp = utils.utils.build_mtgp((norm_train_inputs,train_tasks),norm_train_targets)
        robust_gp.task_covar_module._set_covar_factor(torch.eye(num_tsks)) # no correlation should be considered
        sqrtbeta = torch.sqrt(utils.get_robust_gp.beta_bayes(norm_bounds,tau,delta_max))
        bo.update_gp(robust_gp,sqrtbeta)
        norm_train_inputs,train_tasks,norm_train_targets = bo.step()
        T_stdizd = bo.threshold
        gp = utils.utils.build_mtgp((norm_train_inputs,train_tasks),norm_train_targets)
    # add data
    train_inputs = unnormalize(norm_train_inputs,bounds)
    train_targets = bo.unstd_train_targets
    data_sets.append([train_inputs,train_tasks,train_targets])
    bests.append([bo.best_x,bo.best_y])
    print(f"Best value: {round(bo.best_y[-1],3)} at input: {unnormalize(bo.best_x[-1],bounds).round(decimals=3)}")

# save data
file = open(f"data/{folder}/{function_name}.obj",'wb')
sets = {'data_sets': data_sets, 'bests': bests}
pickle.dump(sets,file)
file.close()


