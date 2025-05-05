#!/usr/bin/env python3

"""
This script reproduces the data used for the illustrations in the paper. It applies the SaMSBO algorithm on a given function.

Usage:
    python3 -m run_bayes <function_name>

Arguments:
    function_name (str): The name of the function to optimize. Should be one of the following:
        - MTBranin
        - LbSync
        - MTPowell

Functions:
    get_samples(gp, min_samples=50, num_samples=100, warmup_steps=100):
        Generates MCMC samples of the hyper-posterior.

Main Script:
    - Initializes parameters and directories.
    - Loads initial data points and thresholds.
    - For each initial point:
        - Configures the objective function and its parameters based on the function name.
        - Normalizes initial points and evaluates them for all tasks.
        - Samples supplementary tasks and concatenates data.
        - Standardizes training targets.
        - Initializes Bayesian Optimization.
        - Builds multi-task Gaussian Process (MTGP) model.
        - Runs optimization loop for a specified number of runs.
    - Saves the final data to a file.
"""


import os
import sys
import pickle
import random
import torch
import numpy.random
from math import ceil
from numpy import load
from botorch.utils.transforms import normalize, unnormalize

from utils.utils import (
    sample_from_task,
    concat_data,
    standardize,
    build_mtgp,
)
from utils.mcmc_samples import get_samples
from utils.get_robust_gp import bayesian_robust_gp
from utils.optim import optimize_gp
from utils.functions import LbSync, MTBranin, MTPowell
from bo.bo_loop import MultiTaskBayesianOptimization

torch.set_default_dtype(torch.float64)

# Constants
NRUNS = 40
DELTA_MAX = 0.05
RHO_MAX = 0.15
TAU = 0.001

seeds = {
    "MTBranin": [4001630359, 1919308292, 1427243158, 2443879631, 2368281548, 1373693466,
                1878128376, 2416680340, 2577055479, 473042899, 2789764286, 2245849798,
                3162095579, 3697226237, 3466549482],
    "LbSync": [ 378460615, 2204309691, 3714415149, 3247551807, 1394259410, 2593874679,
    1338135722,  229394925, 1148606685,  738629800, 3400498408, 3913446390,
    2823833501, 2483689362, 1656936394],
    "MTPowell": [2788530576,  588811540, 1073174780, 1979726901,  237964780, 1720245674,
    371781323,  946294099,   69072920, 1340726846, 1378218114, 1976946218,
        453164431,  977118581,  825415680],
}

def initialize_experiment(function_name, dist):
    """Initialize experiment parameters and directories."""
    folder = f"Final_runs_{int(100 * dist):03d}"
    os.makedirs(f"data/{folder}", exist_ok=True)

    data = load(f"data/X_init_{function_name.split('_')[0]}.npy", allow_pickle=True).item()
    return folder, torch.tensor(data["X_init"]), data["threshold"]

def configure_objective(function_name, dist, seed):
    """Configure the objective function and its parameters."""
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    if function_name == "MTBranin":
        return MTBranin(num_tsks=2, disturbance=dist)
    elif function_name == "LbSync":
        return LbSync(Ktyp="PI", num_lasers=5, num_tsks=2, disturbance=dist)
    elif function_name == "MTPowell":
        return MTPowell(dim=4, num_tsks=2, disturbance=dist)
    else:
        raise ValueError(f"Unknown function name: {function_name}")

def evaluate_initial_points(obj, x0, bounds):
    """Evaluate initial points for all tasks."""
    norm_x0 = normalize(x0, bounds)
    train_targets = torch.zeros(obj.num_tsks, 1)
    for j in range(obj.num_tsks):
        train_targets[j, ...] = obj.f(norm_x0, j)
    return norm_x0, train_targets

def evaluate_supplementary_tasks(obj, norm_bounds, norm_train_inputs, train_tasks, train_targets):
    """Evaluate supplementary tasks and concatenate data."""
    for k in range(1, obj.num_tsks):
        x, t, y = sample_from_task(obj, [k], norm_bounds, n=2 * ceil(2 * obj.dim / (obj.num_tsks - 1)))
        norm_train_inputs, train_tasks, train_targets = concat_data((x, t, y), (norm_train_inputs, train_tasks, train_targets))
    return norm_train_inputs, train_tasks, train_targets

def optimize_and_update_gp(gp, num_tsks, norm_bounds):
    """Optimize and update the Gaussian Process model."""
    gp, _, _ = optimize_gp(gp, mode=1, max_iter=200)
    mu = [gp.mean_module.base_means[r].constant.detach() for r in range(num_tsks)]
    samples, _ = get_samples(gp, min_samples=50, num_samples=100, warmup_steps=100)
    norm_train_inputs, train_tasks = gp.train_inputs[0][:,:-1], gp.train_inputs[0][:, -1:].to(dtype=torch.int32)
    norm_train_targets = gp.train_targets.unsqueeze(-1)
    sample_models = build_mtgp((norm_train_inputs, train_tasks), norm_train_targets)
    sample_models.task_covar_module.add_prior()
    sample_models.pyro_load_from_samples(samples)
    gp = build_mtgp((norm_train_inputs, train_tasks), norm_train_targets, mu=mu)
    robust_gp, sqrtbeta = bayesian_robust_gp(sample_models, gp, norm_bounds, delta_max=DELTA_MAX, tau=TAU, rho_max=RHO_MAX)
    return robust_gp, sqrtbeta

def main():
    function_name = sys.argv[1]
    dist = float(sys.argv[2])

    print(f"rhomax: {RHO_MAX}, dist: {dist}")
    folder, X_init, T = initialize_experiment(function_name, dist)

    data_sets, bests, covars, betas = [], [], [], []

    for i in range(X_init.size(0)):
        obj = configure_objective(function_name, dist, seeds[function_name][i])
        bounds, num_tsks = obj.bounds, obj.num_tsks

        print(f"Round: {i + 1}")
        x0 = X_init[i, ...].view(1, bounds.size(-1))
        norm_bounds = torch.vstack((torch.zeros(1, obj.dim), torch.ones(1, obj.dim)))

        norm_x0, train_targets = evaluate_initial_points(obj, x0, bounds)
        train_tasks = torch.arange(num_tsks).unsqueeze(-1)
        norm_train_inputs = norm_x0.repeat(num_tsks, 1)

        norm_train_inputs, train_tasks, train_targets = evaluate_supplementary_tasks(
            obj, norm_bounds, norm_train_inputs, train_tasks, train_targets
        )

        norm_train_targets = standardize(train_targets, T=T)
        T_stdizd = standardize(T, T)
        num_acq_samps = [1] + [ceil(2 * obj.dim / (num_tsks - 1))] * (num_tsks - 1)
        bo = MultiTaskBayesianOptimization(obj, list(range(num_tsks)), norm_bounds, T_stdizd, T, num_acq_samps)
        covar = torch.zeros(NRUNS, num_tsks, num_tsks)
        beta_ = torch.zeros(NRUNS)
        mod_runs = 4
        sqrtbeta = 1.

        for run in range(NRUNS):
            gp = build_mtgp((norm_train_inputs, train_tasks), norm_train_targets)

            if run >= 45:
                bo.num_acq_samps = [1] * num_tsks
                mod_runs = 15
            if run <= 10 or run % mod_runs == 0:
                robust_gp, sqrtbeta = optimize_and_update_gp(
                    gp, num_tsks, norm_bounds
                )
            else:
                gp, _, _ = optimize_gp(gp, mode=1, max_iter=200)
                mu = [gp.mean_module.base_means[r].constant.detach() for r in range(num_tsks)]
                covar_chol = robust_gp.task_covar_module.covar_factor
                robust_gp = build_mtgp((norm_train_inputs, train_tasks), norm_train_targets, mu=mu)
                robust_gp.task_covar_module._set_covar_factor(covar_chol)
            covar[run, ...] = robust_gp.task_covar_module._eval_covar_matrix()
            beta_[run] = sqrtbeta
            bo.update_gp(robust_gp, sqrtbeta)
            norm_train_inputs, train_tasks, norm_train_targets = bo.step()

        train_inputs = unnormalize(norm_train_inputs, bounds)
        train_targets = bo.unstd_train_targets
        data_sets.append([train_inputs, train_tasks, train_targets])
        bests.append([bo.best_x, bo.best_y])
        covars.append(covar)
        betas.append(beta_)

        print(f"Best value: {round(bo.best_y[-1], 3)} at input: {unnormalize(bo.best_x[-1], bounds).round(decimals=3)}")

    with open(f"data/{folder}/{function_name}_rho_{int(100 * RHO_MAX)}_prior_005_12.obj", "wb") as file:
        pickle.dump({"data_sets": data_sets, "bests": bests, "covar": covars, "betas": betas, "T": T}, file)

if __name__ == "__main__":
    main()
