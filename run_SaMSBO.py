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
        - Builds multi-task Gaussian Process (MTGP) model.
        - Runs optimization loop for a specified number of runs.
    - Saves the final data to a file.
"""


import torch
from botorch.utils.transforms import normalize, unnormalize
import utils.utils
from utils.utils import sample_from_task, concat_data
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

function_name = sys.argv[1]
nruns = 75
delta_max = 0.05
rho = 0.15
dist = 0.3

folder = "Data" + str(int(10 * dist)) + "_2"
if not os.path.exists(f"data/{folder}"):
    os.mkdir(f"data/{folder}")

def get_samples(gp, min_samples = 50, num_samples=100, warmup_steps=100):
    counter = 0
    samples, diagnostics = get_mcmc_samples(
        gp=gp, num_samples=num_samples, warmup_steps=warmup_steps
    )
    while samples["task_covar_module.covar_factor_prior"].shape[0] <= min_samples:
        if counter > 10:
            samples["task_covar_module.covar_factor_prior"] = (
                torch.eye(num_tsks).unsqueeze(0).repeat(2, 1, 1)
            )
            break
        samples0, diagnostics = get_mcmc_samples(
            gp=gp, num_samples=100, warmup_steps=100
        )
        samples = {
            key: torch.vstack((samples[key], samples0[key]))
            for key in samples.keys()
        }
        counter += 1
    return samples, diagnostics

data = load(
    f"data/X_init_" + function_name.split("_")[0] + ".npy", allow_pickle=True
).item()
X_init = data["X_init"]
T = data["threshold"]

# optimize for every initial point
data_sets = []
bests = []
covars = []
betas = []
X_init = torch.tensor(X_init)
for i in range(X_init.size(0)):
    if function_name == "MTBranin":
        tau = 0.001
        num_tsks = 2
        obj = MTBranin(num_tsks=num_tsks, disturbance=dist)
        d = obj.dim
        bounds = obj.bounds
    if function_name == "LbSync":
        tau = 0.001
        num_tsks = 2
        num_lasers = 5
        KType = "PI"
        obj = LbSync(
            Ktyp=KType, num_lasers=num_lasers, num_tsks=num_tsks, disturbance=dist
        )
        d = obj.dim
        bounds = obj.bounds
    if function_name == "MTPowell":
        tau = 0.001
        num_tsks = 2
        obj = MTPowell(dim=4, num_tsks=num_tsks, disturbance=dist)
        d = obj.dim
        bounds = obj.bounds

    print("Round: ", i + 1)
    x0 = X_init[i, ...].view(1, bounds.size(-1))
    norm_bounds = torch.vstack((torch.zeros(1, d), torch.ones(1, d)))
    tasks = list(range(num_tsks))
    num_sup_task_samples = ceil(2 * d / (num_tsks - 1))
    num_acq_samps = [1]
    for _ in range(num_tsks - 1):
        num_acq_samps.append(num_sup_task_samples)

    norm_x0 = normalize(x0, bounds)

    # evalaute initial point for all tasks
    train_targets = torch.zeros(num_tsks, 1)
    for j in range(num_tsks):
        train_targets[j, ...] = obj.f(norm_x0, j)
    train_tasks = torch.arange(num_tsks).unsqueeze(-1)
    norm_train_inputs = norm_x0.repeat(num_tsks, 1)

    # evalaute supplementary tasks
    for k in range(1, num_tsks):
        x, t, y = sample_from_task(
            obj,
            [k],
            norm_bounds + torch.tensor([[obj.max_disturbance], [-obj.max_disturbance]]),
            n=2 * num_sup_task_samples,
        )
        norm_train_inputs, train_tasks, train_targets = concat_data(
            (x, t, y), (norm_train_inputs, train_tasks, train_targets)
        )

    norm_train_targets, mu, std = standardize(train_targets, train_task=train_tasks)
    T_stdizd, _, _ = standardize(T, mu, std)
    bo = BayesianOptimization(
        obj, tasks, norm_bounds, T_stdizd, (mu, std), num_acq_samps
    )
    gp = utils.utils.build_mtgp(
        (norm_train_inputs, train_tasks),
        norm_train_targets)
    covar = torch.zeros(nruns, num_tsks, num_tsks)
    beta_ = torch.zeros(nruns)
    mod_runs = 5
    for run in range(nruns):
        if run >= 45:
            bo.num_acq_samps = [1] * num_tsks
            mod_runs = 15
        if run <= 10 or run % mod_runs == 0:
            repeat = True
            while repeat:
                gp,_,_ = optimize_gp(
                    gp, mode=1, max_iter=200)  # get MAP estimate
                mu = [
                    gp.mean_module.base_means[r].constant.detach()
                    for r in range(num_tsks)
                ]
                noise = gp.likelihood.noise.detach()
                varf = gp.covar_module.outputscale.detach()

                samples,_ = get_samples(gp, min_samples=50, num_samples=100, warmup_steps=100)

                sample_models = utils.utils.build_mtgp(
                    (norm_train_inputs, train_tasks),
                    norm_train_targets
                )
                sample_models.likelihood.noise = noise
                sample_models.task_covar_module.add_prior()
                sample_models.pyro_load_from_samples(samples)

                gp = utils.utils.build_mtgp(
                    (norm_train_inputs, train_tasks),
                    norm_train_targets,
                    mu=mu,
                )
                # gp.likelihood.noise = noise
                robust_gp, sqrtbeta = utils.get_robust_gp.bayesian_robust_gp(
                    sample_models, gp, norm_bounds, delta_max=delta_max, tau=tau, rho_max=rho
                )
                noise = robust_gp.likelihood.noise.detach()
                del sample_models
                repeat = (
                    torch.sqrt(varf - varf**2 / (varf + noise)) * sqrtbeta
                ) >= abs(T_stdizd)
        else:
            gp,_,_ = optimize_gp(
                gp, mode=1, max_iter=200)  # get MAP estimate
            mu = [gp.mean_module.base_means[r].constant.detach() for r in range(num_tsks)]
            covar_chol = robust_gp.task_covar_module.covar_factor
            robust_gp = utils.utils.build_mtgp(
                (norm_train_inputs, train_tasks), norm_train_targets, mu=mu)
            robust_gp.task_covar_module._set_covar_factor(covar_chol)
        covar[run, ...] = robust_gp.task_covar_module._eval_covar_matrix()
        beta_[run] = sqrtbeta
        bo.update_gp(robust_gp, sqrtbeta)
        norm_train_inputs, train_tasks, norm_train_targets = bo.step()
        T_stdizd = bo.threshold
        gp = utils.utils.build_mtgp(
            (norm_train_inputs, train_tasks), norm_train_targets, mu=mu)
    # add data
    train_inputs = unnormalize(norm_train_inputs, bounds)
    train_targets = bo.unstd_train_targets
    data_sets.append([train_inputs, train_tasks, train_targets])
    bests.append([bo.best_x, bo.best_y])
    covars.append(covar)
    betas.append(beta_)
    print(
        f"Best value: {round(bo.best_y[-1],3)} at input: {unnormalize(bo.best_x[-1],bounds).round(decimals=3)}"
    )

# save data
file = open(f"data/{folder}/{function_name}.obj", "wb")
sets = {"data_sets": data_sets, "bests": bests, "covar": covars, "betas": betas}
pickle.dump(sets, file)
file.close()
