#!/usr/bin/env python3

"""
This module provides a utility function for obtaining MCMC samples from a Gaussian Process model using Pyro.

Functions:
    get_mcmc_samples(gp, num_samples=100, warmup_steps=100):
        Runs MCMC sampling on the given Gaussian Process model and returns the samples and diagnostics.

        Parameters:
            gp (gpytorch.models.ExactGP): The Gaussian Process model to sample from.
            num_samples (int, optional): The number of MCMC samples to generate. Default is 100.
            warmup_steps (int, optional): The number of warmup steps for the MCMC sampler. Default is 100.

        Returns:
            tuple: A tuple containing:
                - samp_temp (dict): A dictionary of MCMC samples.
                - diagnostics (dict): A dictionary of diagnostics information from the MCMC run.
"""

import gpytorch
import pyro
from copy import deepcopy
from pyro.infer.mcmc import NUTS, MCMC
from torch import hstack, all


def get_mcmc_samples(gp,num_samples=100, warmup_steps=100):
    train_inputs = gp.train_inputs[0]
    train_targets = gp.train_targets
    gppyro = deepcopy(gp)
    gppyro.task_covar_module.add_prior()
    gppyro.train()

    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled = gppyro.pyro_sample_from_prior()
            output = sampled.likelihood(sampled(x))
            pyro.sample("obs", output, obs=y.squeeze())
        return y

    nuts_kernel = NUTS(pyro_model, jit_compile=False, max_tree_depth=3, full_mass=True)
    mcmc_run = MCMC(
        nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=False
    )
    mcmc_run.run(train_inputs, train_targets)
    diagnostics = mcmc_run.diagnostics()
    samp_temp = mcmc_run.get_samples()
    inds = hstack([all((samp_temp['task_covar_module.covar_factor_prior'][i]@samp_temp['task_covar_module.covar_factor_prior'][i].T)[0]>=0) for i in range(num_samples)])
    samp_temp['task_covar_module.covar_factor_prior'] = samp_temp['task_covar_module.covar_factor_prior'][inds]
    # gppyro.pyro_load_from_samples(samp_temp)
    return samp_temp, diagnostics
