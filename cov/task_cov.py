#!/usr/bin/env python3

"""
This module defines the `IndexKernelAllPriors` class, which extends the `IndexKernel` class from GPyTorch (https://docs.gpytorch.ai/en/stable/kernels.html#specialty-kernels) to include additional priors and constraints.

Classes:
    IndexKernelAllPriors: A kernel that allows for the inclusion of priors and constraints on the covariance factor and variance.

IndexKernelAllPriors:
    Methods:
        __init__(self, num_tasks: int, rank: int, covar_factor_prior: Prior | None = None, covar_factor_constraint: None = None, var_prior: Prior | None = None, var_constraint: Interval | None = None, **kwargs):
            Initializes the IndexKernelAllPriors with the specified number of tasks, rank, and optional priors and constraints.

        covar_factor(self):
            Property that returns the covariance factor, applying any constraints if present.

        covar_factor(self, value):
            Setter for the covariance factor, applying any constraints if present.

        _set_covar_factor(self, value):
            Helper method to set the covariance factor, converting the value to a tensor if necessary.

        _covar_factor_param(self, m: Kernel) -> Tensor:
            Returns the covariance factor parameter for the given kernel.

        _covar_factor_closure(self, m: Kernel, v: Tensor) -> Tensor:
            Closure method to set the covariance factor for the given kernel.

        _var_param(self, m: Kernel) -> Tensor:
            Returns the variance parameter for the given kernel.

        _var_closure(self, m: Kernel, v: Tensor) -> Tensor:
            Closure method to set the variance for the given kernel.

        _eval_covar_matrix(self):
            Evaluates and returns the covariance matrix based on the covariance factor.

        add_prior(self):
            Registers the priors for the covariance factor and variance.
"""


import torch
from torch import Tensor
from gpytorch.constraints import Interval
from gpytorch.priors import Prior
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels import Kernel


class IndexKernelAllPriors(IndexKernel):
    def __init__(
        self,
        num_tasks: int,
        rank: int,
        covar_factor_prior: Prior | None = None,
        covar_factor_constraint: None = None,
        var_prior: Prior | None = None,
        var_constraint: Interval | None = None,
        **kwargs
    ):
        super(IndexKernelAllPriors, self).__init__(
            num_tasks, rank, None, var_constraint, **kwargs
        )
        self.rank = rank
        self.register_parameter(
            name="raw_covar_factor",
            parameter=torch.nn.Parameter(
                torch.randn(*self.batch_shape, num_tasks, rank).squeeze()
            ),
        )
        if covar_factor_prior is not None:
            if not isinstance(covar_factor_prior, Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(covar_factor_prior).__name__
                )
            self.covar_factor_prior = covar_factor_prior

        if covar_factor_constraint is not None:
            self.register_constraint("raw_covar_factor", covar_factor_constraint)

        if var_prior is not None:
            if not isinstance(var_prior, Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got " + type(var_prior).__name__
                )
            self.register_prior(
                "var_prior",
                var_prior,
                self._var_param,
                self._var_closure,
            )

    @property
    def covar_factor(self):
        if hasattr(self, "raw_covar_factor_constraint"):
            return self.raw_covar_factor_constraint.transform(self.raw_covar_factor)
        else:
            return self.raw_covar_factor

    @covar_factor.setter
    def covar_factor(self, value):
        self._set_covar_factor(value)

    def _set_covar_factor(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_covar_factor)

        self.initialize(
            raw_covar_factor=self.raw_covar_factor_constraint.inverse_transform(value)
            if hasattr(self, "raw_covar_factor_constraint")
            else value
        )

    def _covar_factor_param(self, m: Kernel) -> Tensor:
        return m.covar_factor

    def _covar_factor_closure(self, m: Kernel, v: Tensor) -> Tensor:
        return m._set_covar_factor(v)

    def _var_param(self, m: Kernel) -> Tensor:
        return m.var

    def _var_closure(self, m: Kernel, v: Tensor) -> Tensor:
        return m._set_var(v)

    def _eval_covar_matrix(self):
        cf = self.covar_factor
        if len(cf.size()) == 1:
            cf = cf.unsqueeze(-1)
        return cf @ cf.transpose(-1, -2)  # + torch.diag_embed(self.var)

    def add_prior(self):
        self.register_prior(
            "covar_factor_prior",
            self.covar_factor_prior,
            self._covar_factor_param,
            self._covar_factor_closure,
        )
