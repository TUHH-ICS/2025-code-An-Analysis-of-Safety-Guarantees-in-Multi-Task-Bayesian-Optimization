#!/usr/bin/env python3
"""
BayesianOptimization class for performing safe multi-task Bayesian optimization.

Attributes:
    obj: The objective function to be optimized.
    tasks: List of tasks for multi-task optimization.
    bounds: Tensor specifying the bounds for the optimization variables.
    threshold: The safety threshold for the optimization.
    targets_mean_std: Tuple containing the mean and standard deviation of the targets.
    num_acq_samps: List specifying the number of acquisition samples for each task.
    boundary_T: Boundary threshold for the optimization.
    run: Counter for the number of optimization steps.
    best_y: List of best observed values.
    best_x: List of best observed inputs.
    dim: Dimensionality of the optimization problem.
    gp: Gaussian process model used for optimization.

Methods:
    __init__(self, obj, tasks, bounds, threshold, targets_mean_std, num_acq_samps=[1, 1], boundary_T=-15.0):
        Initializes the BayesianOptimization class with the given parameters.

    step(self):
        Performs one step of the Bayesian optimization loop.

    inequality_consts(self, input: Tensor):
        Computes the inequality constraints for the given input.

    update_gp(self, gp, sqrtbeta):
        Updates the Gaussian process model and related attributes.

    _line_search(self, initial_condition, maxiter=20, step_size=2.0):
        Performs a line search to find a feasible initial condition.

    _get_max_observed(self):
        Returns the maximum observed values for each task.

    _get_min_observed(self):
        Returns the minimum observed values for each task.

    _get_best_input(self):
        Returns the best input values for each task.

    _get_initial_cond(self):
        Returns the initial conditions for the optimization.

    get_next_point(self, task, posterior_transform):
        Returns the next point to evaluate for the given task.
"""


import torch
from torch import Tensor
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import qUpperConfidenceBound
from utils.utils import concat_data, unstandardize, standardize
from botorch.acquisition.objective import ScalarizedPosteriorTransform

N_TOL = -1e-6


class BayesianOptimization:
    def __init__(
        self,
        obj,
        tasks,
        bounds,
        threshold,
        targets_mean_std,
        num_acq_samps: list = [1, 1],
        boundary_T=-15.0,
    ):
        self.obj = obj
        self.bounds = bounds
        self.threshold = threshold
        self.boundary_T = boundary_T
        self.mu, self.std = targets_mean_std
        self.num_acq_samps = num_acq_samps
        self.tasks = tasks
        if len(self.num_acq_samps) != len(self.tasks):
            raise ValueError("Number of tasks and number of samples must match")
        self.run = 0
        self.best_y = []
        self.best_x = []
        self.dim = bounds.size(-1)
        self.gp = None

    def step(self):
        self.run += 1
        print("Run : ", self.run)
        print(f"Best value found: {self.observed_max[0]: .3f}")
        print(f"Worst value: {self._get_min_observed()[0]}")
        W = torch.eye(len(self.tasks))
        for i in self.tasks:
            posterior_transform = ScalarizedPosteriorTransform(W[:, i].squeeze())
            new_point = self.get_next_point(i, posterior_transform)
            if i == 0:
                print(f"New Point: {new_point}")
                new_point_task0 = new_point
            if i != 0:
                new_point = torch.vstack((new_point, new_point_task0))
            new_result = self.obj.f(new_point, i)
            self.train_inputs, self.train_tasks, self.unstd_train_targets = concat_data(
                (new_point, i * torch.ones(new_point.shape[0], 1), new_result),
                (self.train_inputs, self.train_tasks, self.unstd_train_targets),
            )
        threshold = unstandardize(self.threshold, self.mu, self.std)
        self.train_targets, self.mu, self.std = standardize(
            self.unstd_train_targets, train_task=self.train_tasks
        )
        self.threshold, _, _ = standardize(threshold, self.mu, self.std)
        self.observed_max = self._get_max_observed()
        self.best_y.append(self.observed_max[0])
        self.best_x.append(self._get_best_input()[0])
        return self.train_inputs, self.train_tasks, self.train_targets

    def inequality_consts(self, input: Tensor):
        self.gp.eval()
        inputx = input.view(int(input.numel() / self.dim), self.dim)
        output = self.gp(torch.hstack((inputx, torch.zeros(inputx.size(0), 1))))
        val = (
            output.mean
            - output.covariance_matrix.diag().sqrt() * self.sqrtbeta
            - self.threshold
        )
        return val.view(inputx.shape[0], 1)

    def update_gp(self, gp, sqrtbeta):
        with torch.no_grad():
            self.train_inputs = gp.train_inputs[0][..., :-1]
            self.train_tasks = gp.train_inputs[0][..., -1:].to(dtype=torch.int32)
            self.train_targets = gp.train_targets.unsqueeze(-1)
            self.unstd_train_targets = unstandardize(
                self.train_targets, self.mu, self.std, self.train_tasks
            )
            self.sqrtbeta = sqrtbeta.detach()
        if self.gp is None:
            self.observed_max = self._get_max_observed()
            self.best_y.append(self.observed_max[0])
            self.best_x.append(self._get_best_input()[0])
        self.gp = gp
        pass

    def _line_search(self, initial_condition, maxiter=20, step_size=2.0):
        k = 1000
        direction = torch.randn(initial_condition.size())
        direction /= (
            torch.linalg.norm(direction, dim=-1, ord=2)
            .unsqueeze(-1)
            .repeat(1, 1, self.dim)
        )
        steps = torch.linspace(0, step_size, k).view(1, k, 1) - step_size / 2
        line_search = initial_condition + steps * direction
        inds = (
            (self.inequality_consts(line_search) >= 0).view(
                initial_condition.size(0), -1
            )
            & torch.all(line_search <= self.bounds[1, :].view(1, 1, self.dim), dim=-1)
            & torch.all(line_search >= self.bounds[0, :].view(1, 1, self.dim), dim=-1)
        )
        for id in range(inds.size(0)):
            possible_steps = steps[:, inds[id, :].squeeze(), :].squeeze()
            if possible_steps.numel() <= 1:
                return initial_condition
            max_step_ind = possible_steps.abs().argmax()
            initial_condition[id] = (
                initial_condition[id] + possible_steps[max_step_ind] * direction[id]
            )
        return initial_condition

    def _get_max_observed(self):
        return [
            torch.max(self.unstd_train_targets[self.train_tasks == i]).item()
            for i in self.tasks
        ]

    def _get_min_observed(self):
        return [
            torch.min(self.unstd_train_targets[self.train_tasks == i]).item()
            for i in self.tasks
        ]

    def _get_best_input(self):
        return [
            self.train_inputs[self.train_tasks.squeeze() == i, ...][
                torch.argmax(self.train_targets[self.train_tasks == i])
            ]
            for i in self.tasks
        ]

    def _get_initial_cond(self):
        _, ind = self.train_targets.sort(dim=0, descending=True)
        sorted_train_inp = self.train_inputs[ind.squeeze(), ...]
        eqfull = self.inequality_consts(sorted_train_inp).squeeze()
        pot_cond = sorted_train_inp.view(self.train_inputs.size())[eqfull >= 0, ...][
            :5, ...
        ]
        return pot_cond.view(pot_cond.size(0), 1, self.dim)

    def get_next_point(self, task, posterior_transform):
        if task == 0:
            init_cond = self._get_initial_cond()
            if init_cond.numel() == 0:
                print(
                    "No feasible initial condition found. Randomly sampling a new one."
                )
                x_new = self.train_inputs[
                    self.train_targets[self.train_tasks == 0].argmax(), :
                ].view(1, self.dim)
                offset = torch.randn(1, self.dim) * 0.005
                ind = (x_new + offset <= self.bounds[1, :].view(1, self.dim)) & (
                    x_new + offset >= self.bounds[0, :].view(1, self.dim)
                )
                x_new[ind] = x_new[ind] + offset[ind]
                x_new[~ind] = x_new[~ind] - offset[~ind]
                return x_new
            else:
                init_cond = self._line_search(init_cond)
            acq = qUpperConfidenceBound(
                self.gp,
                self.sqrtbeta,
                posterior_transform=posterior_transform,
            )
        # if different acquisitions should be used
        else:
            acq = qUpperConfidenceBound(
                self.gp,
                beta=self.sqrtbeta,
                posterior_transform=posterior_transform,
            )
        candidate, tt = optimize_acqf(
            acq_function=acq,
            bounds=(
                self.bounds
                if task == 0
                else self.bounds
                + torch.tensor(
                    [[self.obj.max_disturbance], [-self.obj.max_disturbance]] # max_disturbance is zero for LbSync (only shifts)
                )
            ),
            q=self.num_acq_samps[task],
            num_restarts=init_cond.size(0) if task == 0 else 1,
            raw_samples=512 if task != 0 else None,
            nonlinear_inequality_constraints=(
                [self.inequality_consts] if task == 0 else None
            ),
            batch_initial_conditions=init_cond if task == 0 else None,
            options={"maxiter": 200},
        )
        return candidate
