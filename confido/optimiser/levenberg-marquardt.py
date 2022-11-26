from typing import NamedTuple

import jax.numpy as jnp
from base import AbstractOptimiser
from jax import jacfwd
from jaxtyping import PyTree
from trustregionstate import TrustRegionState


class SubproblemArgs(NamedTuple):
    # TODO: delete this. make a better way to manage the subproblem args
    lagrange_init: float
    max_subproblem_iter: int


class ClassicRegionArgs(NamedTuple):
    # TODO: delete this. make a better way to manage the region update args
    lower_threshold: float
    upper_threshold: float
    decrease_amount: float
    maintain_amount: float
    increase_amount: float


class LevenbergMarquardt(AbstractOptimiser):
    def __init__(
        self,
        fn: PyTree,
        subproblem_method="lagrange",
        region_updating="classic",
        regularisation="none",
        max_iter=50,
        initial_tr_size=0.5,
        subproblem_args=...,
        region_args=...,
    ):
        super().__init__(
            fn,
            subproblem_method,
            region_updating,
            regularisation,
            max_iter,
            initial_tr_size,
            subproblem_args,
            region_args,
        )

    def _init_state(self, fn, state, initial_guess):
        residuals = fn(initial_guess)
        jac_init = jacfwd(fn, initial_guess)
        grad_init = jac_init.T @ residuals
        hess_init = jac_init.T @ jac_init
        reg_mat = state.regularisation(len(hess_init))
        return TrustRegionState(
            state.max_iter,
            state.initial_tr_size,
            0,
            initial_guess,
            self.loss(residuals),
            grad_init,
            hess_init,
            jac_init,
            reg_mat,
            state.subproblem_algo,
            state.region_select,
            state.subproblem_args,
            state.region_args,
            state.args,
        )

    @staticmethod
    def loss(residuals):
        return jnp.sum(residuals**2)

    def _first_order_quantities(self, fn, step, state):
        new_x = state.current_x + step
        residuals = fn(new_x)
        new_f = self.loss(residuals)
        jac = jacfwd(new_x)
        gradient = jac.T @ residuals

        return TrustRegionState(
            state.max_iter,
            state.trust_region_size,
            state.approx_quality,
            new_x,
            new_f,
            gradient,
            state.approx_hess,
            jac,
            state.regularisation,
            state.subproblem_algo,
            state.region_select,
            state.subproblem_args,
            state.region_args,
        )

    def _approximate_hessian(self, fn, step, state: TrustRegionState):
        return TrustRegionState(
            state.max_iter,
            state.trust_region_size,
            state.agreement,
            state.current_x,
            state.current_fx,
            state.gradient,
            state.jac.T @ state.jac,
            state.jac,
            state.regularisation,
            state.subproblem_algo,
            state.region_select,
            state.subproblem_args,
            state.region_args,
            state.args,
        )
