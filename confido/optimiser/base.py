import abc
from typing import Optional

import jax.numpy as jnp
from jaxtyping import PyTree
from region_selection import AbstractRegionSelector
from subproblem import AbstractSubproblem


def TrustRegionState(NamedTuple):
    """
    store the current trust region state.
    max_iter: maximum number of iterations, constant
    trust_region_size: current trust region size
    residual: current residual
    current_x: the current location of the algorithm
    gradient: the gradient of fn at current_x
    approx_hess: the approximation to the hessian at current_x
    jac: if needed (for least squares), the jacobian at x
    subproblem_algo: the algorithm being used to solve the
                    trust region subproblem
    region_select: the algorithm used for selecting the next subproblem
    args: other arguments
    """
    max_iter: int
    trust_region_size: float
    residual: float
    current_x: jnp.array
    gradient: jnp.array
    approx_hess: jnp.array
    jac: jnp.array[Optional]
    subproblem_algo: AbstractSubproblem
    region_select: AbstractRegionSelector
    args: PyTree


def _diverged():
    pass


def _converged(factor: float, tol: float):
    return (factor > 0) and (factor < tol)


class AbstractOptimiser:
    @abc.abstractmethod
    def _approximate_hessian(self, terms: PyTree):
        pass

    @abc.abstractmethod
    def _solve_subproblem(self, terms: PyTree):
        pass

    @abc.abstractmethod
    def _update_region(self, terms: PyTree):
        pass

    @abc.abstractmethod
    def _update_solution(self, terms: PyTree):
        pass

    @abc.abstractmethod
    def _regularisation(self, terms: PyTree):
        pass

    @abc.abstractmethod
    def __call__(self, fn: callable, initial_guess: PyTree, params: PyTree):
        """
        minimise fn with respect to params.
        fn: function fn(x, params) mapping PyTree -> PyTree to be minimise
        """
        pass
