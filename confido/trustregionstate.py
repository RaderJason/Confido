from typing import NamedTuple, Optional

import jax.numpy as jnp
from jaxtyping import PyTree
from region_selection import AbstractRegionSelector
from subproblem import AbstractSubproblem


class TrustRegionState(NamedTuple):
    """
    store the current trust region state.
    max_iter: maximum number of iterations, constant
    trust_region_size: current trust region size
    agreement: current agreement of the model function
            (f_old - f_new)/(model_old - model_new). Potentially
            a running average.
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
    agreement: PyTree
    current_x: jnp.array
    current_fx: jnp.array
    gradient: jnp.array
    approx_hess: jnp.array
    jac: jnp.array[Optional]
    regularisation: jnp.array[Optional]
    subproblem_algo: AbstractSubproblem
    region_select: AbstractRegionSelector
    subproblem_args: PyTree[Optional]
    region_args: PyTree[Optional]
    args: PyTree
