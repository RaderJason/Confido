from base import AbstractSubproblem
from jax.numpy.linalg import norm
from trustregionstate import TrustRegionState


class Cauchy(AbstractSubproblem):
    """
    the cauchy point solution of a trust region problem is found by
    first finding the step p which minimises the linear approximation to
    the trust region problem f + g^t p, then finding a scaler T which
    minimises the quadratic trust region subproblem along the direction
    found by the linear approximation. T can be represented explicitly, see
    Nocedal Wright.
    """

    def solve_subproblem(self, state: TrustRegionState):
        unconstrained = self.unconstrained_soln(state)
        if unconstrained is not None:
            return unconstrained

        grad_norm = norm(state.gradient)

        linear_subproblem = -state.trust_region_size / grad_norm * state.gradient

        # if convexity term < 0, then the quadratic approximation is
        # monotonically decreasing as a function of the scaling and we should simply
        # take the scaling to be 1
        convexity_term = state.gradient.T @ state.approx_hess @ state.gradient

        if convexity_term <= 0:
            return linear_subproblem
        else:
            scaling = (grad_norm**3) / (state.trust_region_size * convexity_term)
            return min(scaling, 1) * linear_subproblem
