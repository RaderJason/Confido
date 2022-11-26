import abc

from jax.numpy.linalg import cholesky, norm
from jax.scipy.linalg import solve_triangular
from trustregionstate import TrustRegionState


class AbstractSubproblem:
    @staticmethod
    def unconstrained_soln(state: TrustRegionState):
        """
        gives the exact minimiser of the subproblem when
        ||Dp|| < trust region.
        """
        # TODO check cho_solve and see if this can be done
        # in one step

        cholesky_factor = cholesky(state.approx_hess)
        step = solve_triangular(cholesky_factor.T, -state.grad, lower=False)
        step = solve_triangular(cholesky_factor, step, lower=True)

        if norm(state.regularisation_mat @ step) < state.trust_region_size:
            return step
        else:
            pass

    @abc.abstractmethod
    def solve_subproblem(self, state: TrustRegionState):
        pass
