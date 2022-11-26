import jax.numpy as jnp
from base import AbstractSubproblem
from jax.numpy.linalg import cholesky, norm
from jax.scipy.linalg import solve_triangular
from optimiser import TrustRegionState


class Lagrange(AbstractSubproblem):
    def solve_subproblem(self, state: TrustRegionState):
        """
        state.subproblem_args = {
            lagrange_init: initial lagrange multiplier weight,
            max_iters: max number of iterations for lagrange subproblem
            }

        Iteratively solve the trust region subproblem using
        lagrange multipliers approach following Nocedal Wright.
        This is the standard approach used in LM.
        """

        ###
        # TODO: add a check to make sure that lagrange > -lambda_1 for lambda_1
        # the smallest eigenvalue of B. Otherwise we have no guarantees that
        # this algorithm will do anything
        #
        # TODO: handle "the hard case", when q_1^T g = 0(q_1 the first eigenvec of
        # B). See pages 87-88 of Nocedal Wright
        ###

        # NOTE: this should be the same as choosing lagrange_init = 0
        unconstrained = self.unconstrained_soln(state)
        if unconstrained is not None:
            return unconstrained

        n_iters = 0

        lagrange_init = state.subproblem_args.lagrange_init
        max_iters = state.subproblem_args.max_iters

        lagrange_mult = lagrange_init
        for _ in range(max_iters):
            cholesky_factor = cholesky(
                state.hess_approx + lagrange_mult * jnp.eye(state.hess_approx.shape[0])
            )

            # standard Cholesky solve
            step = solve_triangular(cholesky_factor.T, -state.grad, lower=False)
            step = solve_triangular(cholesky_factor, step, lower=True)

            d_step = solve_triangular(cholesky_factor, step, lower=True)

            step_norm = norm(step)

            lagrange_multp1 = lagrange_mult + (step_norm / norm(d_step)) ** 2 * (
                (step_norm - state.trust_region) / state.trust_region
            )

            if jnp.isnan(lagrange_multp1) or jnp.any(jnp.isnan(step)):
                lagrange_multp1 = 10 * lagrange_mult

            lagrange_mult = lagrange_multp1

            if step_norm < state.trust_region or n_iters > max_iters:
                break
            n_iters += 1

        return step
