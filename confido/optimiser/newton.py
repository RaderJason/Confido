from base import AbstractOptimiser
from jax import grad, hessian
from jaxtyping import PyTree
from trustregionstate import TrustRegionState


class NewtonTrustRegion(AbstractOptimiser):
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
        loss = fn(initial_guess)
        grad_init = grad(fn, initial_guess)
        hess_init = hessian(fn, initial_guess)
        reg_mat = state.regularisation(len(hess_init))
        return TrustRegionState(
            state.max_iter,
            state.initial_tr_size,
            0,
            initial_guess,
            loss,
            grad_init,
            hess_init,
            None,
            reg_mat,
            state.subproblem_algo,
            state.region_select,
            state.subproblem_args,
            state.region_args,
            state.args,
        )

    def _first_order_quantities(self, fn, step, state):
        new_x = state.current_x + step
        new_f = fn(new_x)
        gradient = grad(fn, new_x)

        return TrustRegionState(
            state.max_iter,
            state.trust_region_size,
            state.approx_quality,
            new_x,
            new_f,
            gradient,
            state.approx_hess,
            state.jac,
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
            hessian(fn, state.current_x),
            state.jac,
            state.regularisation,
            state.subproblem_algo,
            state.region_select,
            state.subproblem_args,
            state.region_args,
            state.args,
        )
