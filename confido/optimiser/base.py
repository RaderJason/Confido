import abc
from typing import NamedTuple

import jax.numpy as jnp
import regularisation as reg
from jaxtyping import PyTree
from region_selection import ClassicRegion
from subproblem import Cauchy, Dogleg, Lagrange
from trustregionstate import TrustRegionState


def _diverged():
    pass


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


class AbstractOptimiser:
    def __init__(
        self,
        fn: PyTree,
        subproblem_method="lagrange",
        region_updating="classic",
        regularisation="none",
        max_iter=50,
        initial_tr_size=0.5,
        subproblem_args=SubproblemArgs(lagrange_init=1, max_subproblem_iter=5),
        region_args=ClassicRegionArgs(0.25, 0.75, 1 / 2, 1, 2),
    ):
        # these are all initialised in _init_state when
        # LM is called. It's not important for now.
        jac_init = None
        grad_init = None
        hess_approx = None

        ###
        # NOTE: when this is initialised we store the function
        # to compute the reg matrix in self.state. However, the
        # state passed around the rest of the time is the concrete
        # matrix for regularisation, potentially updated each time.
        ###

        reg_fun = self._choose_reg(regularisation)

        subalgorithm = self._choose_subproblem_algorithm(subproblem_method)
        region_update_method = self._choose_region_update(region_updating)

        self.state = TrustRegionState(
            max_iter,
            initial_tr_size,
            0,
            None,
            None,
            grad_init,
            hess_approx,
            jac_init,
            reg_fun,
            subalgorithm,
            region_update_method,
            subproblem_args,
            region_args,
        )

    @abc.abstractmethod
    def _approximate_hessian(self, fn, step, state: TrustRegionState):
        pass

    @abc.abstractmethod
    def _first_order_quantities(self, state: TrustRegionState):
        pass

    @abc.abstractmethod
    def _init_state(self, fn, state, initial_guess):
        pass

    @staticmethod
    def _converged(factor: float, tol: float):
        return (factor > 0) and (factor < tol)

    @staticmethod
    def _choose_reg(n: int, regularisation: str):
        if regularisation == "none":
            return reg.no_regularisation
        if regularisation == "smooth":
            return reg.second_diff
        if regularisation == "classic":
            return reg.classic_lm
        else:
            raise ValueError(
                "Only regularisation options currently suppored are \
                    'none', 'smooth', and 'classic'"
            )

    @staticmethod
    def _choose_subproblem_algorithm(subproblem_method: str):
        if subproblem_method == "lagrange":
            return Lagrange()
        if subproblem_method == "cauchy":
            return Cauchy()
        if subproblem_method == "dogleg":
            return Dogleg()
        else:
            raise ValueError(
                "Only supported subproblem algorithms at this time \
                    are 'cauchy', 'dogleg', and 'lagrange'"
            )

    @staticmethod
    def _choose_region_update(region_updating: str):
        if region_updating == "classic":
            return ClassicRegion()
        else:
            raise ValueError(
                "Only supported region update methods at this time: 'classic'"
            )

    def _approx_quality(
        f_old: float,
        f_new: float,
        step: jnp.array,
        grad: jnp.array,
        hess_approx: jnp.array,
    ):
        """
        An indication of how well the quadratic approximation is fitting
        the true function within the trust region. This is true decrease in f
        over expected decrease in f
        """

        # for a quadratic approximation m this is m(0) - m(p)
        expected_decrease = -grad.T @ step - 1 / 2 * step.T @ hess_approx @ step

        true_decrease = f_old - f_new

        return true_decrease / expected_decrease

    @staticmethod
    def _update_subproblem_args(state, new_info):
        ###
        # TODO: implement me! (and move me)
        # may need to be moved entirely, but we may want to
        # carry around an extra state and process it
        ###
        return state.suproblem_args

    @staticmethod
    def _update_region_args(state, new_info):
        ###
        # TODO: implement me! (and move me)
        # may need to be moved entirely, but we may want to
        # carry around an extra state and process it
        ###
        return state.region_args

    def _update_regularisation(state, new_info):
        return state.regularisation

    def _update_state(
        self,
        fn,
        state: TrustRegionState,
        step: jnp.array,
        approx_quality: float,
        tr_size: float,
    ):
        """
        update the trust region state with new info given the value of the update
        """
        if approx_quality < 0:
            # if the approx quality is < 0 then we increased and LM
            # is a monotonically decreasing algorithm so we reject the step and
            # continue with a smaller trust region.
            # there has to be a cleaner way to make this update though.
            return state._replace(trust_region_size=tr_size)

        state = self._first_order_quantities(fn, step, state)
        state = self._approximate_hessian(fn, step, state)
        subproblem_args = self._update_subproblem_args(state)
        region_args = self._update_region_args(state)
        regularisation = self._update_regularisation

        return state._replace(
            trust_region_size=tr_size,
            regularisation=regularisation,
            subproblem_args=subproblem_args,
            region_args=region_args
            )

    def __call__(self, fn: callable, initial_guess: PyTree, params: PyTree, tol=1e-4):
        iters = 0
        if params is None:
            fun = fn
        else:

            def fun(x):
                fn(x, params)

        state = self._init_state(fun, self.state, initial_guess)

        # NOTE: this might cause a problem with JAX
        state.current_x = initial_guess
        while not self._converged(self.state.gradient, tol) and iters < state.max_iter:
            iters += 1
            proposed_step = state.subalgorithm.solve(state)

            proposed_f = self.loss(fun(state.current_x + proposed_step))

            approx_quality = self._approx_quality(
                state.current_fx,
                proposed_f,
                proposed_step,
                state.gradient,
                state.hess_approx,
            )

            tr_size = state.region_select.update(state, proposed_step, approx_quality)

            state = self._update_state(
                fun, state, proposed_step, approx_quality, tr_size
            )
            iters += 1
        return state
