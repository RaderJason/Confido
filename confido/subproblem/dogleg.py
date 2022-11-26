from base import AbstractSubproblem
from trustregionstate import TrustRegionState


class Dogleg(AbstractSubproblem):
    def solve_subproblem(self, state: TrustRegionState):
        unconstrained = self.unconstrained_soln(state)
        if unconstrained is not None:
            return unconstrained

        return super().solve_subproblem()
