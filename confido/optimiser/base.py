import abc

from jaxtyping import PyTree


class AbstractOptimiser:
    @abc.abstractmethod
    def approximate_hessian(self, terms: PyTree):
        pass
