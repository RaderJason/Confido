import abc


class AbstractRegionSelector:
    @abc.abstractmethod
    def update(self, state, proposed_step, approx_quality):
        pass
