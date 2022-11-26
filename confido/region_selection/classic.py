import jax.numpy as jnp
from base import AbstractRegionSelector
from trustregionstate import TrustRegionState


class ClassicRegion(AbstractRegionSelector):
    def update(
        self, state: TrustRegionState, proposed_step: jnp.array, approx_quality: float
    ):
        """
        The classic trust region update which states that under a
        given threshold, the trust region should be decreased. Above
        that threshold but below an upper threshold a relatively small
        increase/maintenance factor is applied, and if it is above the
        upper threshold then a larger increase is applied up to
        a maximum trust region size.
        """
        region_size = state.trust_region_size
        if approx_quality < state.region_args.lower_threshold:
            return region_size * state.region_args.decrease_amount
        if approx_quality < state.region_args.upper_threshold:
            return region_size * state.region_args.maintain_amount
        else:
            return min(state.region_args.increase_amount, state.region_args.max_tr)
