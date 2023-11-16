from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType

@PublicAPI
class OutputWriter:
    """Writer API for saving experiences from policy evaluation."""

    @PublicAPI
    def write(self, sample_batch: SampleBatchType):
        if False:
            return 10
        'Saves a batch of experiences.\n\n        Args:\n            sample_batch: SampleBatch or MultiAgentBatch to save.\n        '
        raise NotImplementedError

@PublicAPI
class NoopOutput(OutputWriter):
    """Output writer that discards its outputs."""

    @override(OutputWriter)
    def write(self, sample_batch: SampleBatchType):
        if False:
            return 10
        pass