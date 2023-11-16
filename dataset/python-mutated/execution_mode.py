from enum import Enum
from pyflink.java_gateway import get_gateway
__all__ = ['RuntimeExecutionMode']

class RuntimeExecutionMode(Enum):
    """
    Runtime execution mode of DataStream programs. Among other things, this controls task
    scheduling, network shuffle behavior, and time semantics. Some operations will also change
    their record emission behaviour based on the configured execution mode.

    :data:`STREAMING`:

    The Pipeline will be executed with Streaming Semantics. All tasks will be deployed before
    execution starts, checkpoints will be enabled, and both processing and event time will be
    fully supported.

    :data:`BATCH`:

    The Pipeline will be executed with Batch Semantics. Tasks will be scheduled gradually based
    on the scheduling region they belong, shuffles between regions will be blocking, watermarks
    are assumed to be "perfect" i.e. no late data, and processing time is assumed to not advance
    during execution.

    :data:`AUTOMATIC`:

    Flink will set the execution mode to BATCH if all sources are bounded, or STREAMING if there
    is at least one source which is unbounded.
    """
    STREAMING = 0
    BATCH = 1
    AUTOMATIC = 2

    @staticmethod
    def _from_j_execution_mode(j_execution_mode) -> 'RuntimeExecutionMode':
        if False:
            return 10
        return RuntimeExecutionMode[j_execution_mode.name()]

    def _to_j_execution_mode(self):
        if False:
            i = 10
            return i + 15
        gateway = get_gateway()
        JRuntimeExecutionMode = gateway.jvm.org.apache.flink.api.common.RuntimeExecutionMode
        return getattr(JRuntimeExecutionMode, self.name)