import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict
import ray
if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces.physical_operator import PhysicalOperator
    from ray.data._internal.execution.streaming_executor_state import OpState, Topology
logger = logging.getLogger(__name__)
ENABLED_BACKPRESSURE_POLICIES = []
ENABLED_BACKPRESSURE_POLICIES_CONFIG_KEY = 'backpressure_policies.enabled'

def get_backpressure_policies(topology: 'Topology'):
    if False:
        print('Hello World!')
    data_context = ray.data.DataContext.get_current()
    policies = data_context.get_config(ENABLED_BACKPRESSURE_POLICIES_CONFIG_KEY, ENABLED_BACKPRESSURE_POLICIES)
    return [policy(topology) for policy in policies]

class BackpressurePolicy(ABC):
    """Interface for back pressure policies."""

    @abstractmethod
    def __init__(self, topology: 'Topology'):
        if False:
            i = 10
            return i + 15
        ...

    def calculate_max_blocks_to_read_per_op(self, topology: 'Topology') -> Dict['OpState', int]:
        if False:
            print('Hello World!')
        "Determine how many blocks of data we can read from each operator.\n        The `DataOpTask`s of the operators will stop reading blocks when the limit is\n        reached. Then the execution of these tasks will be paused when the streaming\n        generator backpressure threshold is reached.\n        Used in `streaming_executor_state.py::process_completed_tasks()`.\n\n        Returns: A dict mapping from each operator's OpState to the desired number of\n            blocks to read. For operators that are not in the dict, all available blocks\n            will be read.\n\n        Note: Only one backpressure policy that implements this method can be enabled\n            at a time.\n        "
        return {}

    def can_add_input(self, op: 'PhysicalOperator') -> bool:
        if False:
            while True:
                i = 10
        'Determine if we can add a new input to the operator. If returns False, the\n        operator will be backpressured and will not be able to run new tasks.\n        Used in `streaming_executor_state.py::select_operator_to_run()`.\n\n        Returns: True if we can add a new input to the operator, False otherwise.\n\n        Note, if multiple backpressure policies are enabled, the operator will be\n        backpressured if any of the policies returns False.\n        '
        return True

class ConcurrencyCapBackpressurePolicy(BackpressurePolicy):
    """A backpressure policy that caps the concurrency of each operator.

    The concurrency cap limits the number of concurrently running tasks.
    It will be set to an intial value, and will ramp up exponentially.

    The concrete stategy is as follows:
    - Each PhysicalOperator is assigned an initial concurrency cap.
    - An PhysicalOperator can run new tasks if the number of running tasks is less
      than the cap.
    - When the number of finished tasks reaches a threshold, the concurrency cap will
      increase.
    """
    INIT_CAP = 4
    INIT_CAP_CONFIG_KEY = 'backpressure_policies.concurrency_cap.init_cap'
    CAP_MULTIPLY_THRESHOLD = 0.5
    CAP_MULTIPLY_THRESHOLD_CONFIG_KEY = 'backpressure_policies.concurrency_cap.cap_multiply_threshold'
    CAP_MULTIPLIER = 2.0
    CAP_MULTIPLIER_CONFIG_KEY = 'backpressure_policies.concurrency_cap.cap_multiplier'

    def __init__(self, topology: 'Topology'):
        if False:
            print('Hello World!')
        self._concurrency_caps: dict['PhysicalOperator', float] = {}
        data_context = ray.data.DataContext.get_current()
        self._init_cap = data_context.get_config(self.INIT_CAP_CONFIG_KEY, self.INIT_CAP)
        self._cap_multiplier = data_context.get_config(self.CAP_MULTIPLIER_CONFIG_KEY, self.CAP_MULTIPLIER)
        self._cap_multiply_threshold = data_context.get_config(self.CAP_MULTIPLY_THRESHOLD_CONFIG_KEY, self.CAP_MULTIPLY_THRESHOLD)
        assert self._init_cap > 0
        assert 0 < self._cap_multiply_threshold <= 1
        assert self._cap_multiplier > 1
        logger.debug(f'ConcurrencyCapBackpressurePolicy initialized with config: {self._init_cap}, {self._cap_multiply_threshold}, {self._cap_multiplier}')
        for (op, _) in topology.items():
            self._concurrency_caps[op] = self._init_cap

    def can_add_input(self, op: 'PhysicalOperator') -> bool:
        if False:
            while True:
                i = 10
        metrics = op.metrics
        while metrics.num_tasks_finished >= self._concurrency_caps[op] * self._cap_multiply_threshold:
            self._concurrency_caps[op] *= self._cap_multiplier
            logger.debug(f'Concurrency cap for {op} increased to {self._concurrency_caps[op]}')
        return metrics.num_tasks_running < self._concurrency_caps[op]

class StreamingOutputBackpressurePolicy(BackpressurePolicy):
    """A backpressure policy that throttles the streaming outputs of the `DataOpTask`s.

    The are 2 levels of configs to control the behavior:
    - At the Ray Core level, we use
      `MAX_BLOCKS_IN_GENERATOR_BUFFER` to limit the number of blocks buffered in
      the streaming generator of each OpDataTask. When it's reached, the task will
      be blocked at `yield` until the caller reads another `ObjectRef.
    - At the Ray Data level, we use
      `MAX_BLOCKS_IN_GENERATOR_BUFFER` to limit the number of blocks buffered in the
      output queue of each operator. When it's reached, we'll stop reading from the
      streaming generators of the op's tasks, and thus trigger backpressure at the
      Ray Core level.

    Thus, total number of buffered blocks for each operator can be
    `MAX_BLOCKS_IN_GENERATOR_BUFFER * num_running_tasks +
    MAX_BLOCKS_IN_OP_OUTPUT_QUEUE`.
    """
    MAX_BLOCKS_IN_GENERATOR_BUFFER = 10
    MAX_BLOCKS_IN_GENERATOR_BUFFER_CONFIG_KEY = 'backpressure_policies.streaming_output.max_blocks_in_generator_buffer'
    MAX_BLOCKS_IN_OP_OUTPUT_QUEUE = 20
    MAX_BLOCKS_IN_OP_OUTPUT_QUEUE_CONFIG_KEY = 'backpressure_policies.streaming_output.max_blocks_in_op_output_queue'

    def __init__(self, topology: 'Topology'):
        if False:
            while True:
                i = 10
        data_context = ray.data.DataContext.get_current()
        self._max_num_blocks_in_streaming_gen_buffer = data_context.get_config(self.MAX_BLOCKS_IN_GENERATOR_BUFFER_CONFIG_KEY, self.MAX_BLOCKS_IN_GENERATOR_BUFFER)
        assert self._max_num_blocks_in_streaming_gen_buffer > 0
        data_context._task_pool_data_task_remote_args['_generator_backpressure_num_objects'] = 2 * self._max_num_blocks_in_streaming_gen_buffer
        self._max_num_blocks_in_op_output_queue = data_context.get_config(self.MAX_BLOCKS_IN_OP_OUTPUT_QUEUE_CONFIG_KEY, self.MAX_BLOCKS_IN_OP_OUTPUT_QUEUE)
        assert self._max_num_blocks_in_op_output_queue > 0

    def calculate_max_blocks_to_read_per_op(self, topology: 'Topology') -> Dict['OpState', int]:
        if False:
            i = 10
            return i + 15
        max_blocks_to_read_per_op: Dict['OpState', int] = {}
        downstream_num_active_tasks = 0
        for (op, state) in reversed(topology.items()):
            max_blocks_to_read_per_op[state] = self._max_num_blocks_in_op_output_queue - state.outqueue_num_blocks()
            if downstream_num_active_tasks == 0:
                max_blocks_to_read_per_op[state] = max(max_blocks_to_read_per_op[state], 1)
            downstream_num_active_tasks += len(op.get_active_tasks())
        return max_blocks_to_read_per_op