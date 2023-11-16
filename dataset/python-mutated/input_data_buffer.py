from typing import Callable, List, Optional
from ray.data._internal.execution.interfaces import ExecutionOptions, PhysicalOperator, RefBundle
from ray.data._internal.stats import StatsDict

class InputDataBuffer(PhysicalOperator):
    """Defines the input data for the operator DAG.

    For example, this may hold cached blocks from a previous Dataset execution, or
    the arguments for read tasks.
    """

    def __init__(self, input_data: Optional[List[RefBundle]]=None, input_data_factory: Callable[[int], List[RefBundle]]=None, num_output_blocks: Optional[int]=None):
        if False:
            return 10
        'Create an InputDataBuffer.\n\n        Args:\n            input_data: The list of bundles to output from this operator.\n            input_data_factory: The factory to get input data, if input_data is None.\n            num_output_blocks: The number of output blocks. If not specified, progress\n                bars total will be set based on num output bundles instead.\n        '
        if input_data is not None:
            assert input_data_factory is None
            self._input_data = input_data[:]
            self._is_input_initialized = True
            self._initialize_metadata()
        else:
            assert input_data_factory is not None
            self._input_data_factory = input_data_factory
            self._is_input_initialized = False
        self._num_output_blocks = num_output_blocks
        super().__init__('Input', [], target_max_block_size=None)

    def start(self, options: ExecutionOptions) -> None:
        if False:
            return 10
        if not self._is_input_initialized:
            self._input_data = self._input_data_factory(self.actual_target_max_block_size)
            self._is_input_initialized = True
            self._initialize_metadata()
        for bundle in self._input_data:
            self._metrics.on_input_received(bundle)
        super().start(options)

    def has_next(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return len(self._input_data) > 0

    def _get_next_inner(self) -> RefBundle:
        if False:
            for i in range(10):
                print('nop')
        return self._input_data.pop(0)

    def _set_num_output_blocks(self, num_output_blocks):
        if False:
            print('Hello World!')
        self._num_output_blocks = num_output_blocks

    def num_outputs_total(self) -> int:
        if False:
            print('Hello World!')
        return self._num_output_blocks or self._num_output_bundles

    def get_stats(self) -> StatsDict:
        if False:
            return 10
        return {}

    def _add_input_inner(self, refs, input_index) -> None:
        if False:
            return 10
        raise ValueError('Inputs are not allowed for this operator.')

    def _initialize_metadata(self):
        if False:
            while True:
                i = 10
        assert self._input_data is not None and self._is_input_initialized
        self._num_output_bundles = len(self._input_data)
        block_metadata = []
        for bundle in self._input_data:
            block_metadata.extend([m for (_, m) in bundle.blocks])
        self._stats = {'input': block_metadata}