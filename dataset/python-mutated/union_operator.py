from typing import List
from ray.data._internal.execution.interfaces import ExecutionOptions, PhysicalOperator, RefBundle
from ray.data._internal.execution.operators.base_physical_operator import NAryOperator
from ray.data._internal.stats import StatsDict

class UnionOperator(NAryOperator):
    """An operator that combines output blocks from
    two or more input operators into a single output."""

    def __init__(self, *input_ops: PhysicalOperator):
        if False:
            i = 10
            return i + 15
        'Create a UnionOperator.\n\n        Args:\n            input_ops: Operators generating input data for this operator to union.\n        '
        self._preserve_order = False
        self._input_buffers: List[List[RefBundle]] = [[] for _ in range(len(input_ops))]
        self._input_idx_to_output = 0
        self._output_buffer: List[RefBundle] = []
        self._stats: StatsDict = {}
        super().__init__(*input_ops)

    def start(self, options: ExecutionOptions):
        if False:
            print('Hello World!')
        self._preserve_order = options.preserve_order
        super().start(options)

    def num_outputs_total(self) -> int:
        if False:
            print('Hello World!')
        num_outputs = 0
        for input_op in self.input_dependencies:
            num_outputs += input_op.num_outputs_total()
        return num_outputs

    def _add_input_inner(self, refs: RefBundle, input_index: int) -> None:
        if False:
            while True:
                i = 10
        assert not self.completed()
        assert 0 <= input_index <= len(self._input_dependencies), input_index
        if not self._preserve_order:
            self._output_buffer.append(refs)
        elif input_index == self._input_idx_to_output:
            self._output_buffer.append(refs)
        else:
            self._input_buffers[input_index].append(refs)

    def input_done(self, input_index: int) -> None:
        if False:
            print('Hello World!')
        'When `self._preserve_order` is True, change the\n        output buffer source to the next input dependency\n        once the current input dependency calls `input_done()`.'
        if not self._preserve_order:
            return
        if not input_index == self._input_idx_to_output:
            return
        next_input_idx = self._input_idx_to_output + 1
        if next_input_idx < len(self._input_buffers):
            self._output_buffer.extend(self._input_buffers[next_input_idx])
            self._input_buffers[next_input_idx].clear()
            self._input_idx_to_output = next_input_idx
        super().input_done(input_index)

    def all_inputs_done(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._preserve_order:
            for (idx, input_buffer) in enumerate(self._input_buffers):
                assert len(input_buffer) == 0, f'Input at index {idx} still has {len(input_buffer)} blocks remaining.'
        super().all_inputs_done()

    def has_next(self) -> bool:
        if False:
            print('Hello World!')
        return len(self._output_buffer) > 0

    def _get_next_inner(self) -> RefBundle:
        if False:
            for i in range(10):
                print('nop')
        return self._output_buffer.pop(0)

    def get_stats(self) -> StatsDict:
        if False:
            for i in range(10):
                print('nop')
        return self._stats