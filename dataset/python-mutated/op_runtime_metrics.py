from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces.physical_operator import PhysicalOperator

@dataclass
class RunningTaskInfo:
    inputs: RefBundle
    num_outputs: int
    bytes_outputs: int

@dataclass
class OpRuntimeMetrics:
    """Runtime metrics for a PhysicalOperator.

    Metrics are updated dynamically during the execution of the Dataset.
    This class can be used for either observablity or scheduling purposes.

    DO NOT modify the fields of this class directly. Instead, use the provided
    callback methods.
    """
    num_inputs_received: int = 0
    bytes_inputs_received: int = 0
    num_inputs_processed: int = field(default=0, metadata={'map_only': True})
    bytes_inputs_processed: int = field(default=0, metadata={'map_only': True})
    num_outputs_generated: int = field(default=0, metadata={'map_only': True})
    bytes_outputs_generated: int = field(default=0, metadata={'map_only': True, 'export_metric': True})
    num_outputs_taken: int = 0
    bytes_outputs_taken: int = 0
    num_outputs_of_finished_tasks: int = field(default=0, metadata={'map_only': True})
    bytes_outputs_of_finished_tasks: int = field(default=0, metadata={'map_only': True})
    num_tasks_submitted: int = field(default=0, metadata={'map_only': True})
    num_tasks_running: int = field(default=0, metadata={'map_only': True})
    num_tasks_have_outputs: int = field(default=0, metadata={'map_only': True})
    num_tasks_finished: int = field(default=0, metadata={'map_only': True})
    obj_store_mem_alloc: int = field(default=0, metadata={'map_only': True, 'export_metric': True})
    obj_store_mem_freed: int = field(default=0, metadata={'map_only': True, 'export_metric': True})
    obj_store_mem_cur: int = field(default=0, metadata={'map_only': True, 'export_metric': True})
    obj_store_mem_peak: int = field(default=0, metadata={'map_only': True})
    obj_store_mem_spilled: int = field(default=0, metadata={'map_only': True, 'export_metric': True})
    block_generation_time: float = field(default=0, metadata={'map_only': True, 'export_metric': True})

    def __init__(self, op: 'PhysicalOperator'):
        if False:
            i = 10
            return i + 15
        from ray.data._internal.execution.operators.map_operator import MapOperator
        self._op = op
        self._is_map = isinstance(op, MapOperator)
        self._running_tasks: Dict[int, RunningTaskInfo] = {}
        self._extra_metrics: Dict[str, Any] = {}

    @property
    def extra_metrics(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Return a dict of extra metrics.'
        return self._extra_metrics

    def as_dict(self, metrics_only: bool=False):
        if False:
            print('Hello World!')
        'Return a dict representation of the metrics.'
        result = []
        for f in fields(self):
            if f.metadata.get('export', True):
                if not self._is_map and f.metadata.get('map_only', False) or (metrics_only and (not f.metadata.get('export_metric', False))):
                    continue
                value = getattr(self, f.name)
                result.append((f.name, value))
        resource_usage = self._op.current_resource_usage()
        result.extend([('cpu_usage', resource_usage.cpu or 0), ('gpu_usage', resource_usage.gpu or 0)])
        result.extend(self._extra_metrics.items())
        return dict(result)

    @classmethod
    def get_metric_keys(cls):
        if False:
            print('Hello World!')
        'Return a list of metric keys.'
        return [f.name for f in fields(cls) if f.metadata.get('export_metric', False)] + ['cpu_usage', 'gpu_usage']

    @property
    def average_num_outputs_per_task(self) -> Optional[float]:
        if False:
            return 10
        'Average number of output blocks per task, or None if no task has finished.'
        if self.num_tasks_finished == 0:
            return None
        else:
            return self.num_outputs_of_finished_tasks / self.num_tasks_finished

    @property
    def average_bytes_outputs_per_task(self) -> Optional[float]:
        if False:
            return 10
        'Average size in bytes of output blocks per task,\n        or None if no task has finished.'
        if self.num_tasks_finished == 0:
            return None
        else:
            return self.bytes_outputs_of_finished_tasks / self.num_tasks_finished

    @property
    def input_buffer_bytes(self) -> int:
        if False:
            print('Hello World!')
        'Size in bytes of input blocks that are not processed yet.'
        return self.bytes_inputs_received - self.bytes_inputs_processed

    @property
    def output_buffer_bytes(self) -> int:
        if False:
            return 10
        'Size in bytes of output blocks that are not taken by the downstream yet.'
        return self.bytes_outputs_generated - self.bytes_outputs_taken

    def on_input_received(self, input: RefBundle):
        if False:
            print('Hello World!')
        'Callback when the operator receives a new input.'
        self.num_inputs_received += 1
        input_size = input.size_bytes()
        self.bytes_inputs_received += input_size
        self.obj_store_mem_cur += input_size
        if self.obj_store_mem_cur > self.obj_store_mem_peak:
            self.obj_store_mem_peak = self.obj_store_mem_cur

    def on_output_taken(self, output: RefBundle):
        if False:
            for i in range(10):
                print('nop')
        'Callback when an output is taken from the operator.'
        output_bytes = output.size_bytes()
        self.num_outputs_taken += 1
        self.bytes_outputs_taken += output_bytes
        self.obj_store_mem_cur -= output_bytes

    def on_task_submitted(self, task_index: int, inputs: RefBundle):
        if False:
            for i in range(10):
                print('nop')
        'Callback when the operator submits a task.'
        self.num_tasks_submitted += 1
        self.num_tasks_running += 1
        self._running_tasks[task_index] = RunningTaskInfo(inputs, 0, 0)

    def on_output_generated(self, task_index: int, output: RefBundle):
        if False:
            print('Hello World!')
        'Callback when a new task generates an output.'
        num_outputs = len(output)
        output_bytes = output.size_bytes()
        self.num_outputs_generated += num_outputs
        self.bytes_outputs_generated += output_bytes
        task_info = self._running_tasks[task_index]
        if task_info.num_outputs == 0:
            self.num_tasks_have_outputs += 1
        task_info.num_outputs += num_outputs
        task_info.bytes_outputs += output_bytes
        self.obj_store_mem_alloc += output_bytes
        self.obj_store_mem_cur += output_bytes
        if self.obj_store_mem_cur > self.obj_store_mem_peak:
            self.obj_store_mem_peak = self.obj_store_mem_cur
        for (block_ref, meta) in output.blocks:
            assert meta.exec_stats and meta.exec_stats.wall_time_s
            self.block_generation_time += meta.exec_stats.wall_time_s
            trace_allocation(block_ref, 'operator_output')

    def on_task_finished(self, task_index: int):
        if False:
            for i in range(10):
                print('nop')
        'Callback when a task is finished.'
        self.num_tasks_running -= 1
        self.num_tasks_finished += 1
        task_info = self._running_tasks[task_index]
        self.num_outputs_of_finished_tasks += task_info.num_outputs
        self.bytes_outputs_of_finished_tasks += task_info.bytes_outputs
        inputs = self._running_tasks[task_index].inputs
        self.num_inputs_processed += len(inputs)
        total_input_size = inputs.size_bytes()
        self.bytes_inputs_processed += total_input_size
        blocks = [input[0] for input in inputs.blocks]
        metadata = [input[1] for input in inputs.blocks]
        ctx = ray.data.context.DataContext.get_current()
        if ctx.enable_get_object_locations_for_metrics:
            locations = ray.experimental.get_object_locations(blocks)
            for (block, meta) in zip(blocks, metadata):
                if locations[block].get('did_spill', False):
                    assert meta.size_bytes is not None
                    self.obj_store_mem_spilled += meta.size_bytes
        self.obj_store_mem_freed += total_input_size
        self.obj_store_mem_cur -= total_input_size
        inputs.destroy_if_owned()
        del self._running_tasks[task_index]