import dataclasses
from dataclasses import dataclass
from typing import Dict, Union
from ludwig.utils.data_utils import flatten_dict

@dataclass
class DeviceUsageMetrics:
    max_memory_used: float
    average_memory_used: float

@dataclass
class SystemResourceMetrics:
    code_block_tag: str
    cpu_name: str
    cpu_architecture: str
    num_cpu: int
    total_cpu_memory_size: float
    ludwig_version: str
    total_execution_time: float
    disk_footprint: float
    max_cpu_utilization: float
    max_cpu_memory_usage: float
    min_global_cpu_memory_available: float
    max_global_cpu_utilization: float
    average_cpu_utilization: float
    average_cpu_memory_usage: float
    average_global_cpu_memory_available: float
    average_global_cpu_utilization: float
    device_usage: Dict[str, DeviceUsageMetrics]

@dataclass
class TorchProfilerMetrics:
    torch_cpu_time: float
    torch_cuda_time: float
    num_oom_events: int
    device_usage: Dict[str, DeviceUsageMetrics]

def profiler_dataclass_to_flat_dict(data: Union[SystemResourceMetrics, TorchProfilerMetrics]) -> Dict:
    if False:
        print('Hello World!')
    'Returns a flat dictionary representation, with the device_usage key removed.'
    nested_dict = dataclasses.asdict(data)
    nested_dict[''] = nested_dict.pop('device_usage')
    return flatten_dict(nested_dict, sep='')