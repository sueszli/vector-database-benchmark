import collections
import itertools
import os
import pathlib
import re
import pynvml
from typing import Union

class Device:
    _nvml_bit_affinity = 64
    _nvml_affinity_elements = (os.cpu_count() + _nvml_bit_affinity - 1) // _nvml_bit_affinity

    def __init__(self, device_idx):
        if False:
            return 10
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self):
        if False:
            i = 10
            return i + 15
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_uuid(self):
        if False:
            print('Hello World!')
        return pynvml.nvmlDeviceGetUUID(self.handle)

    def get_cpu_affinity(self):
        if False:
            i = 10
            return i + 15
        affinity_string = ''
        for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, Device._nvml_affinity_elements):
            affinity_string = '{:064b}'.format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()
        ret = [i for (i, e) in enumerate(affinity_list) if e != 0]
        return ret

def get_thread_siblings_list():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of 2-element integer tuples representing pairs of\n    hyperthreading cores.\n    '
    path = '/sys/devices/system/cpu/cpu*/topology/thread_siblings_list'
    thread_siblings_list = []
    pattern = re.compile('(\\d+)\\D(\\d+)')
    for fname in pathlib.Path(path[0]).glob(path[1:]):
        with open(fname) as f:
            content = f.read().strip()
            res = pattern.findall(content)
            if res:
                pair = tuple(sorted(map(int, res[0])))
                thread_siblings_list.append(pair)
    thread_siblings_list = list(set(thread_siblings_list))
    return thread_siblings_list

def build_thread_siblings_dict(siblings_list):
    if False:
        while True:
            i = 10
    siblings_dict = {}
    for siblings_tuple in siblings_list:
        for core in siblings_tuple:
            siblings_dict[core] = siblings_tuple
    return siblings_dict

def group_list_by_dict(affinity, siblings_dict):
    if False:
        for i in range(10):
            print('nop')
    sorted_affinity = sorted(affinity, key=lambda x: siblings_dict.get(x, (x,)))
    grouped = itertools.groupby(sorted_affinity, key=lambda x: siblings_dict.get(x, (x,)))
    grouped_affinity = []
    for (key, group) in grouped:
        grouped_affinity.append(tuple(group))
    return grouped_affinity

def group_affinity_by_siblings(socket_affinities):
    if False:
        i = 10
        return i + 15
    siblings_list = get_thread_siblings_list()
    siblings_dict = build_thread_siblings_dict(siblings_list)
    grouped_socket_affinities = []
    for socket_affinity in socket_affinities:
        grouped_socket_affinities.append(group_list_by_dict(socket_affinity, siblings_dict))
    return grouped_socket_affinities

def ungroup_affinities(affinities, cores):
    if False:
        print('Hello World!')
    ungrouped_affinities = []
    for affinity in affinities:
        if cores == 'all_logical':
            ungrouped_affinities.append(list(itertools.chain(*affinity)))
        elif cores == 'single_logical':
            ungrouped_affinities.append([group[0] for group in affinity])
        else:
            raise RuntimeError('Unknown cores mode')
    return ungrouped_affinities

def check_socket_affinities(socket_affinities):
    if False:
        print('Hello World!')
    for (i, j) in itertools.product(socket_affinities, socket_affinities):
        if not set(i) == set(j) and (not set(i).isdisjoint(set(j))):
            raise RuntimeError(f'Sets of cores should be either identical or disjoint, but got {i} and {j}.')

def get_socket_affinities(nproc_per_node, exclude_unavailable_cores=True):
    if False:
        while True:
            i = 10
    devices = [Device(i) for i in range(nproc_per_node)]
    socket_affinities = [dev.get_cpu_affinity() for dev in devices]
    if exclude_unavailable_cores:
        available_cores = os.sched_getaffinity(0)
        socket_affinities = [list(set(affinity) & available_cores) for affinity in socket_affinities]
    check_socket_affinities(socket_affinities)
    return socket_affinities

def get_grouped_socket_affinities(nproc_per_node, exclude_unavailable_cores=True):
    if False:
        i = 10
        return i + 15
    socket_affinities = get_socket_affinities(nproc_per_node, exclude_unavailable_cores)
    grouped_socket_affinities = group_affinity_by_siblings(socket_affinities)
    return grouped_socket_affinities

def set_socket_affinity(gpu_id, nproc_per_node, cores):
    if False:
        return 10
    "\n    The process is assigned with all available physical CPU cores from the CPU\n    socket connected to the GPU with a given id.\n\n    Args:\n        gpu_id: index of a GPU\n        nproc_per_node: number of processes per node\n        cores: 'all_logical' or 'single_logical'\n    "
    grouped_socket_affinities = get_grouped_socket_affinities(nproc_per_node)
    ungrouped_affinities = ungroup_affinities(grouped_socket_affinities, cores)
    os.sched_setaffinity(0, ungrouped_affinities[gpu_id])

def set_socket_single_affinity(gpu_id, nproc_per_node, cores):
    if False:
        print('Hello World!')
    "\n    The process is assigned with the first available physical CPU core from the\n    list of all CPU physical cores from the CPU socket connected to the GPU with\n    a given id.\n\n    Args:\n        gpu_id: index of a GPU\n        nproc_per_node: number of processes per node\n        cores: 'all_logical' or 'single_logical'\n    "
    grouped_socket_affinities = get_grouped_socket_affinities(nproc_per_node)
    single_grouped_socket_affinities = [group[:1] for group in grouped_socket_affinities]
    ungrouped_affinities = ungroup_affinities(single_grouped_socket_affinities, cores)
    os.sched_setaffinity(0, ungrouped_affinities[gpu_id])

def set_socket_single_unique_affinity(gpu_id, nproc_per_node, cores):
    if False:
        for i in range(10):
            print('nop')
    "\n    The process is assigned with a single unique available physical CPU core\n    from the list of all CPU cores from the CPU socket connected to the GPU with\n    a given id.\n\n    Args:\n        gpu_id: index of a GPU\n        nproc_per_node: number of processes per node\n        cores: 'all_logical' or 'single_logical'\n    "
    grouped_socket_affinities = get_grouped_socket_affinities(nproc_per_node)
    affinities = []
    assigned_groups = set()
    for grouped_socket_affinity in grouped_socket_affinities:
        for group in grouped_socket_affinity:
            if group not in assigned_groups:
                affinities.append([group])
                assigned_groups.add(group)
                break
    ungrouped_affinities = ungroup_affinities(affinities, cores)
    os.sched_setaffinity(0, ungrouped_affinities[gpu_id])

def set_socket_unique_affinity(gpu_id, nproc_per_node, cores, mode, balanced=True):
    if False:
        while True:
            i = 10
    "\n    The process is assigned with a unique subset of available physical CPU\n    cores from the CPU socket connected to a GPU with a given id.\n    Assignment automatically includes hyperthreading siblings (if siblings are\n    available).\n\n    Args:\n        gpu_id: index of a GPU\n        nproc_per_node: number of processes per node\n        cores: 'all_logical' or 'single_logical'\n        mode: 'contiguous' or 'interleaved'\n        balanced: assign an equal number of physical cores to each process,\n    "
    grouped_socket_affinities = get_grouped_socket_affinities(nproc_per_node)
    grouped_socket_affinities_to_device_ids = collections.defaultdict(list)
    for (idx, grouped_socket_affinity) in enumerate(grouped_socket_affinities):
        grouped_socket_affinities_to_device_ids[tuple(grouped_socket_affinity)].append(idx)
    min_physical_cores_per_gpu = min([len(cores) // len(gpus) for (cores, gpus) in grouped_socket_affinities_to_device_ids.items()])
    grouped_unique_affinities = [None] * nproc_per_node
    for (grouped_socket_affinity, device_ids) in grouped_socket_affinities_to_device_ids.items():
        devices_per_group = len(device_ids)
        if balanced:
            cores_per_device = min_physical_cores_per_gpu
            grouped_socket_affinity = grouped_socket_affinity[:devices_per_group * min_physical_cores_per_gpu]
        else:
            cores_per_device = len(grouped_socket_affinity) // devices_per_group
        for (socket_subgroup_id, device_id) in enumerate(device_ids):
            if mode == 'interleaved':
                unique_grouped_affinity = list(grouped_socket_affinity[socket_subgroup_id::devices_per_group])
            elif mode == 'contiguous':
                unique_grouped_affinity = list(grouped_socket_affinity[socket_subgroup_id * cores_per_device:(socket_subgroup_id + 1) * cores_per_device])
            else:
                raise RuntimeError('Unknown set_socket_unique_affinity mode')
            grouped_unique_affinities[device_id] = unique_grouped_affinity
    ungrouped_affinities = ungroup_affinities(grouped_unique_affinities, cores)
    os.sched_setaffinity(0, ungrouped_affinities[gpu_id])
from enum import Enum, auto

class AffinityMode(Enum):
    none = auto()
    socket = auto()
    socket_single = auto()
    socket_single_unique = auto()
    socket_unique_interleaved = auto()
    socket_unique_contiguous = auto()

def set_affinity(gpu_id, nproc_per_node=None, *, mode: Union[str, AffinityMode]=AffinityMode.socket_unique_contiguous, cores='all_logical', balanced=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    The process is assigned with a proper CPU affinity that matches CPU-GPU\n    hardware architecture on a given platform. Usually, it improves and\n    stabilizes the performance of deep learning training workloads.\n\n    This function assumes that the workload runs in multi-process single-device\n    mode (there are multiple training processes, and each process is running on\n    a single GPU). This is typical for multi-GPU data-parallel training\n    workloads (e.g., using `torch.nn.parallel.DistributedDataParallel`).\n\n    Available affinity modes:\n    * \'socket\' - the process is assigned with all available physical CPU cores\n    from the CPU socket connected to the GPU with a given id.\n    * \'socket_single\' - the process is assigned with the first available\n    physical CPU core from the list of all CPU cores from the CPU socket\n    connected to the GPU with a given id (multiple GPUs could be assigned with\n    the same CPU core).\n    * \'socket_single_unique\' - the process is assigned with a single unique\n    available physical CPU core from the list of all CPU cores from the CPU\n    socket connected to the GPU with a given id.\n    * \'socket_unique_interleaved\' - the process is assigned with a unique\n    subset of available physical CPU cores from the CPU socket connected to a\n    GPU with a given id, cores are assigned with interleaved indexing pattern\n    * \'socket_unique_contiguous\' - (the default) the process is assigned with a\n    unique subset of available physical CPU cores from the CPU socket connected\n    to a GPU with a given id, cores are assigned with contiguous indexing\n    pattern\n\n    Available "cores" modes:\n    * \'all_logical\' - assigns the process with all logical cores associated with\n    a given corresponding physical core (i.e., automatically includes all\n    available hyperthreading siblings)\n    * \'single_logical\' - assigns the process with only one logical core\n    associated with a given corresponding physical core (i.e., excludes\n    hyperthreading siblings)\n\n    \'socket_unique_contiguous\' is the recommended mode for deep learning\n    training workloads on NVIDIA DGX machines.\n\n    Args:\n        gpu_id: integer index of a GPU, value from 0 to \'nproc_per_node\' - 1\n        nproc_per_node: number of processes per node\n        mode: affinity mode\n        balanced: assign an equal number of physical cores to each process,\n            affects only \'socket_unique_interleaved\' and\n            \'socket_unique_contiguous\' affinity modes\n        cores: \'all_logical\' or \'single_logical\'\n\n    Returns a set of logical CPU cores on which the process is eligible to run.\n\n    Example:\n\n    import argparse\n    import os\n\n    import gpu_affinity\n    import torch\n\n\n    def main():\n        parser = argparse.ArgumentParser()\n        parser.add_argument(\n            \'--local_rank\',\n            type=int,\n            default=os.getenv(\'LOCAL_RANK\', 0),\n        )\n        args = parser.parse_args()\n\n        nproc_per_node = torch.cuda.device_count()\n\n        affinity = gpu_affinity.set_affinity(args.local_rank, nproc_per_node)\n        print(f\'{args.local_rank}: core affinity: {affinity}\')\n\n\n    if __name__ == "__main__":\n        main()\n\n    Launch the example with:\n    python -m torch.distributed.launch --nproc_per_node <#GPUs> example.py\n\n\n    WARNING: On DGX A100, only half of the CPU cores have direct access to GPUs.\n    This function restricts execution only to the CPU cores directly connected\n    to GPUs, so on DGX A100, it will limit the code to half of the CPU cores and\n    half of CPU memory bandwidth (which may be fine for many DL models).\n\n    WARNING: Intel\'s OpenMP implementation resets affinity on the first call to\n    an OpenMP function after a fork. It\'s recommended to run with env variable:\n    `KMP_AFFINITY=disabled` if the affinity set by gpu_affinity should be\n    preserved after a fork (e.g. in PyTorch DataLoader workers).\n    '
    if not isinstance(mode, AffinityMode):
        mode = AffinityMode[mode]
    pynvml.nvmlInit()
    if nproc_per_node is None:
        nproc_per_node = pynvml.nvmlDeviceGetCount()
    if mode == AffinityMode.none:
        pass
    elif mode == AffinityMode.socket:
        set_socket_affinity(gpu_id, nproc_per_node, cores)
    elif mode == AffinityMode.socket_single:
        set_socket_single_affinity(gpu_id, nproc_per_node, cores)
    elif mode == AffinityMode.socket_single_unique:
        set_socket_single_unique_affinity(gpu_id, nproc_per_node, cores)
    elif mode == AffinityMode.socket_unique_interleaved:
        set_socket_unique_affinity(gpu_id, nproc_per_node, cores, 'interleaved', balanced)
    elif mode == AffinityMode.socket_unique_contiguous:
        set_socket_unique_affinity(gpu_id, nproc_per_node, cores, 'contiguous', balanced)
    else:
        raise RuntimeError('Unknown affinity mode')
    affinity = os.sched_getaffinity(0)
    return affinity