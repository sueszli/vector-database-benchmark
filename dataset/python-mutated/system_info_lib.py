"""Library for getting system information during TensorFlow tests."""
import glob
import multiprocessing
import platform
import re
import socket
import cpuinfo
import psutil
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from tensorflow.tools.test import gpu_info_lib

def gather_machine_configuration():
    if False:
        return 10
    'Gather Machine Configuration.  This is the top level fn of this library.'
    config = test_log_pb2.MachineConfiguration()
    config.cpu_info.CopyFrom(gather_cpu_info())
    config.platform_info.CopyFrom(gather_platform_info())
    for d in gather_available_device_info():
        config.available_device_info.add().CopyFrom(d)
    for gpu in gpu_info_lib.gather_gpu_devices():
        config.device_info.add().Pack(gpu)
    config.memory_info.CopyFrom(gather_memory_info())
    config.hostname = gather_hostname()
    return config

def gather_hostname():
    if False:
        for i in range(10):
            print('nop')
    return socket.gethostname()

def gather_memory_info():
    if False:
        return 10
    'Gather memory info.'
    mem_info = test_log_pb2.MemoryInfo()
    vmem = psutil.virtual_memory()
    mem_info.total = vmem.total
    mem_info.available = vmem.available
    return mem_info

def gather_cpu_info():
    if False:
        return 10
    'Gather CPU Information.  Assumes all CPUs are the same.'
    cpu_info = test_log_pb2.CPUInfo()
    cpu_info.num_cores = multiprocessing.cpu_count()
    try:
        with gfile.GFile('/proc/self/status', 'rb') as fh:
            nc = re.search('(?m)^Cpus_allowed:\\s*(.*)$', fh.read().decode('utf-8'))
        if nc:
            cpu_info.num_cores_allowed = bin(int(nc.group(1).replace(',', ''), 16)).count('1')
    except errors.OpError:
        pass
    finally:
        if cpu_info.num_cores_allowed == 0:
            cpu_info.num_cores_allowed = cpu_info.num_cores
    info = cpuinfo.get_cpu_info()
    cpu_info.cpu_info = info['brand']
    cpu_info.num_cores = info['count']
    cpu_info.mhz_per_cpu = info['hz_advertised_raw'][0] / 1000000.0
    l2_cache_size = re.match('(\\d+)', str(info.get('l2_cache_size', '')))
    if l2_cache_size:
        cpu_info.cache_size['L2'] = int(l2_cache_size.group(0)) * 1024
    try:
        cpu_governors = set([gfile.GFile(f, 'r').readline().rstrip() for f in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')])
        if cpu_governors:
            if len(cpu_governors) > 1:
                cpu_info.cpu_governor = 'mixed'
            else:
                cpu_info.cpu_governor = list(cpu_governors)[0]
    except errors.OpError:
        pass
    return cpu_info

def gather_available_device_info():
    if False:
        for i in range(10):
            print('nop')
    'Gather list of devices available to TensorFlow.\n\n  Returns:\n    A list of test_log_pb2.AvailableDeviceInfo messages.\n  '
    device_info_list = []
    devices = device_lib.list_local_devices()
    for d in devices:
        device_info = test_log_pb2.AvailableDeviceInfo()
        device_info.name = d.name
        device_info.type = d.device_type
        device_info.memory_limit = d.memory_limit
        device_info.physical_description = d.physical_device_desc
        device_info_list.append(device_info)
    return device_info_list

def gather_platform_info():
    if False:
        for i in range(10):
            print('nop')
    'Gather platform info.'
    platform_info = test_log_pb2.PlatformInfo()
    (platform_info.bits, platform_info.linkage) = platform.architecture()
    platform_info.machine = platform.machine()
    platform_info.release = platform.release()
    platform_info.system = platform.system()
    platform_info.version = platform.version()
    return platform_info