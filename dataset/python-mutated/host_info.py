"""Helps to collect information about the host of an experiment."""
import os
import platform
import re
import subprocess
from xml.etree import ElementTree
import warnings
from typing import List
import cpuinfo
from sacred.utils import optional_kwargs_decorator
from sacred.settings import SETTINGS
__all__ = ('host_info_gatherers', 'get_host_info', 'host_info_getter')
host_info_gatherers = {}

class IgnoreHostInfo(Exception):
    """Used by host_info_getters to signal that this cannot be gathered."""

class HostInfoGetter:

    def __init__(self, getter_function, name):
        if False:
            i = 10
            return i + 15
        self.getter_function = getter_function
        self.name = name

    def __call__(self):
        if False:
            i = 10
            return i + 15
        return self.getter_function()

    def get_info(self):
        if False:
            i = 10
            return i + 15
        return self.getter_function()

def host_info_gatherer(name):
    if False:
        i = 10
        return i + 15

    def wrapper(f):
        if False:
            for i in range(10):
                print('nop')
        return HostInfoGetter(f, name)
    return wrapper

def check_additional_host_info(additional_host_info: List[HostInfoGetter]):
    if False:
        return 10
    names_taken = [x.name for x in _host_info_gatherers_list]
    for getter in additional_host_info:
        if getter.name in names_taken:
            error_msg = 'Key {} used in `additional_host_info` already exists as a default gatherer function. Do not use the following keys: {}'.format(getter.name, names_taken)
            raise KeyError(error_msg)

def get_host_info(additional_host_info: List[HostInfoGetter]=None):
    if False:
        while True:
            i = 10
    'Collect some information about the machine this experiment runs on.\n\n    Returns\n    -------\n    dict\n        A dictionary with information about the CPU, the OS and the\n        Python version of this machine.\n\n    '
    additional_host_info = additional_host_info or []
    additional_host_info = additional_host_info + _host_info_gatherers_list
    all_host_info_gatherers = host_info_gatherers.copy()
    for getter in additional_host_info:
        all_host_info_gatherers[getter.name] = getter
    host_info = {}
    for (k, v) in all_host_info_gatherers.items():
        try:
            host_info[k] = v()
        except IgnoreHostInfo:
            pass
    return host_info

@optional_kwargs_decorator
def host_info_getter(func, name=None):
    if False:
        print('Hello World!')
    '\n    The decorated function is added to the process of collecting the host_info.\n\n    This just adds the decorated function to the global\n    ``sacred.host_info.host_info_gatherers`` dictionary.\n    The functions from that dictionary are used when collecting the host info\n    using :py:func:`~sacred.host_info.get_host_info`.\n\n    Parameters\n    ----------\n    func : callable\n        A function that can be called without arguments and returns some\n        json-serializable information.\n    name : str, optional\n        The name of the corresponding entry in host_info.\n        Defaults to the name of the function.\n\n    Returns\n    -------\n    The function itself.\n\n    '
    warnings.warn('The host_info_getter is deprecated. Please use the `additional_host_info` argument in the Experiment constructor.', DeprecationWarning)
    name = name or func.__name__
    host_info_gatherers[name] = func
    return func

@host_info_gatherer(name='hostname')
def _hostname():
    if False:
        print('Hello World!')
    return platform.node()

@host_info_gatherer(name='os')
def _os():
    if False:
        print('Hello World!')
    return [platform.system(), platform.platform()]

@host_info_gatherer(name='python_version')
def _python_version():
    if False:
        i = 10
        return i + 15
    return platform.python_version()

@host_info_gatherer(name='cpu')
def _cpu():
    if False:
        return 10
    if not SETTINGS.HOST_INFO.INCLUDE_CPU_INFO:
        return
    if platform.system() == 'Windows':
        return _get_cpu_by_pycpuinfo()
    try:
        if platform.system() == 'Darwin':
            return _get_cpu_by_sysctl()
        elif platform.system() == 'Linux':
            return _get_cpu_by_proc_cpuinfo()
    except Exception:
        return _get_cpu_by_pycpuinfo()

@host_info_gatherer(name='gpus')
def _gpus():
    if False:
        for i in range(10):
            print('nop')
    if not SETTINGS.HOST_INFO.INCLUDE_GPU_INFO:
        return
    try:
        xml = subprocess.check_output(['nvidia-smi', '-q', '-x']).decode('utf-8', 'replace')
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as e:
        raise IgnoreHostInfo() from e
    gpu_info = {'gpus': []}
    for child in ElementTree.fromstring(xml):
        if child.tag == 'driver_version':
            gpu_info['driver_version'] = child.text
        if child.tag != 'gpu':
            continue
        fb_memory_usage = child.find('fb_memory_usage').find('total').text
        if fb_memory_usage == 'Insufficient Permissions':
            mig = child.find('mig_devices').find('mig_device')
            fb_memory_usage = mig.find('fb_memory_usage').find('total').text
        gpu = {'model': child.find('product_name').text, 'total_memory': int(fb_memory_usage.split()[0]), 'persistence_mode': child.find('persistence_mode').text == 'Enabled'}
        gpu_info['gpus'].append(gpu)
    return gpu_info

@host_info_gatherer(name='ENV')
def _environment():
    if False:
        while True:
            i = 10
    keys_to_capture = SETTINGS.HOST_INFO.CAPTURED_ENV
    return {k: os.environ[k] for k in keys_to_capture if k in os.environ}
_host_info_gatherers_list = [_hostname, _os, _python_version, _cpu, _gpus, _environment]

def _get_cpu_by_sysctl():
    if False:
        i = 10
        return i + 15
    os.environ['PATH'] += ':/usr/sbin'
    command = ['sysctl', '-n', 'machdep.cpu.brand_string']
    return subprocess.check_output(command).decode().strip()

def _get_cpu_by_proc_cpuinfo():
    if False:
        for i in range(10):
            print('nop')
    command = ['cat', '/proc/cpuinfo']
    all_info = subprocess.check_output(command).decode()
    model_pattern = re.compile('^\\s*model name\\s*:')
    for line in all_info.split('\n'):
        if model_pattern.match(line):
            return model_pattern.sub('', line, 1).strip()

def _get_cpu_by_pycpuinfo():
    if False:
        for i in range(10):
            print('nop')
    return cpuinfo.get_cpu_info().get('brand_raw', 'Unknown')