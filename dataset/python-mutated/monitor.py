import datetime
import json
import signal
import sys
import time
from typing import Any, Dict, List
import psutil
import pynvml
sys.path.append('/opt/rocm/libexec/rocm_smi')
try:
    from ctypes import byref, c_uint32, c_uint64
    from rsmiBindings import rocmsmi, rsmi_process_info_t, rsmi_status_t
except ImportError as e:
    pass

def get_processes_running_python_tests() -> List[Any]:
    if False:
        for i in range(10):
            print('nop')
    python_processes = []
    for process in psutil.process_iter():
        try:
            if 'python' in process.name() and process.cmdline():
                python_processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return python_processes

def get_per_process_cpu_info() -> List[Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    processes = get_processes_running_python_tests()
    per_process_info = []
    for p in processes:
        info = {'pid': p.pid, 'cmd': ' '.join(p.cmdline()), 'cpu_percent': p.cpu_percent(), 'rss_memory': p.memory_info().rss}
        try:
            memory_full_info = p.memory_full_info()
            info['uss_memory'] = memory_full_info.uss
            if 'pss' in memory_full_info:
                info['pss_memory'] = memory_full_info.pss
        except psutil.AccessDenied as e:
            pass
        per_process_info.append(info)
    return per_process_info

def get_per_process_gpu_info(handle: Any) -> List[Dict[str, Any]]:
    if False:
        while True:
            i = 10
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    per_process_info = []
    for p in processes:
        info = {'pid': p.pid, 'gpu_memory': p.usedGpuMemory}
        per_process_info.append(info)
    return per_process_info

def rocm_ret_ok(ret: int) -> Any:
    if False:
        for i in range(10):
            print('nop')
    return ret == rsmi_status_t.RSMI_STATUS_SUCCESS

def rocm_list_devices() -> List[int]:
    if False:
        while True:
            i = 10
    num = c_uint32(0)
    ret = rocmsmi.rsmi_num_monitor_devices(byref(num))
    if rocm_ret_ok(ret):
        return list(range(num.value))
    return []

def rocm_get_mem_use(device: int) -> float:
    if False:
        i = 10
        return i + 15
    memoryUse = c_uint64()
    memoryTot = c_uint64()
    ret = rocmsmi.rsmi_dev_memory_usage_get(device, 0, byref(memoryUse))
    if rocm_ret_ok(ret):
        ret = rocmsmi.rsmi_dev_memory_total_get(device, 0, byref(memoryTot))
        if rocm_ret_ok(ret):
            return float(memoryUse.value) / float(memoryTot.value)
    return 0.0

def rocm_get_gpu_use(device: int) -> float:
    if False:
        i = 10
        return i + 15
    percent = c_uint32()
    ret = rocmsmi.rsmi_dev_busy_percent_get(device, byref(percent))
    if rocm_ret_ok(ret):
        return float(percent.value)
    return 0.0

def rocm_get_pid_list() -> List[Any]:
    if False:
        for i in range(10):
            print('nop')
    num_items = c_uint32()
    ret = rocmsmi.rsmi_compute_process_info_get(None, byref(num_items))
    if rocm_ret_ok(ret):
        buff_sz = num_items.value + 10
        procs = (rsmi_process_info_t * buff_sz)()
        procList = []
        ret = rocmsmi.rsmi_compute_process_info_get(byref(procs), byref(num_items))
        for i in range(num_items.value):
            procList.append(procs[i].process_id)
        return procList
    return []

def rocm_get_per_process_gpu_info() -> List[Dict[str, Any]]:
    if False:
        return 10
    per_process_info = []
    for pid in rocm_get_pid_list():
        proc = rsmi_process_info_t()
        ret = rocmsmi.rsmi_compute_process_info_by_pid_get(int(pid), byref(proc))
        if rocm_ret_ok(ret):
            info = {'pid': pid, 'gpu_memory': proc.vram_usage}
            per_process_info.append(info)
    return per_process_info
if __name__ == '__main__':
    handle = None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        pass
    rsmi_handles = []
    try:
        ret = rocmsmi.rsmi_init(0)
        rsmi_handles = rocm_list_devices()
    except Exception:
        pass
    kill_now = False

    def exit_gracefully(*args: Any) -> None:
        if False:
            i = 10
            return i + 15
        global kill_now
        kill_now = True
    signal.signal(signal.SIGTERM, exit_gracefully)
    while not kill_now:
        try:
            stats = {'time': datetime.datetime.utcnow().isoformat('T') + 'Z', 'total_cpu_percent': psutil.cpu_percent(), 'per_process_cpu_info': get_per_process_cpu_info()}
            if handle is not None:
                stats['per_process_gpu_info'] = get_per_process_gpu_info(handle)
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats['total_gpu_utilization'] = gpu_utilization.gpu
                stats['total_gpu_mem_utilization'] = gpu_utilization.memory
            if rsmi_handles:
                stats['per_process_gpu_info'] = rocm_get_per_process_gpu_info()
                gpu_utilization = 0.0
                gpu_memory = 0.0
                for dev in rsmi_handles:
                    gpu_utilization += rocm_get_gpu_use(dev)
                    gpu_memory += rocm_get_mem_use(dev)
                stats['total_gpu_utilization'] = gpu_utilization
                stats['total_gpu_mem_utilization'] = gpu_memory
        except Exception as e:
            stats = {'time': datetime.datetime.utcnow().isoformat('T') + 'Z', 'error': str(e)}
        finally:
            print(json.dumps(stats))
            time.sleep(1)