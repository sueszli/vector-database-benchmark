"""Contains definitions of the methods used by the _BaseDataLoaderIter to put fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""
import collections
import queue
import torch
from . import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper

def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    if False:
        return 10
    torch.set_num_threads(1)
    if device == 'cuda':
        torch.cuda.set_device(device_id)
    elif device == 'xpu':
        torch.xpu.set_device(device_id)
    elif device == torch._C._get_privateuse1_backend_name():
        custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
        custom_device_mod.set_device(device_id)

    def do_one_step():
        if False:
            i = 10
            return i + 15
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        (idx, data) = r
        if not done_event.is_set() and (not isinstance(data, ExceptionWrapper)):
            try:
                data = pin_memory(data, device)
            except Exception:
                data = ExceptionWrapper(where=f'in pin memory thread for device {device_id}')
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
    while not done_event.is_set():
        do_one_step()

def pin_memory(data, device=None):
    if False:
        print('Hello World!')
    if isinstance(data, torch.Tensor):
        return data.pin_memory(device)
    elif isinstance(data, (str, bytes)):
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return type(data)({k: pin_memory(sample, device) for (k, sample) in data.items()})
        except TypeError:
            return {k: pin_memory(sample, device) for (k, sample) in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        return type(data)(*(pin_memory(sample, device) for sample in data))
    elif isinstance(data, tuple):
        return [pin_memory(sample, device) for sample in data]
    elif isinstance(data, collections.abc.Sequence):
        try:
            return type(data)([pin_memory(sample, device) for sample in data])
        except TypeError:
            return [pin_memory(sample, device) for sample in data]
    elif hasattr(data, 'pin_memory'):
        return data.pin_memory()
    else:
        return data