import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
get_cuda_stream: Optional[Callable[[int], int]]
if torch.cuda._is_compiled():
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
else:
    get_cuda_stream = None
_device_t = Union[torch.device, str, int, None]
caching_worker_device_properties: Dict[str, Any] = {}
caching_worker_current_devices: Dict[str, int] = {}

class DeviceInterfaceMeta(type):

    def __new__(metacls, *args, **kwargs):
        if False:
            return 10
        class_member = args[2]
        if 'Event' in class_member:
            assert inspect.isclass(class_member['Event']) and issubclass(class_member['Event'], _EventBase), 'DeviceInterface member Event should be inherit from _EventBase'
        if 'Stream' in class_member:
            assert inspect.isclass(class_member['Stream']) and issubclass(class_member['Stream'], _StreamBase), 'DeviceInterface member Stream should be inherit from _StreamBase'
        return super().__new__(metacls, *args, **kwargs)

class DeviceInterface(metaclass=DeviceInterfaceMeta):
    """
    This is a simple device runtime interface for Inductor. It enables custom
    backends to be integrated with Inductor in a device-agnostic semantic.
    """

    class device:

        def __new__(cls, device: _device_t):
            if False:
                i = 10
                return i + 15
            raise NotImplementedError()

    class Worker:
        """
        Worker API to query device properties that will work in multi processing
        workers that cannot use the GPU APIs (due to processing fork() and
        initialization time issues). Properties are recorded in the main process
        before we fork the workers.
        """

        @staticmethod
        def set_device(device: int):
            if False:
                i = 10
                return i + 15
            raise NotImplementedError()

        @staticmethod
        def current_device() -> int:
            if False:
                return 10
            raise NotImplementedError()

        @staticmethod
        def get_device_properties(device: _device_t=None):
            if False:
                return 10
            raise NotImplementedError()

    @staticmethod
    def current_device():
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @staticmethod
    def set_device(device: _device_t):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @staticmethod
    def device_count():
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @staticmethod
    def is_available() -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @staticmethod
    def stream(stream: torch.Stream):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @staticmethod
    def current_stream():
        if False:
            return 10
        raise NotImplementedError()

    @staticmethod
    def set_stream(stream: torch.Stream):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @staticmethod
    def get_raw_stream():
        if False:
            return 10
        raise NotImplementedError()

    @staticmethod
    def synchronize(device: _device_t=None):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @staticmethod
    def get_device_properties(device: _device_t=None):
        if False:
            return 10
        raise NotImplementedError()

    @staticmethod
    def get_compute_capability(device: _device_t=None):
        if False:
            print('Hello World!')
        raise NotImplementedError()

class CudaInterface(DeviceInterface):
    device = torch.cuda.device
    Event = torch.cuda.Event
    Stream = torch.cuda.Stream

    class Worker:

        @staticmethod
        def set_device(device: int):
            if False:
                return 10
            caching_worker_current_devices['cuda'] = device

        @staticmethod
        def current_device() -> int:
            if False:
                print('Hello World!')
            if 'cuda' in caching_worker_current_devices:
                return caching_worker_current_devices['cuda']
            return torch.cuda.current_device()

        @staticmethod
        def get_device_properties(device: _device_t=None):
            if False:
                return 10
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == 'cuda'
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = CudaInterface.Worker.current_device()
            if 'cuda' not in caching_worker_device_properties:
                device_prop = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
                caching_worker_device_properties['cuda'] = device_prop
            return caching_worker_device_properties['cuda'][device]
    current_device = staticmethod(torch.cuda.current_device)
    set_device = staticmethod(torch.cuda.set_device)
    device_count = staticmethod(torch.cuda.device_count)
    stream = staticmethod(torch.cuda.stream)
    current_stream = staticmethod(torch.cuda.current_stream)
    set_stream = staticmethod(torch.cuda.set_stream)
    _set_stream_by_id = staticmethod(torch.cuda._set_stream_by_id)
    synchronize = staticmethod(torch.cuda.synchronize)
    get_device_properties = staticmethod(torch.cuda.get_device_properties)
    get_raw_stream = staticmethod(get_cuda_stream)

    @staticmethod
    def is_available() -> bool:
        if False:
            i = 10
            return i + 15
        return torch.cuda.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t=None):
        if False:
            for i in range(10):
                print('nop')
        (major, min) = torch.cuda.get_device_capability(device)
        return major * 10 + min
device_interfaces: Dict[str, Type[DeviceInterface]] = {}

def register_interface_for_device(device: str, device_interface: Type[DeviceInterface]):
    if False:
        return 10
    device_interfaces[device] = device_interface

def get_interface_for_device(device: str) -> Type[DeviceInterface]:
    if False:
        for i in range(10):
            print('nop')
    if device in device_interfaces:
        return device_interfaces[device]
    raise NotImplementedError(f'No interface for device {device}')

def get_registered_device_interfaces() -> Iterable[Tuple[str, Type[DeviceInterface]]]:
    if False:
        for i in range(10):
            print('nop')
    return device_interfaces.items()
register_interface_for_device('cuda', CudaInterface)