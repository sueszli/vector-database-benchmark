import logging
from typing import Callable, Generic, List
from typing_extensions import ParamSpec
logger = logging.getLogger(__name__)
P = ParamSpec('P')

class CallbackRegistry(Generic[P]):

    def __init__(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.callback_list: List[Callable[P, None]] = []

    def add_callback(self, cb: Callable[P, None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.callback_list.append(cb)

    def fire_callbacks(self, *args: P.args, **kwargs: P.kwargs) -> None:
        if False:
            return 10
        for cb in self.callback_list:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.exception('Exception in callback for %s registered with CUDA trace', self.name)
CUDAEventCreationCallbacks: 'CallbackRegistry[int]' = CallbackRegistry('CUDA event creation')
CUDAEventDeletionCallbacks: 'CallbackRegistry[int]' = CallbackRegistry('CUDA event deletion')
CUDAEventRecordCallbacks: 'CallbackRegistry[int, int]' = CallbackRegistry('CUDA event record')
CUDAEventWaitCallbacks: 'CallbackRegistry[int, int]' = CallbackRegistry('CUDA event wait')
CUDAMemoryAllocationCallbacks: 'CallbackRegistry[int]' = CallbackRegistry('CUDA memory allocation')
CUDAMemoryDeallocationCallbacks: 'CallbackRegistry[int]' = CallbackRegistry('CUDA memory deallocation')
CUDAStreamCreationCallbacks: 'CallbackRegistry[int]' = CallbackRegistry('CUDA stream creation')
CUDADeviceSynchronizationCallbacks: 'CallbackRegistry[[]]' = CallbackRegistry('CUDA device synchronization')
CUDAStreamSynchronizationCallbacks: 'CallbackRegistry[int]' = CallbackRegistry('CUDA stream synchronization')
CUDAEventSynchronizationCallbacks: 'CallbackRegistry[int]' = CallbackRegistry('CUDA event synchronization')

def register_callback_for_cuda_event_creation(cb: Callable[[int], None]) -> None:
    if False:
        i = 10
        return i + 15
    CUDAEventCreationCallbacks.add_callback(cb)

def register_callback_for_cuda_event_deletion(cb: Callable[[int], None]) -> None:
    if False:
        return 10
    CUDAEventDeletionCallbacks.add_callback(cb)

def register_callback_for_cuda_event_record(cb: Callable[[int, int], None]) -> None:
    if False:
        for i in range(10):
            print('nop')
    CUDAEventRecordCallbacks.add_callback(cb)

def register_callback_for_cuda_event_wait(cb: Callable[[int, int], None]) -> None:
    if False:
        while True:
            i = 10
    CUDAEventWaitCallbacks.add_callback(cb)

def register_callback_for_cuda_memory_allocation(cb: Callable[[int], None]) -> None:
    if False:
        i = 10
        return i + 15
    CUDAMemoryAllocationCallbacks.add_callback(cb)

def register_callback_for_cuda_memory_deallocation(cb: Callable[[int], None]) -> None:
    if False:
        for i in range(10):
            print('nop')
    CUDAMemoryDeallocationCallbacks.add_callback(cb)

def register_callback_for_cuda_stream_creation(cb: Callable[[int], None]) -> None:
    if False:
        for i in range(10):
            print('nop')
    CUDAStreamCreationCallbacks.add_callback(cb)

def register_callback_for_cuda_device_synchronization(cb: Callable[[], None]) -> None:
    if False:
        print('Hello World!')
    CUDADeviceSynchronizationCallbacks.add_callback(cb)

def register_callback_for_cuda_stream_synchronization(cb: Callable[[int], None]) -> None:
    if False:
        print('Hello World!')
    CUDAStreamSynchronizationCallbacks.add_callback(cb)

def register_callback_for_cuda_event_synchronization(cb: Callable[[int], None]) -> None:
    if False:
        for i in range(10):
            print('nop')
    CUDAEventSynchronizationCallbacks.add_callback(cb)