import logging
import threading
import cupy
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.const import ENV
NCCL_STREAM_POOL_SIZE = 32
MAX_GPU_PER_ACTOR = 16
logger = logging.getLogger(__name__)

class StreamPool:
    """The class that represents a stream pool associated with a GPU.

    When multistream is enabled, we will allocate a pool of streams for each
    GPU, and get available stream from this pool when a collective kernel is
    initialized. This enables overlapping computation/communication kernels
    using multiple CUDA streams, given that the streams a appropriately
    synchronized. The class is thread-safe.


    Args:
        device_idx: the absolute index of the device for this pool.
    """

    def __init__(self, device_idx):
        if False:
            for i in range(10):
                print('nop')
        self.device_idx = device_idx
        self._initialized = False
        self._initialized_lock = threading.Lock()
        self._pool = [None] * NCCL_STREAM_POOL_SIZE
        self._counter = 0
        self._pool_lock = threading.Lock()

    def get_stream(self):
        if False:
            print('Hello World!')
        'Get an available stream from the pool.\n\n        The function locks the stream pool and releases the lock before\n        returning.\n\n        Returns:\n            stream (cupy.cuda.Stream): the returned stream from pool.\n        '
        self._initialized_lock.acquire()
        if not self._initialized:
            self._init_once()
        self._initialized_lock.release()
        self._pool_lock.acquire()
        stream = self._pool[self._counter]
        self._counter = (self._counter + 1) % NCCL_STREAM_POOL_SIZE
        self._pool_lock.release()
        return stream

    def _init_once(self):
        if False:
            i = 10
            return i + 15
        'Initialize the stream pool only for once.'
        with nccl_util.Device(self.device_idx):
            for i in range(NCCL_STREAM_POOL_SIZE):
                if ENV.NCCL_USE_MULTISTREAM.val:
                    logger.debug('NCCL multistream enabled.')
                    self._pool[i] = cupy.cuda.Stream(null=False, non_blocking=False)
                else:
                    logger.debug('NCCL multistream disabled.')
                    self._pool[i] = cupy.cuda.Stream.null
        self._init_flag = True
_device_stream_pool_map = dict()

def _init_stream_pool():
    if False:
        while True:
            i = 10
    global _device_stream_pool_map
    for i in range(MAX_GPU_PER_ACTOR):
        _device_stream_pool_map[i] = StreamPool(i)

def get_stream_pool(device_idx):
    if False:
        while True:
            i = 10
    'Get the CUDA stream pool of a GPU device.'
    lock = threading.Lock()
    lock.acquire()
    if not _device_stream_pool_map:
        _init_stream_pool()
    lock.release()
    return _device_stream_pool_map[device_idx]