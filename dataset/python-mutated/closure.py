import os
import threading
from queue import Empty as EmptyQueue, Queue
from torch._lazy.device_context import get_device_context

class ClosureHandler:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def run(self, closure):
        if False:
            i = 10
            return i + 15
        'Run closure function\n\n        Args:\n        closure: callable function to run\n        '
        closure()

    def __call__(self, closures):
        if False:
            return 10
        for closure in closures:
            self.run(closure)

class AsyncClosureHandler(ClosureHandler):
    """Handler for Asynchronous Step Closures
    Args:
        max_queue_size: The maximum length of the closure queue after which
        the training loop will block until closures are evaluated.
        By default, a reasonable limit of a maximum of 100 on the queue.
        This value can be set using the `XLA_MAX_ASYNC_QUEUE` environment
        variable.
    """

    def __init__(self, max_queue_size=100):
        if False:
            while True:
                i = 10
        super().__init__()
        self._closure_queue: Queue = Queue(int(os.environ.get('LTC_MAX_ASYNC_QUEUE', max_queue_size)))
        self._closure_exception: Queue = Queue()
        self._closure_lock = threading.Lock()
        self._closure_event_loop_finished = threading.Event()
        self._closure_event_loop = None

    def start_event_loop(self):
        if False:
            return 10
        'Start closure event loop if not started'
        if self._closure_event_loop is None:

            def event_loop():
                if False:
                    print('Hello World!')
                while True:
                    try:
                        closure = self._closure_queue.get(block=True, timeout=3)
                        closure()
                        self._closure_queue.task_done()
                    except EmptyQueue:
                        with self._closure_lock:
                            if self._closure_queue.empty():
                                self._closure_event_loop_finished.set()
                                return
                    except Exception as e:
                        self._closure_exception.put(e)
                        return
            self._closure_event_loop = threading.Thread(target=event_loop)
            self._closure_event_loop.start()

    def run(self, closure):
        if False:
            return 10
        with self._closure_lock:
            self._closure_queue.put(closure, block=True)
            if self._closure_event_loop is None or not self._closure_event_loop.is_alive():
                try:
                    e = self._closure_exception.get(block=False)
                    raise RuntimeError('Cannot run asynchronous closure due to previously raised exception') from e
                except EmptyQueue:
                    self._closure_event_loop = None
                    self.start_event_loop()

def add_step_closure(closure, args=(), run_async=False):
    if False:
        for i in range(10):
            print('nop')
    'Adds a closure to the list of the ones to be run at the end of the step.\n    Many times during model training there is the need to print/report (print to\n    console, post to tensorboard, etc...) information which require the content of\n    intermediary tensors to be inspected.\n    Inspecting different tensors content in different points of the model code\n    requires many executions and typically causes performance issues.\n    Adding a step closure will ensure that it will be run after the barrier, when\n    all the live tensors will be already materialized to device data.\n    Live tensors which will include the ones captured by the closure arguments.\n    So using `add_step_closure()` will ensure a single execution will be\n    performed, even when multiple closures are queued, requiring multiple tensors\n    to be inspected.\n    Step closures will be run sequentially in the order they have been queued.\n    Note that even though using this API the execution will be optimized, it is\n    advised to throttle the printing/reporting events once every N steps.\n    Args:\n      closure (callable): The function to be called.\n      args (tuple): The arguments to be passed to the closure.\n      run_async: If True, run the closure asynchronously.\n    '
    devctx = get_device_context()
    closures_type = 'async_step_closures' if run_async else 'step_closures'
    step_closures = getattr(devctx, closures_type, None)
    if step_closures is None:
        step_closures = []
        setattr(devctx, closures_type, step_closures)
    step_closures.append(lambda a=args: closure(*a))

def run_step_closures():
    if False:
        return 10
    devctx = get_device_context()
    async_step_closures = getattr(devctx, 'async_step_closures', None)
    if async_step_closures is not None:
        devctx.async_step_closures = []
        async_closure_handler = getattr(devctx, 'async_closure_handler', None)
        if async_closure_handler is None:
            async_closure_handler = AsyncClosureHandler()
            devctx.async_closure_handler = async_closure_handler
        async_closure_handler(async_step_closures)
    step_closures = getattr(devctx, 'step_closures', None)
    if step_closures is not None:
        devctx.step_closures = []
        closure_handler = getattr(devctx, 'closure_handler', None)
        if closure_handler is None:
            closure_handler = ClosureHandler()
            devctx.closure_handler = closure_handler
        closure_handler(step_closures)
    return devctx