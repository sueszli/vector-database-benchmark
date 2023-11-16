"""Run the GUI event loop in a non-main thread. This allows using the
GUI from plugins to other apps (e.g.: Jupyter or Tensorboard) where the GUI
cannot be started in the main thread. Currently does not work in macOS.

.. note:: This is a singleton class implemented with this module as a
   holder. The ``async_event_loop`` singleton is started whenever this
   module is imported.  If you are using remote visualization with WebRTC,
   you must call ``enable_webrtc()`` before importing this module.
"""
import threading
from collections import deque
import open3d as o3d

class _AsyncEventLoop:

    class _Task:
        _g_next_id = 0

        def __init__(self, func, *args, **kwargs):
            if False:
                while True:
                    i = 10
            self.task_id = self._g_next_id
            self.func = (func, args, kwargs)
            _AsyncEventLoop._Task._g_next_id += 1

    def __init__(self):
        if False:
            print('Hello World!')
        o3d.utility.reset_print_function()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._run_queue = deque()
        self._return_vals = {}
        self._started = False
        self._start()

    def _start(self):
        if False:
            return 10
        if not self._started:
            self._thread = threading.Thread(name='GUIMain', target=self._thread_main)
            self._thread.start()
            self._started = True

    def run_sync(self, func, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Enqueue task, wait for completion and return result. Can run in any\n        thread.\n        '
        if not self._started:
            raise RuntimeError('GUI thread has exited.')
        with self._lock:
            task = _AsyncEventLoop._Task(func, *args, **kwargs)
            self._run_queue.append(task)
        while True:
            with self._cv:
                self._cv.wait_for(lambda : task.task_id in self._return_vals)
            with self._lock:
                return self._return_vals.pop(task.task_id)

    def _thread_main(self):
        if False:
            for i in range(10):
                print('nop')
        'Main GUI thread event loop'
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        done = False
        while not done:
            while len(self._run_queue) > 0:
                with self._lock:
                    task = self._run_queue.popleft()
                (func, args, kwargs) = task.func
                retval = func(*args, **kwargs)
                with self._cv:
                    self._return_vals[task.task_id] = retval
                    self._cv.notify_all()
            done = not app.run_one_tick()
        self._started = False
async_event_loop = _AsyncEventLoop()