import logging
import psutil
import subprocess
import time
from multiprocessing import Process
from threading import Thread, Lock
import psutil
logger = logging.getLogger(__name__)

class ProcessMonitor(Thread):

    def __init__(self, *child_processes, **params):
        if False:
            for i in range(10):
                print('nop')
        super(ProcessMonitor, self).__init__(target=self._start)
        self._child_processes = []
        self._callbacks = params.pop('callbacks', [])
        self._lock = Lock()
        self.daemon = True
        self.working = False
        self.add_child_processes(*child_processes)

    def _start(self):
        if False:
            return 10
        self.working = True
        while self.working:
            for i in reversed(range(len(self._child_processes))):
                process = self._child_processes[i]
                if not self.is_process_alive(process):
                    logger.info('Subprocess %d exited with code %d', process.pid, self.exit_code(process))
                    if self.working:
                        self.run_callbacks(process)
                    self._child_processes.pop(i)
            time.sleep(0.5)

    def stop(self, *_):
        if False:
            for i in range(10):
                print('nop')
        self.working = False

    def exit(self, *_):
        if False:
            for i in range(10):
                print('nop')
        self.stop()
        self.kill_processes()

    def add_child_processes(self, *processes):
        if False:
            i = 10
            return i + 15
        assert all([self.is_supported(p) for p in processes])
        self._child_processes.extend(processes)

    def add_callbacks(self, *callbacks):
        if False:
            return 10
        self._callbacks.extend(callbacks)

    def remove_callbacks(self, *callbacks):
        if False:
            print('Hello World!')
        for handler in callbacks:
            idx = self._callbacks.index(handler)
            if idx != -1:
                self._callbacks.pop(idx)

    def run_callbacks(self, process=None):
        if False:
            while True:
                i = 10
        for callback in self._callbacks:
            if self.working:
                callback(process)

    def kill_processes(self, *_):
        if False:
            print('Hello World!')
        for process in self._child_processes:
            self.kill_process(process)

    @classmethod
    def kill_process(cls, process):
        if False:
            while True:
                i = 10
        if cls.is_process_alive(process):
            process_info = psutil.Process(process.pid)
            children = process_info.children(recursive=True)
            try:
                for c in children:
                    c.terminate()
                    if c.wait(timeout=60) is not None:
                        c.kill()
                process.terminate()
                if isinstance(process, (psutil.Popen, subprocess.Popen)):
                    process.communicate()
                elif isinstance(process, Process):
                    process.join()
            except Exception as exc:
                logger.error('Error terminating process %s: %r', process, exc)
            else:
                logger.warning('Subprocess %s terminated', cls._pid(process))

    @staticmethod
    def _pid(process):
        if False:
            i = 10
            return i + 15
        if process:
            return process.pid

    @staticmethod
    def is_supported(process):
        if False:
            i = 10
            return i + 15
        return isinstance(process, (psutil.Popen, subprocess.Popen, Process))

    @staticmethod
    def exit_code(process):
        if False:
            return 10
        if isinstance(process, (psutil.Popen, subprocess.Popen)):
            process.poll()
            return process.returncode
        elif isinstance(process, Process):
            return process.exitcode

    @staticmethod
    def is_process_alive(process):
        if False:
            while True:
                i = 10
        if isinstance(process, (psutil.Popen, subprocess.Popen)):
            process.poll()
            return process.returncode is None
        elif isinstance(process, Process):
            return process.is_alive()
        return False