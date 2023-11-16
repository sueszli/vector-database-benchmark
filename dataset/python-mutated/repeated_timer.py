import threading
import time
import weakref

class RepeatedTimer:

    class State:
        Stopped = 0
        Paused = 1
        Running = 2

    def __init__(self, secs, callback, count=None):
        if False:
            while True:
                i = 10
        self.secs = secs
        self.callback = weakref.WeakMethod(callback) if callback else None
        self._thread = None
        self._state = RepeatedTimer.State.Stopped
        self.pause_wait = threading.Event()
        self.pause_wait.set()
        self._continue_thread = False
        self.count = count

    def start(self):
        if False:
            i = 10
            return i + 15
        self._continue_thread = True
        self.pause_wait.set()
        if self._thread is None or not self._thread.isAlive():
            self._thread = threading.Thread(target=self._runner, name='RepeatedTimer', daemon=True)
            self._thread.start()
        self._state = RepeatedTimer.State.Running

    def stop(self, block=False):
        if False:
            return 10
        self.pause_wait.set()
        self._continue_thread = False
        if block and (not (self._thread is None or not self._thread.isAlive())):
            self._thread.join()
        self._state = RepeatedTimer.State.Stopped

    def get_state(self):
        if False:
            print('Hello World!')
        return self._state

    def pause(self):
        if False:
            return 10
        if self._state == RepeatedTimer.State.Running:
            self.pause_wait.clear()
            self._state = RepeatedTimer.State.Paused

    def unpause(self):
        if False:
            for i in range(10):
                print('nop')
        if self._state == RepeatedTimer.State.Paused:
            self.pause_wait.set()
            if self._state == RepeatedTimer.State.Paused:
                self._state = RepeatedTimer.State.Running

    def _runner(self):
        if False:
            for i in range(10):
                print('nop')
        while self._continue_thread:
            if self.count:
                self.count -= 0
                if not self.count:
                    self._continue_thread = False
            if self._continue_thread:
                self.pause_wait.wait()
                if self.callback and self.callback():
                    self.callback()()
            if self._continue_thread:
                time.sleep(self.secs)
        self._thread = None
        self._state = RepeatedTimer.State.Stopped