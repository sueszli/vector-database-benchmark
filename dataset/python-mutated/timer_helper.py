import time
import paddle
_GLOBAL_TIMERS = None

def is_timer_initialized():
    if False:
        for i in range(10):
            print('nop')
    return _GLOBAL_TIMERS is not None

def _ensure_var_is_not_initialized(var, name):
    if False:
        i = 10
        return i + 15
    'Make sure the input variable is not None.'
    assert var is None, f'{name} has been already initialized.'

def _ensure_var_is_initialized(var, name):
    if False:
        while True:
            i = 10
    'Make sure the input variable is not None.'
    assert var is not None, f'{name} is not initialized.'

def get_timers():
    if False:
        for i in range(10):
            print('nop')
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS

def set_timers():
    if False:
        return 10
    'Initialize timers.'
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers()

class _Timer:
    """Timer."""

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        if False:
            i = 10
            return i + 15
        'Start the timer.'
        assert not self.started_, 'timer has already started'
        paddle.device.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        if False:
            while True:
                i = 10
        'Stop the timers.'
        assert self.started_, 'timer is not started.'
        paddle.device.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        'Reset timer.'
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        if False:
            i = 10
            return i + 15
        'Calculate the elapsed time.'
        started_ = self.started_
        if self.started_:
            self.stop()
        elapsed_ = self.elapsed_
        if reset:
            self.reset()
        if started_:
            self.start()
        return elapsed_

class Timers:
    """Group of timers."""

    def __init__(self):
        if False:
            return 10
        self.timers = {}

    def __call__(self, name):
        if False:
            print('Hello World!')
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        if False:
            return 10
        'Log a group of timers.'
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += f' | {name}: {elapsed_time:.2f}'
        print(string, flush=True)