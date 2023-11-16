import math as _math
import time as _time
import numpy as _numpy
import cupy as _cupy
from cupy_backends.cuda.api import runtime

class _PerfCaseResult:
    """ An obscure object encompassing timing results recorded by
    :func:`~cupyx.profiler.benchmark`. Simple statistics can be obtained by
    converting an instance of this class to a string.

    .. warning::
        This API is currently experimental and subject to change in future
        releases.

    """

    def __init__(self, name, ts, devices):
        if False:
            while True:
                i = 10
        assert ts.ndim == 2
        assert ts.shape[0] == len(devices) + 1
        assert ts.shape[1] > 0
        self.name = name
        self._ts = ts
        self._devices = devices

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        ' Returns a string representation of the object.\n\n        Returns:\n            str: A string representation of the object.\n        '
        return self.to_str(show_gpu=True)

    @property
    def cpu_times(self) -> _numpy.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'A :class:`numpy.ndarray` of shape ``(n_repeat,)``, holding times spent\n        on CPU in seconds.\n\n        These values are delta of the host-side performance counter\n        (:func:`time.perf_counter`) between each repeat step.\n        '
        return self._ts[0]

    @property
    def gpu_times(self) -> _numpy.ndarray:
        if False:
            return 10
        'A :class:`numpy.ndarray` of shape ``(len(devices), n_repeat)``,\n        holding times spent on GPU in seconds.\n\n        These values are measured using ``cudaEventElapsedTime`` with events\n        recoreded before/after each repeat step.\n        '
        return self._ts[1:]

    @staticmethod
    def _to_str_per_item(device_name, t):
        if False:
            for i in range(10):
                print('nop')
        assert t.ndim == 1
        assert t.size > 0
        t_us = t * 1000000.0
        s = '    {}: {:9.03f} us'.format(device_name, t_us.mean())
        if t.size > 1:
            s += '   +/- {:6.03f} (min: {:9.03f} / max: {:9.03f}) us'.format(t_us.std(), t_us.min(), t_us.max())
        return s

    def to_str(self, show_gpu=False):
        if False:
            print('Hello World!')
        results = [self._to_str_per_item('CPU', self._ts[0])]
        if show_gpu:
            for (i, d) in enumerate(self._devices):
                results.append(self._to_str_per_item('GPU-{}'.format(d), self._ts[1 + i]))
        return '{:<20s}:{}'.format(self.name, ' '.join(results))

    def __str__(self):
        if False:
            return 10
        return self.to_str(show_gpu=True)

def benchmark(func, args=(), kwargs={}, n_repeat=10000, *, name=None, n_warmup=10, max_duration=_math.inf, devices=None):
    if False:
        return 10
    " Timing utility for measuring time spent by both CPU and GPU.\n\n    This function is a very convenient helper for setting up a timing test. The\n    GPU time is properly recorded by synchronizing internal streams. As a\n    result, to time a multi-GPU function all participating devices must be\n    passed as the ``devices`` argument so that this helper knows which devices\n    to record. A simple example is given as follows:\n\n    .. code-block:: py\n\n        import cupy as cp\n        from cupyx.profiler import benchmark\n\n        def f(a, b):\n            return 3 * cp.sin(-a) * b\n\n        a = 0.5 - cp.random.random((100,))\n        b = cp.random.random((100,))\n        print(benchmark(f, (a, b), n_repeat=1000))\n\n\n    Args:\n        func (callable): a callable object to be timed.\n        args (tuple): positional argumens to be passed to the callable.\n        kwargs (dict): keyword arguments to be passed to the callable.\n        n_repeat (int): number of times the callable is called. Increasing\n            this value would improve the collected statistics at the cost\n            of longer test time.\n        name (str): the function name to be reported. If not given, the\n            callable's ``__name__`` attribute is used.\n        n_warmup (int): number of times the callable is called. The warm-up\n            runs are not timed.\n        max_duration (float): the maximum time (in seconds) that the entire\n            test can use. If the taken time is longer than this limit, the test\n            is stopped and the statistics collected up to the breakpoint is\n            reported.\n        devices (tuple): a tuple of device IDs (int) that will be timed during\n            the timing test. If not given, the current device is used.\n\n    Returns:\n        :class:`~cupyx.profiler._time._PerfCaseResult`:\n            an object collecting all test results.\n\n    "
    if name is None:
        name = func.__name__
    if devices is None:
        devices = (_cupy.cuda.get_device_id(),)
    if not callable(func):
        raise ValueError('`func` should be a callable object.')
    if not isinstance(args, tuple):
        raise ValueError('`args` should be of tuple type.')
    if not isinstance(kwargs, dict):
        raise ValueError('`kwargs` should be of dict type.')
    if not isinstance(n_repeat, int):
        raise ValueError('`n_repeat` should be an integer.')
    if not isinstance(name, str):
        raise ValueError('`name` should be a string.')
    if not isinstance(n_warmup, int):
        raise ValueError('`n_warmup` should be an integer.')
    if not _numpy.isreal(max_duration):
        raise ValueError('`max_duration` should be given in seconds')
    if not isinstance(devices, tuple):
        raise ValueError('`devices` should be of tuple type')
    return _repeat(func, args, kwargs, n_repeat, name, n_warmup, max_duration, devices)

def _repeat(func, args, kwargs, n_repeat, name, n_warmup, max_duration, devices):
    if False:
        return 10
    events_1 = []
    events_2 = []
    for i in devices:
        prev_device = runtime.getDevice()
        try:
            runtime.setDevice(i)
            events_1.append(_cupy.cuda.stream.Event())
            events_2.append(_cupy.cuda.stream.Event())
        finally:
            runtime.setDevice(prev_device)
    for i in range(n_warmup):
        func(*args, **kwargs)
    for (event, device) in zip(events_1, devices):
        prev_device = runtime.getDevice()
        try:
            runtime.setDevice(device)
            event.record()
        finally:
            runtime.setDevice(prev_device)
        event.synchronize()
    cpu_times = []
    gpu_times = [[] for i in events_1]
    duration = 0
    for i in range(n_repeat):
        for (event, device) in zip(events_1, devices):
            prev_device = runtime.getDevice()
            try:
                runtime.setDevice(device)
                event.record()
            finally:
                runtime.setDevice(prev_device)
        t1 = _time.perf_counter()
        func(*args, **kwargs)
        t2 = _time.perf_counter()
        cpu_time = t2 - t1
        cpu_times.append(cpu_time)
        for (event, device) in zip(events_2, devices):
            prev_device = runtime.getDevice()
            try:
                runtime.setDevice(device)
                event.record()
            finally:
                runtime.setDevice(prev_device)
        for (event, device) in zip(events_2, devices):
            prev_device = runtime.getDevice()
            try:
                runtime.setDevice(device)
                event.synchronize()
            finally:
                runtime.setDevice(prev_device)
        for (i, (ev1, ev2)) in enumerate(zip(events_1, events_2)):
            gpu_time = _cupy.cuda.get_elapsed_time(ev1, ev2) * 0.001
            gpu_times[i].append(gpu_time)
        duration += _time.perf_counter() - t1
        if duration > max_duration:
            break
    ts = _numpy.asarray([cpu_times] + gpu_times, dtype=_numpy.float64)
    return _PerfCaseResult(name, ts, devices=devices)