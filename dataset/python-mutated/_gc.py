import gc
import logging
import platform
import time
from typing import Iterable
from prometheus_client.core import REGISTRY, CounterMetricFamily, Gauge, GaugeMetricFamily, Histogram, Metric
from twisted.internet import task
from synapse.metrics._types import Collector
'Prometheus metrics for garbage collection'
logger = logging.getLogger(__name__)
MIN_TIME_BETWEEN_GCS = (1.0, 10.0, 30.0)
running_on_pypy = platform.python_implementation() == 'PyPy'
gc_unreachable = Gauge('python_gc_unreachable_total', 'Unreachable GC objects', ['gen'])
gc_time = Histogram('python_gc_time', 'Time taken to GC (sec)', ['gen'], buckets=[0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 15.0, 30.0, 45.0, 60.0])

class GCCounts(Collector):

    def collect(self) -> Iterable[Metric]:
        if False:
            i = 10
            return i + 15
        cm = GaugeMetricFamily('python_gc_counts', 'GC object counts', labels=['gen'])
        for (n, m) in enumerate(gc.get_count()):
            cm.add_metric([str(n)], m)
        yield cm

def install_gc_manager() -> None:
    if False:
        i = 10
        return i + 15
    'Disable automatic GC, and replace it with a task that runs every 100ms\n\n    This means that (a) we can limit how often GC runs; (b) we can get some metrics\n    about GC activity.\n\n    It does nothing on PyPy.\n    '
    if running_on_pypy:
        return
    REGISTRY.register(GCCounts())
    gc.disable()
    _last_gc = [0.0, 0.0, 0.0]

    def _maybe_gc() -> None:
        if False:
            for i in range(10):
                print('nop')
        threshold = gc.get_threshold()
        counts = gc.get_count()
        end = time.time()
        for i in (2, 1, 0):
            if threshold[i] < counts[i] and MIN_TIME_BETWEEN_GCS[i] < end - _last_gc[i]:
                if i == 0:
                    logger.debug('Collecting gc %d', i)
                else:
                    logger.info('Collecting gc %d', i)
                start = time.time()
                unreachable = gc.collect(i)
                end = time.time()
                _last_gc[i] = end
                gc_time.labels(i).observe(end - start)
                gc_unreachable.labels(i).set(unreachable)
    gc_task = task.LoopingCall(_maybe_gc)
    gc_task.start(0.1)

class PyPyGCStats(Collector):

    def collect(self) -> Iterable[Metric]:
        if False:
            i = 10
            return i + 15
        stats = gc.get_stats(memory_pressure=False)
        s = stats._s
        pypy_gc_time = CounterMetricFamily('pypy_gc_time_seconds_total', 'Total time spent in PyPy GC', labels=[])
        pypy_gc_time.add_metric([], s.total_gc_time / 1000)
        yield pypy_gc_time
        pypy_mem = GaugeMetricFamily('pypy_memory_bytes', 'Memory tracked by PyPy allocator', labels=['state', 'class', 'kind'])
        pypy_mem.add_metric(['used', '', 'jit'], s.jit_backend_used)
        pypy_mem.add_metric(['allocated', '', 'jit'], s.jit_backend_allocated)
        pypy_mem.add_metric(['used', '', 'arenas'], s.total_arena_memory)
        pypy_mem.add_metric(['allocated', '', 'arenas'], s.peak_arena_memory)
        pypy_mem.add_metric(['used', '', 'rawmalloced'], s.total_rawmalloced_memory)
        pypy_mem.add_metric(['allocated', '', 'rawmalloced'], s.peak_rawmalloced_memory)
        pypy_mem.add_metric(['used', '', 'nursery'], s.nursery_size)
        pypy_mem.add_metric(['allocated', '', 'nursery'], s.nursery_size)
        pypy_mem.add_metric(['used', 'totals', 'gc'], s.total_gc_memory)
        pypy_mem.add_metric(['allocated', 'totals', 'gc'], s.total_allocated_memory)
        pypy_mem.add_metric(['used', 'totals', 'gc_peak'], s.peak_memory)
        pypy_mem.add_metric(['allocated', 'totals', 'gc_peak'], s.peak_allocated_memory)
        yield pypy_mem
if running_on_pypy:
    REGISTRY.register(PyPyGCStats())