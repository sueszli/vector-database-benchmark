"""
========================================
Returning a generator in joblib.Parallel
========================================

This example illustrates memory optimization enabled by using
:class:`joblib.Parallel` to get a generator on the outputs of parallel jobs.
We first create tasks that return results with large memory footprints.
If we call :class:`~joblib.Parallel` for several of these tasks directly, we
observe a high memory usage, as all the results are held in RAM before being
processed

Using ``return_as='generator'`` allows to progressively consume the outputs
as they arrive and keeps the memory at an acceptable level.

In this case, the output of the `Parallel` call is a generator that yields the
results in the order the tasks have been submitted with. If the order of the
tasks does not matter (for instance if they are consumed by a commutative
aggregation function), then using ``return_as='generator_unordered'`` can be
even more efficient.

"""
import time
from psutil import Process
from threading import Thread

class MemoryMonitor(Thread):
    """Monitor the memory usage in MB in a separate thread.

    Note that this class is good enough to highlight the memory profile of
    Parallel in this example, but is not a general purpose profiler fit for
    all cases.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.stop = False
        self.memory_buffer = []
        self.start()

    def get_memory(self):
        if False:
            return 10
        'Get memory of a process and its children.'
        p = Process()
        memory = p.memory_info().rss
        for c in p.children():
            memory += c.memory_info().rss
        return memory

    def run(self):
        if False:
            print('Hello World!')
        memory_start = self.get_memory()
        while not self.stop:
            self.memory_buffer.append(self.get_memory() - memory_start)
            time.sleep(0.2)

    def join(self):
        if False:
            while True:
                i = 10
        self.stop = True
        super().join()
import numpy as np

def return_big_object(i):
    if False:
        return 10
    time.sleep(0.1)
    return i * np.ones((10000, 200), dtype=np.float64)

def accumulator_sum(generator):
    if False:
        while True:
            i = 10
    result = 0
    for value in generator:
        result += value
        print('.', end='', flush=True)
    print('')
    return result
from joblib import Parallel, delayed
monitor = MemoryMonitor()
print("Running tasks with return_as='list'...")
res = Parallel(n_jobs=2, return_as='list')((delayed(return_big_object)(i) for i in range(150)))
print('Accumulate results:', end='')
res = accumulator_sum(res)
print('All tasks completed and reduced successfully.')
del res
monitor.join()
peak = max(monitor.memory_buffer) / 1000000000.0
print(f'Peak memory usage: {peak:.2f}GB')
monitor_gen = MemoryMonitor()
print("Create result generator with return_as='generator'...")
res = Parallel(n_jobs=2, return_as='generator')((delayed(return_big_object)(i) for i in range(150)))
print('Accumulate results:', end='')
res = accumulator_sum(res)
print('All tasks completed and reduced successfully.')
del res
monitor_gen.join()
peak = max(monitor_gen.memory_buffer) / 1000000.0
print(f'Peak memory usage: {peak:.2f}MB')
import matplotlib.pyplot as plt
plt.figure(0)
plt.semilogy(np.maximum.accumulate(monitor.memory_buffer), label='return_as="list"')
plt.semilogy(np.maximum.accumulate(monitor_gen.memory_buffer), label='return_as="generator"')
plt.xlabel('Time')
plt.xticks([], [])
plt.ylabel('Memory usage')
plt.yticks([10000000.0, 100000000.0, 1000000000.0], ['10MB', '100MB', '1GB'])
plt.legend()
plt.show()

def return_big_object_delayed(i):
    if False:
        return 10
    if (i + 20) % 60:
        time.sleep(0.1)
    else:
        time.sleep(5)
    return i * np.ones((10000, 200), dtype=np.float64)
monitor_delayed_gen = MemoryMonitor()
print("Create result generator on delayed tasks with return_as='generator'...")
res = Parallel(n_jobs=2, return_as='generator')((delayed(return_big_object_delayed)(i) for i in range(150)))
print('Accumulate results:', end='')
res = accumulator_sum(res)
print('All tasks completed and reduced successfully.')
del res
monitor_delayed_gen.join()
peak = max(monitor_delayed_gen.memory_buffer) / 1000000.0
print(f'Peak memory usage: {peak:.2f}MB')
monitor_delayed_gen_unordered = MemoryMonitor()
print("Create result generator on delayed tasks with return_as='generator_unordered'...")
res = Parallel(n_jobs=2, return_as='generator_unordered')((delayed(return_big_object_delayed)(i) for i in range(150)))
print('Accumulate results:', end='')
res = accumulator_sum(res)
print('All tasks completed and reduced successfully.')
del res
monitor_delayed_gen_unordered.join()
peak = max(monitor_delayed_gen_unordered.memory_buffer) / 1000000.0
print(f'Peak memory usage: {peak:.2f}MB')
plt.figure(1)
plt.semilogy(np.maximum.accumulate(monitor_delayed_gen.memory_buffer), label='return_as="generator"')
plt.semilogy(np.maximum.accumulate(monitor_delayed_gen_unordered.memory_buffer), label='return_as="generator_unordered"')
plt.xlabel('Time')
plt.xticks([], [])
plt.ylabel('Memory usage')
plt.yticks([10000000.0, 100000000.0, 1000000000.0], ['10MB', '100MB', '1GB'])
plt.legend()
plt.show()