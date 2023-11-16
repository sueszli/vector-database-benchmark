from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import time
import memory_profiler
from IPython import get_ipython
import psutil
import warnings
previous_call_memory_usage = memory_profiler.memory_usage()[0]
t1 = time.time()
keep_watching = True
watching_memory = True
try:
    input_cells = get_ipython().user_ns['In']
except Exception:
    warnings.warn('Not running on notebook')

def start_watching_memory():
    if False:
        print('Hello World!')
    'Register memory profiling tools to IPython instance.'
    global watching_memory
    watching_memory = True
    ip = get_ipython()
    ip.events.register('post_run_cell', watch_memory)
    ip.events.register('pre_run_cell', pre_run_cell)

def stop_watching_memory():
    if False:
        return 10
    'Unregister memory profiling tools from IPython instance.'
    global watching_memory
    watching_memory = False
    ip = get_ipython()
    try:
        ip.events.unregister('post_run_cell', watch_memory)
    except ValueError:
        print('ERROR: problem when unregistering')
        pass
    try:
        ip.events.unregister('pre_run_cell', pre_run_cell)
    except ValueError:
        print('ERROR: problem when unregistering')
        pass

def watch_memory():
    if False:
        return 10
    'Bring in the global memory usage value from the previous iteration'
    global previous_call_memory_usage, keep_watching, watching_memory, input_cells
    new_memory_usage = memory_profiler.memory_usage()[0]
    memory_delta = new_memory_usage - previous_call_memory_usage
    keep_watching = False
    total_memory = psutil.virtual_memory()[0] / 1024 / 1024
    time_delta_secs = time.time() - t1
    num_commands = len(input_cells) - 1
    cmd = 'In [{}]'.format(num_commands)
    output_template = '{cmd} used {memory_delta:0.4f} Mb RAM in {time_delta:0.2f}s, total RAM usage {memory_usage:0.2f} Mb, total RAM memory {total_memory:0.2f} Mb'
    output = output_template.format(time_delta=time_delta_secs, cmd=cmd, memory_delta=memory_delta, memory_usage=new_memory_usage, total_memory=total_memory)
    if watching_memory:
        print(str(output))
    previous_call_memory_usage = new_memory_usage

def pre_run_cell():
    if False:
        return 10
    'Capture current time before we execute the current command'
    global t1
    t1 = time.time()