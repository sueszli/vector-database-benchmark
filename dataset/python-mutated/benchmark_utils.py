import contextlib
import time
import os
import json
import torch
from torch.profiler import profile, ProfilerActivity

def synchronize():
    if False:
        return 10
    pass

def dump_chrome_trace(f, input, trace_filename, optimize_ctx, activities, num_runs=1, devices=None, kwargs_for_f=None, kwargs_for_profiler=None):
    if False:
        return 10
    '\n    Output the chrome trace of running f(input, **kwargs_for_f) with [optimize_ctx]\n    [num_runs] times to [trace_filename].\n\n    [activities] are the activities that the profiler will record, e.g. ProfilerActivity.CUDA.\n    Return total runtime without the profiler\n\n    Outputs to trace_filename\n    '
    if devices is None:
        devices = ['cuda']
    global synchronize
    if devices != ['cpu'] and torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
    if kwargs_for_f is None:
        kwargs_for_f = {}
    if kwargs_for_profiler is None:
        kwargs_for_profiler = {}
    with optimize_ctx:
        torch.manual_seed(1337)
        for _ in range(5):
            f(input, **kwargs_for_f)
            synchronize()
        torch.manual_seed(1337)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            f(input, **kwargs_for_f)
            synchronize()
        t1 = time.perf_counter()
    timing = t1 - t0
    with profile(activities=activities, **kwargs_for_profiler) as prof:
        with optimize_ctx:
            synchronize()
            torch.manual_seed(1337)
            for _ in range(num_runs):
                f(input, **kwargs_for_f)
                synchronize()
    prof.export_chrome_trace(trace_filename)
    return timing

def get_chrome_trace_events(filename):
    if False:
        i = 10
        return i + 15
    f = open(filename)
    data = json.load(f)
    events = data['traceEvents']
    return events

def is_gpu_compute_event(event):
    if False:
        while True:
            i = 10
    global gpu_pids
    return 'pid' in event and event['pid'] in gpu_pids and ('ph' in event) and (event['ph'] == 'X')

def get_sorted_gpu_events(events):
    if False:
        return 10
    sorted_gpu_events = []
    for event in events:
        if not is_gpu_compute_event(event):
            continue
        sorted_gpu_events.append(event)
    return sorted(sorted_gpu_events, key=lambda x: x['ts'])

def get_duration(sorted_gpu_events):
    if False:
        print('Hello World!')
    if len(sorted_gpu_events) == 0:
        return 0
    event = sorted_gpu_events[0]
    current_end_time = event['ts'] + event['dur']
    total_duration = event['dur']
    for event in sorted_gpu_events[1:]:
        start_time = max(event['ts'], current_end_time)
        end_time = event['ts'] + event['dur']
        total_duration = total_duration + max(end_time - start_time, 0)
        current_end_time = max(current_end_time, end_time)
    return total_duration

def get_sorted_gpu_mm_conv_events(events):
    if False:
        while True:
            i = 10

    def is_mm_conv_event(event):
        if False:
            while True:
                i = 10
        return 'name' in event and ('gemm' in event['name'] or 'conv' in event['name'] or 'cutlass' in event['name'] or ('wgrad' in event['name']))
    gpu_events = get_sorted_gpu_events(events)
    sorted_events = []
    for event in gpu_events:
        if not is_mm_conv_event(event):
            continue
        sorted_events.append(event)
    return sorted_events
gpu_pids = []

def compute_utilization(filename: str, total_length: float):
    if False:
        i = 10
        return i + 15
    '\n    Process the chrome traces outputs by the pytorch profiler to compute GPU Utilization\n    and percent of times spent on matmul and convolution\n\n    Args:\n        filename(str): Name of chrome traces file produced by pytorch profiler\n\n        total_length(float): total length of the process without profiler in second\n\n    Return:\n        tuple: (GPU Utilization, percent of time spent on matmul and convolution)\n    '
    events = get_chrome_trace_events(filename)
    global gpu_pids
    gpu_pids = []
    for event in events:
        if 'name' not in event:
            continue
        if event['name'] == 'process_labels' and 'GPU' in event['args']['labels']:
            gpu_pids.append(event['pid'])
    total_length = total_length * 1000000.0
    sorted_gpu_events = get_sorted_gpu_events(events)
    utilization = get_duration(sorted_gpu_events) / total_length
    sorted_gpu_mm_conv_events = get_sorted_gpu_mm_conv_events(events)
    mm_conv_utilization = get_duration(sorted_gpu_mm_conv_events) / total_length
    return (utilization, mm_conv_utilization)

def benchmark_utilization(f, input, trace_folder, optimize_ctx=None, trace_file_name='tmp_chrome_trace', num_runs=1):
    if False:
        i = 10
        return i + 15
    '\n    Benchmark the GPU Utilization and percent of time spent on matmul and convolution operations of\n    running f(input, **kwargs_for_f) with [optimize_ctx] [num_runs] times.\n    It will produce a chrome trace file in trace_folder/trace_file_name.json\n\n    Example:\n\n    ```\n    def f(a):\n        return a.sum()\n    a = torch.rand(2**20, device="cuda")\n    utilization, mm_conv_utilization = benchmark_utilization(f, a, "tmp", trace_file_name = "tmp_chrome_trace")\n    ```\n\n    Args:\n        f: function to benchmark\n\n        input: input to :attr:`f`\n\n        trace_folder: name of the folder to store the chrome trace\n\n        optimize_ctx: the context in which f will run\n\n        trace_file_name: name of the dumped chrome trace file, default to "tmp_chrome_trace"\n\n        num_runs: number of times to run f, excluding the warm-up runs, default to 1.\n\n    Return:\n        tuple: (GPU Utilization, percent of time spent on matmul and convolution)\n\n    '
    isExist = os.path.exists(trace_folder)
    if not isExist:
        os.makedirs(trace_folder)
        print('create folder ' + trace_folder)
    if optimize_ctx is None:
        optimize_ctx = contextlib.nullcontext()
    chrome_trace_file_name = os.path.join(trace_folder, trace_file_name + '.json')
    total_length = dump_chrome_trace(f, input, chrome_trace_file_name, optimize_ctx, [ProfilerActivity.CUDA], num_runs=num_runs, devices='cuda')
    (utilization, mm_conv_utilization) = compute_utilization(chrome_trace_file_name, total_length)
    return (utilization, mm_conv_utilization)