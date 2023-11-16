import pickle
import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import Any
from itertools import groupby
import base64
import warnings
cache = lru_cache(None)
__all__ = ['format_flamegraph', 'segments', 'memory', 'compare']

def _frame_fmt(f, full_filename=False):
    if False:
        while True:
            i = 10
    i = f['line']
    fname = f['filename']
    if not full_filename:
        fname = fname.split('/')[-1]
    func = f['name']
    return f'{fname}:{i}:{func}'

@cache
def _frame_filter(name, filename):
    if False:
        for i in range(10):
            print('nop')
    omit_functions = ['unwind::unwind', 'CapturedTraceback::gather', 'gather_with_cpp', '_start', '__libc_start_main', 'PyEval_', 'PyObject_', 'PyFunction_']
    omit_filenames = ['core/boxing', '/Register', '/Redispatch', 'pythonrun.c', 'Modules/main.c', 'Objects/call.c', 'Objects/methodobject.c', 'pycore_ceval.h', 'ceval.c', 'cpython/abstract.h']
    for of in omit_functions:
        if of in name:
            return False
    for of in omit_filenames:
        if of in filename:
            return False
    return True

def _frames_fmt(frames, full_filename=False, reverse=False):
    if False:
        i = 10
        return i + 15
    if reverse:
        frames = reversed(frames)
    return [_frame_fmt(f, full_filename) for f in frames if _frame_filter(f['name'], f['filename'])]

def _block_extra_legacy(b):
    if False:
        print('Hello World!')
    if 'history' in b:
        frames = b['history'][0].get('frames', [])
        real_size = b['history'][0]['real_size']
    else:
        real_size = b.get('requested_size', b['size'])
        frames = []
    return (frames, real_size)

def _block_extra(b):
    if False:
        while True:
            i = 10
    if 'frames' not in b:
        return _block_extra_legacy(b)
    return (b['frames'], b['requested_size'])

def format_flamegraph(flamegraph_lines, flamegraph_script=None):
    if False:
        i = 10
        return i + 15
    if flamegraph_script is None:
        flamegraph_script = f'/tmp/{os.getuid()}_flamegraph.pl'
    if not os.path.exists(flamegraph_script):
        import urllib.request
        print(f'Downloading flamegraph.pl to: {flamegraph_script}')
        urllib.request.urlretrieve('https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl', flamegraph_script)
        subprocess.check_call(['chmod', '+x', flamegraph_script])
    args = [flamegraph_script, '--countname', 'bytes']
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8')
    assert p.stdin is not None
    assert p.stdout is not None
    p.stdin.write(flamegraph_lines)
    p.stdin.close()
    result = p.stdout.read()
    p.stdout.close()
    p.wait()
    assert p.wait() == 0
    return result

def _write_blocks(f, prefix, blocks):
    if False:
        while True:
            i = 10

    def frames_fragment(frames):
        if False:
            return 10
        if not frames:
            return '<non-python>'
        return ';'.join(_frames_fmt(frames, reverse=True))
    for b in blocks:
        if 'history' not in b:
            (frames, accounted_for_size) = _block_extra(b)
            f.write(f"{prefix};{b['state']};{frames_fragment(frames)} {accounted_for_size}\n")
        else:
            accounted_for_size = 0
            for h in b['history']:
                sz = h['real_size']
                accounted_for_size += sz
                if 'frames' in h:
                    frames = h['frames']
                    f.write(f"{prefix};{b['state']};{frames_fragment(frames)} {sz}\n")
                else:
                    f.write(f"{prefix};{b['state']};<no-context> {sz}\n")
        gaps = b['size'] - accounted_for_size
        if gaps:
            f.write(f"{prefix};{b['state']};<gaps> {gaps}\n")

def segments(snapshot, format_flamegraph=format_flamegraph):
    if False:
        i = 10
        return i + 15
    f = io.StringIO()
    for seg in snapshot['segments']:
        prefix = f"stream_{seg['stream']};seg_{seg['address']}"
        _write_blocks(f, prefix, seg['blocks'])
    return format_flamegraph(f.getvalue())

def memory(snapshot, format_flamegraph=format_flamegraph):
    if False:
        return 10
    f = io.StringIO()
    for seg in snapshot['segments']:
        prefix = f"stream_{seg['stream']}"
        _write_blocks(f, prefix, seg['blocks'])
    return format_flamegraph(f.getvalue())

def compare(before, after, format_flamegraph=format_flamegraph):
    if False:
        for i in range(10):
            print('nop')

    def _seg_key(seg):
        if False:
            i = 10
            return i + 15
        return (seg['address'], seg['total_size'])

    def _seg_info(seg):
        if False:
            i = 10
            return i + 15
        return f"stream_{seg['stream']};seg_{seg['address']}"
    f = io.StringIO()
    before_segs = {_seg_key(seg) for seg in before}
    after_segs = {_seg_key(seg) for seg in after}
    print(f'only_before = {[a for (a, _) in before_segs - after_segs]}')
    print(f'only_after = {[a for (a, _) in after_segs - before_segs]}')
    for seg in before:
        if _seg_key(seg) not in after_segs:
            _write_blocks(f, f'only_before;{_seg_info(seg)}', seg['blocks'])
    for seg in after:
        if _seg_key(seg) not in before_segs:
            _write_blocks(f, f'only_after;{_seg_info(seg)}', seg['blocks'])
    return format_flamegraph(f.getvalue())

def _format_size(num):
    if False:
        while True:
            i = 10
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}B'
        num /= 1024.0
    return f'{num:.1f}YiB'

class Bytes:

    def __init__(self, value):
        if False:
            return 10
        self.value = value

    def __add__(self, rhs):
        if False:
            i = 10
            return i + 15
        return Bytes(self.value + rhs)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return _format_size(self.value)

def calc_active(seg):
    if False:
        return 10
    return sum((b['size'] for b in seg['blocks'] if b['state'] == 'active_allocated'))

def _report_free(free_external, free_internal):
    if False:
        print('Hello World!')
    total = free_external + free_internal
    suffix = ''
    if total != 0:
        pct = free_internal / total * 100
        suffix = f' ({pct:.1f}% internal)'
    return f'{Bytes(total)}{suffix}'
PAGE_SIZE = 1024 * 1024 * 20
legend = f"\nLegend:\n    [a     ] - a segment in the allocator\n     ^-- a page {Bytes(PAGE_SIZE)} of memory in the segment\n    a-z: pages filled with a single block's content\n    ' ': page is completely free\n    *: page if completely full with multiple blocks\n    0-9: page is partially full with tensors of multiple blocks (9 == 90% full)\n    (X% internal) - of the free memory, X% is free because we rounded the size of the allocation.\n"

def segsum(data):
    if False:
        while True:
            i = 10
    'Visually reports how the allocator has filled its segments.\n\n    This printout can help debug fragmentation issues since free fragments\n    will appear as gaps in this printout.  The amount of free space is reported\n    for each segment.\n    We distinguish between internal free memory which occurs because the\n    allocator rounds the allocation size, and external free memory, which are\n    the gaps between allocations in a segment.\n    Args:\n        data: snapshot dictionary created from _snapshot()\n    '
    segments = []
    out = io.StringIO()
    out.write(f'Summary of segments >= {Bytes(PAGE_SIZE)} in size\n')
    total_reserved = 0
    total_allocated = 0
    free_external = 0
    free_internal = 0
    for seg in sorted(data['segments'], key=lambda x: (x['total_size'], calc_active(x))):
        total_reserved += seg['total_size']
        seg_free_external = 0
        seg_free_internal = 0
        seg_allocated = 0
        all_ranges = []
        boffset = 0
        for b in seg['blocks']:
            active = b['state'] == 'active_allocated'
            if active:
                (_, allocated_size) = _block_extra(b)
                all_ranges.append((boffset, allocated_size, True))
                seg_allocated += allocated_size
                seg_free_internal += b['size'] - allocated_size
            else:
                seg_free_external += b['size']
            boffset += b['size']
        total_allocated += seg_allocated
        free_external += seg_free_external
        free_internal += seg_free_internal
        nseg = (seg['total_size'] - 1) // PAGE_SIZE + 1
        occupied = [' ' for _ in range(nseg)]
        frac = [0.0 for _ in range(nseg)]
        active_size = 0
        for (i, (start_, size, active)) in enumerate(all_ranges):
            active_size += size
            finish_ = start_ + size
            start = start_ // PAGE_SIZE
            finish = (finish_ - 1) // PAGE_SIZE + 1
            m = chr(ord('a' if active else 'A') + i % 26)
            for j in range(start, finish):
                s = max(start_, j * PAGE_SIZE)
                e = min(finish_, (j + 1) * PAGE_SIZE)
                frac[j] += (e - s) / PAGE_SIZE
                if occupied[j] != ' ':
                    occupied[j] = '0123456789*'[int(frac[j] * 10)]
                else:
                    occupied[j] = m
        stream = '' if seg['stream'] == 0 else f", stream_{seg['stream']}"
        body = ''.join(occupied)
        assert seg_free_external + seg_free_internal + seg_allocated == seg['total_size']
        stream = f" stream_{seg['stream']}" if seg['stream'] != 0 else ''
        if seg['total_size'] >= PAGE_SIZE:
            out.write(f"[{body}] {Bytes(seg['total_size'])} allocated, {_report_free(seg_free_external, seg_free_internal)} free{stream}\n")
    out.write(f"segments: {len(data['segments'])}\n")
    out.write(f'total_reserved: {Bytes(total_reserved)}\n')
    out.write(f'total_allocated: {Bytes(total_allocated)}\n')
    internal_external = f' ({Bytes(free_internal)} internal + {Bytes(free_external)} external)' if free_internal else ''
    out.write(f'total_free: {_report_free(free_external, free_internal)}\n')
    out.write(legend)
    assert free_internal + free_external + total_allocated == total_reserved
    return out.getvalue()

def trace(data):
    if False:
        for i in range(10):
            print('nop')
    out = io.StringIO()

    def format(entries):
        if False:
            while True:
                i = 10
        segment_intervals: list = []
        segment_addr_to_name = {}
        allocation_addr_to_name = {}
        free_names: list = []
        next_name = 0

        def _name():
            if False:
                i = 10
                return i + 15
            nonlocal next_name
            if free_names:
                return free_names.pop()
            (r, m) = (next_name // 26, next_name % 26)
            next_name += 1
            return f"{chr(ord('a') + m)}{('' if r == 0 else r)}"

        def find_segment(addr):
            if False:
                while True:
                    i = 10
            for (name, saddr, size) in segment_intervals:
                if addr >= saddr and addr < saddr + size:
                    return (name, saddr)
            for (i, seg) in enumerate(data['segments']):
                saddr = seg['address']
                size = seg['allocated_size']
                if addr >= saddr and addr < saddr + size:
                    return (f'seg_{i}', saddr)
            return (None, None)
        count = 0
        out.write(f'{len(entries)} entries\n')
        total_reserved = 0
        for seg in data['segments']:
            total_reserved += seg['total_size']
        for (count, e) in enumerate(entries):
            if e['action'] == 'alloc':
                (addr, size) = (e['addr'], e['size'])
                n = _name()
                (seg_name, seg_addr) = find_segment(addr)
                if seg_name is None:
                    seg_name = 'MEM'
                    offset = addr
                else:
                    offset = addr - seg_addr
                out.write(f'{n} = {seg_name}[{offset}:{Bytes(size)}]\n')
                allocation_addr_to_name[addr] = (n, size, count)
                count += size
            elif e['action'] == 'free_requested':
                (addr, size) = (e['addr'], e['size'])
                (name, _, _) = allocation_addr_to_name.get(addr, (addr, None, None))
                out.write(f'del {name} # {Bytes(size)}\n')
            elif e['action'] == 'free_completed':
                (addr, size) = (e['addr'], e['size'])
                count -= size
                (name, _, _) = allocation_addr_to_name.get(addr, (addr, None, None))
                out.write(f'# free completed for {name} {Bytes(size)}\n')
                if name in allocation_addr_to_name:
                    free_names.append(name)
                    del allocation_addr_to_name[name]
            elif e['action'] == 'segment_alloc':
                (addr, size) = (e['addr'], e['size'])
                name = _name()
                out.write(f'{name} = cudaMalloc({addr}, {Bytes(size)})\n')
                segment_intervals.append((name, addr, size))
                segment_addr_to_name[addr] = name
            elif e['action'] == 'segment_free':
                (addr, size) = (e['addr'], e['size'])
                name = segment_addr_to_name.get(addr, addr)
                out.write(f'cudaFree({name}) # {Bytes(size)}\n')
                if name in segment_addr_to_name:
                    free_names.append(name)
                    del segment_addr_to_name[name]
            elif e['action'] == 'oom':
                size = e['size']
                free = e['device_free']
                out.write(f'raise OutOfMemoryError() # {Bytes(size)} requested, {Bytes(free)} free in CUDA\n')
            else:
                out.write(f'{e}\n')
        out.write(f'TOTAL MEM: {Bytes(count)}')
    for (i, d) in enumerate(data['device_traces']):
        if d:
            out.write(f'Device {i} ----------------\n')
            format(d)
    return out.getvalue()
_memory_viz_template = '\n<!DOCTYPE html>\n<html>\n<head>\n</head>\n<body>\n<script type="module">\nimport {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"\nconst local_files = $SNAPSHOT\nadd_local_files(local_files, $VIZ_KIND)\n</script>\n</body>\n'

def _format_viz(data, viz_kind, device):
    if False:
        return 10
    if device is not None:
        warnings.warn('device argument is deprecated, plots now contain all device')
    buffer = pickle.dumps(data)
    buffer += b'\x00' * (3 - len(buffer) % 3)
    encoded_buffer = base64.b64encode(buffer).decode('utf-8')
    json_format = json.dumps([{'name': 'snapshot.pickle', 'base64': encoded_buffer}])
    return _memory_viz_template.replace('$VIZ_KIND', repr(viz_kind)).replace('$SNAPSHOT', json_format)

def trace_plot(data, device=None, plot_segments=False):
    if False:
        print('Hello World!')
    'Generate a visualization over time of the memory usage recorded by the trace as an html file.\n\n    Args:\n        data: Memory snapshot as generated from torch.cuda.memory._snapshot()\n        device (torch.device, optional): Generate the trace for this device, needed if multiple devices have allocations.\n        plot_segments (bool, optional): Plots memory returned from cudaMalloc, rather than individual allocations.\n                                        Defaults to False.\n\n    Returns:\n        str: HTML of visualization\n    '
    return _format_viz(data, 'Active Memory Timeline' if not plot_segments else 'Active Cached Memory Timeline', device)

def _profile_to_snapshot(profile):
    if False:
        i = 10
        return i + 15
    import torch
    from torch.profiler._memory_profiler import Action, TensorKey
    from torch._C._profiler import _EventType
    memory_profile = profile._memory_profile()
    allocation_stacks = {}
    for event in memory_profile._op_tree.sorted_nodes:
        if event.tag == _EventType.Allocation:
            parent = event.parent
            python_parents = []
            while parent:
                if parent.tag in (_EventType.PyCall, _EventType.PyCCall):
                    python_parents.append(parent)
                parent = parent.parent
            key = TensorKey.from_allocation(event.extra_fields)
            if key and event.extra_fields.alloc_size > 0:
                allocation_stacks[key] = python_parents
    device_count = torch.cuda.device_count()
    snapshot = {'device_traces': [[] for _ in range(device_count + 1)], 'segments': [{'device': device, 'address': None, 'total_size': 0, 'stream': 0, 'blocks': []} for device in range(device_count + 1)]}

    def to_device(device):
        if False:
            for i in range(10):
                print('nop')
        if device.type == 'cuda':
            return device.index
        else:
            return device_count

    def allocate(size, tensor_key, version, during_trace=True):
        if False:
            print('Hello World!')
        device = to_device(tensor_key.device)
        addr = tensor_key.storage.ptr
        seg = snapshot['segments'][device]
        if seg['address'] is None or seg['address'] > addr:
            seg['address'] = addr
        seg['total_size'] = max(seg['total_size'], addr + size)
        category = memory_profile._categories.get(tensor_key, version)
        category = category.name.lower() if category is not None else 'unknown'
        stack = allocation_stacks.get(tensor_key, ())
        stack = [{'filename': 'none', 'line': 0, 'name': p.name} for p in stack]
        r = {'action': 'alloc', 'addr': addr, 'size': size, 'stream': 0, 'frames': stack, 'category': category}
        if during_trace:
            snapshot['device_traces'][device].append(r)
        return r

    def free(alloc, device):
        if False:
            while True:
                i = 10
        for e in ('free_requested', 'free_completed'):
            snapshot['device_traces'][device].append({'action': e, 'addr': alloc['addr'], 'size': alloc['size'], 'stream': 0, 'frames': alloc['frames']})
    kv_to_elem = {}
    for (time, action, (tensor_key, version), size) in memory_profile.timeline:
        if not isinstance(tensor_key, TensorKey):
            continue
        if action == Action.CREATE:
            kv_to_elem[tensor_key, version] = allocate(size, tensor_key, version)
        elif action == Action.DESTROY:
            free(kv_to_elem.pop((tensor_key, version)), to_device(tensor_key.device))
        elif action == Action.INCREMENT_VERSION:
            free(kv_to_elem.pop((tensor_key, version)), to_device(tensor_key.device))
            kv_to_elem[tensor_key, version + 1] = allocate(size, tensor_key, version + 1)
        elif action == Action.PREEXISTING:
            kv_to_elem[tensor_key, version] = allocate(size, tensor_key, version, during_trace=False)
    blocks_at_end = [(to_device(tensor_key.device), event['addr'], event['size'], event['frames']) for ((tensor_key, version), event) in kv_to_elem.items()]
    for (device, blocks) in groupby(sorted(blocks_at_end), key=lambda x: x[0]):
        seg = snapshot['segments'][device]
        last_addr = seg['address']
        for (_, addr, size, frames) in blocks:
            if last_addr < addr:
                seg['blocks'].append({'size': addr - last_addr, 'state': 'inactive'})
            seg['blocks'].append({'size': size, 'state': 'active_allocated', 'requested_size': size, 'frames': frames})
            last_addr = addr + size
        if last_addr < seg['total_size']:
            seg['blocks'].append({'size': seg['total_size'] - last_addr, 'state': 'inactive'})
    snapshot['segments'] = [seg for seg in snapshot['segments'] if seg['blocks']]
    for seg in snapshot['segments']:
        seg['total_size'] -= seg['address']
        if not seg['blocks']:
            seg['blocks'].append({'size': seg['total_size'], 'state': 'inactive'})
    return snapshot

def profile_plot(profile, device=None):
    if False:
        i = 10
        return i + 15
    'Generate a visualization over time of the memory usage recorded by kineto memory profiling as an html file.\n\n    Args:\n        profile: profile as generated by `torch.profiler.profile(profile_memory=True)`\n        device (torch.device, optional): Generate the trace for this device, needed if multiple devices have allocations.\n\n    Returns:\n        str: HTML of visualization\n    '
    snapshot = _profile_to_snapshot(profile)
    return _format_viz(snapshot, 'Active Memory Timeline', device)

def segment_plot(data: Any, device=None):
    if False:
        while True:
            i = 10
    return _format_viz(data, 'Allocator State History', device)
if __name__ == '__main__':
    import os.path
    thedir = os.path.realpath(os.path.dirname(__file__))
    if thedir in sys.path:
        sys.path.remove(thedir)
    import argparse
    fn_name = 'torch.cuda.memory._snapshot()'
    pickled = f'pickled memory statistics from {fn_name}'
    parser = argparse.ArgumentParser(description=f'Visualize memory dumps produced by {fn_name}')
    subparsers = parser.add_subparsers(dest='action')

    def _output(p):
        if False:
            print('Hello World!')
        p.add_argument('-o', '--output', default='output.svg', help='flamegraph svg (default: output.svg)')
    description = 'Prints overall allocation statistics and a visualization of how the allocators segments are currently filled.'
    stats_a = subparsers.add_parser('stats', description=description)
    stats_a.add_argument('input', help=pickled)
    description = 'Prints buffer of the most recent allocation events embedded in the snapshot in a Pythonic style.'
    trace_a = subparsers.add_parser('trace', description=description)
    trace_a.add_argument('input', help=pickled)
    description = 'Generate a flamegraph that visualizes what memory is stored in each allocator segment (aka block)'
    segments_a = subparsers.add_parser('segments', description=description)
    segments_a.add_argument('input', help=pickled)
    _output(segments_a)
    description = 'Generate a flamegraph the program locations contributing to CUDA memory usage.'
    memory_a = subparsers.add_parser('memory', description=description)
    memory_a.add_argument('input', help=pickled)
    _output(memory_a)
    description = 'Generate a flamegraph that shows segments (aka blocks) that have been added or removed between two different memorys snapshots.'
    compare_a = subparsers.add_parser('compare', description=description)
    compare_a.add_argument('before', help=pickled)
    compare_a.add_argument('after', help=pickled)
    _output(compare_a)
    plots = (('trace_plot', 'Generate a visualization over time of the memory usage recorded by the trace as an html file.'), ('segment_plot', 'Visualize how allocations are packed into allocator segments at each point in a trace as an html file.'))
    for (cmd, description) in plots:
        trace_plot_a = subparsers.add_parser(cmd, description=description)
        trace_plot_a.add_argument('input', help=pickled)
        help = 'visualize trace from this device (default: chooses the only device with trace info or errors)'
        trace_plot_a.add_argument('-d', '--device', type=int, default=None, help=help)
        help = 'path to save the visualization(default: output.html)'
        trace_plot_a.add_argument('-o', '--output', default='output.html', help=help)
        if cmd == 'trace_plot':
            help = 'visualize change to segments rather than individual allocations'
            trace_plot_a.add_argument('-s', '--segments', action='store_true', help=help)
    args = parser.parse_args()

    def _read(name):
        if False:
            print('Hello World!')
        if name == '-':
            f = sys.stdin.buffer
        else:
            f = open(name, 'rb')
        data = pickle.load(f)
        if isinstance(data, list):
            data = {'segments': data, 'traces': []}
        return data

    def _write(name, data):
        if False:
            return 10
        with open(name, 'w') as f:
            f.write(data)
    if args.action == 'segments':
        data = _read(args.input)
        _write(args.output, segments(data))
    elif args.action == 'memory':
        data = _read(args.input)
        _write(args.output, memory(data))
    elif args.action == 'stats':
        data = _read(args.input)
        print(segsum(data))
    elif args.action == 'trace':
        data = _read(args.input)
        print(trace(data))
    elif args.action == 'compare':
        before = _read(args.before)
        after = _read(args.after)
        _write(args.output, compare(before, after))
    elif args.action == 'trace_plot':
        data = _read(args.input)
        _write(args.output, trace_plot(data, device=args.device, plot_segments=args.segments))
    elif args.action == 'segment_plot':
        data = _read(args.input)
        _write(args.output, segment_plot(data, device=args.device))