"""Flame graph module."""
import inspect
import runpy
import signal
import time
from collections import defaultdict
from vprof import base_profiler
_SAMPLE_INTERVAL = 0.001

class _StatProfiler:
    """Statistical profiler.

    Samples call stack at regular intervals specified by _SAMPLE_INTERVAL.
    """

    def __init__(self):
        if False:
            return 10
        self._stats = defaultdict(int)
        self._start_time = None
        self.base_frame = None
        self.run_time = None

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        'Enables statistical profiler.'
        signal.signal(signal.SIGPROF, self.sample)
        signal.setitimer(signal.ITIMER_PROF, _SAMPLE_INTERVAL)
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tbf):
        if False:
            return 10
        'Disables statistical profiler.'
        self.run_time = time.time() - self._start_time
        signal.setitimer(signal.ITIMER_PROF, 0)

    def sample(self, signum, frame):
        if False:
            print('Hello World!')
        'Samples current stack and writes result in self._stats.\n\n        Args:\n            signum: Signal that activates handler.\n            frame: Frame on top of the stack when signal is handled.\n        '
        stack = []
        while frame and frame != self.base_frame:
            stack.append((frame.f_code.co_name, frame.f_code.co_filename, frame.f_code.co_firstlineno))
            frame = frame.f_back
        self._stats[tuple(stack)] += 1
        signal.setitimer(signal.ITIMER_PROF, _SAMPLE_INTERVAL)

    @staticmethod
    def _insert_stack(stack, sample_count, call_tree):
        if False:
            while True:
                i = 10
        'Inserts a stack into the call tree.\n\n        Args:\n            stack: Call stack.\n            sample_count: Sample count of call stack.\n            call_tree: Call tree.\n        '
        curr_level = call_tree
        for func in stack:
            next_level_index = {node['stack']: node for node in curr_level['children']}
            if func not in next_level_index:
                new_node = {'stack': func, 'children': [], 'sampleCount': 0}
                curr_level['children'].append(new_node)
                curr_level = new_node
            else:
                curr_level = next_level_index[func]
        curr_level['sampleCount'] = sample_count

    def _fill_sample_count(self, node):
        if False:
            return 10
        'Counts and fills sample counts inside call tree.'
        node['sampleCount'] += sum((self._fill_sample_count(child) for child in node['children']))
        return node['sampleCount']

    @staticmethod
    def _get_percentage(sample_count, total_samples):
        if False:
            for i in range(10):
                print('nop')
        'Return percentage of sample_count in total_samples.'
        if total_samples != 0:
            return 100 * round(float(sample_count) / total_samples, 3)
        return 0.0

    def _format_tree(self, node, total_samples):
        if False:
            i = 10
            return i + 15
        'Reformats call tree for the UI.'
        (funcname, filename, _) = node['stack']
        sample_percent = self._get_percentage(node['sampleCount'], total_samples)
        color_hash = base_profiler.hash_name('%s @ %s' % (funcname, filename))
        return {'stack': node['stack'], 'children': [self._format_tree(child, total_samples) for child in node['children']], 'sampleCount': node['sampleCount'], 'samplePercentage': sample_percent, 'colorHash': color_hash}

    @property
    def call_tree(self):
        if False:
            print('Hello World!')
        'Returns call tree.'
        call_tree = {'stack': 'base', 'sampleCount': 0, 'children': []}
        for (stack, sample_count) in self._stats.items():
            self._insert_stack(reversed(stack), sample_count, call_tree)
        self._fill_sample_count(call_tree)
        if not call_tree['children']:
            return {}
        return self._format_tree(call_tree['children'][0], call_tree['sampleCount'])

class FlameGraphProfiler(base_profiler.BaseProfiler):
    """Statistical profiler wrapper.

    Runs statistical profiler and returns collected stats.
    """

    def _profile_package(self):
        if False:
            for i in range(10):
                print('nop')
        'Runs statistical profiler on a package.'
        with _StatProfiler() as prof:
            prof.base_frame = inspect.currentframe()
            try:
                runpy.run_path(self._run_object, run_name='__main__')
            except SystemExit:
                pass
        call_tree = prof.call_tree
        return {'objectName': self._object_name, 'sampleInterval': _SAMPLE_INTERVAL, 'runTime': prof.run_time, 'callStats': call_tree, 'totalSamples': call_tree.get('sampleCount', 0), 'timestamp': int(time.time())}

    def profile_package(self):
        if False:
            for i in range(10):
                print('nop')
        'Runs package profiler in a separate process.'
        return base_profiler.run_in_separate_process(self._profile_package)

    def _profile_module(self):
        if False:
            i = 10
            return i + 15
        'Runs statistical profiler on a module.'
        with open(self._run_object, 'rb') as srcfile, _StatProfiler() as prof:
            code = compile(srcfile.read(), self._run_object, 'exec')
            prof.base_frame = inspect.currentframe()
            try:
                exec(code, self._globs, None)
            except SystemExit:
                pass
        call_tree = prof.call_tree
        return {'objectName': self._object_name, 'sampleInterval': _SAMPLE_INTERVAL, 'runTime': prof.run_time, 'callStats': call_tree, 'totalSamples': call_tree.get('sampleCount', 0), 'timestamp': int(time.time())}

    def profile_module(self):
        if False:
            for i in range(10):
                print('nop')
        'Runs module profiler in a separate process.'
        return base_profiler.run_in_separate_process(self._profile_module)

    def profile_function(self):
        if False:
            i = 10
            return i + 15
        'Runs statistical profiler on a function.'
        with _StatProfiler() as prof:
            result = self._run_object(*self._run_args, **self._run_kwargs)
        call_tree = prof.call_tree
        return {'objectName': self._object_name, 'sampleInterval': _SAMPLE_INTERVAL, 'runTime': prof.run_time, 'callStats': call_tree, 'totalSamples': call_tree.get('sampleCount', 0), 'result': result, 'timestamp': int(time.time())}