"""Functions used to extract and analyze stacks.  Faster than Python libs."""
import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
_get_thread_key = threading.get_ident
_source_mapper_stacks = collections.defaultdict(lambda : [SentinelMapper()])
_source_filter_stacks = collections.defaultdict(lambda : [SentinelFilter()])

class StackTraceTransform(object):
    """Base class for stack trace transformation functions."""
    _stack_dict = None
    _thread_key = None

    def __enter__(self):
        if False:
            return 10
        if self._thread_key is None:
            self._thread_key = _get_thread_key()
        else:
            assert self._thread_key == _get_thread_key(), 'Shared across threads?'
        stack = self._stack_dict[self._thread_key]
        self.parent = stack[-1]
        stack.append(self)
        self.update()
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        if False:
            print('Hello World!')
        top = self._stack_dict[self._thread_key].pop()
        assert top is self, 'Concurrent access?'

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('subclasses need to override this')

class StackTraceMapper(StackTraceTransform):
    """Allows remapping traceback information to different source code."""
    _stack_dict = _source_mapper_stacks

    def __init__(self):
        if False:
            print('Hello World!')
        self.internal_map = _tf_stack.PyBindSourceMap()

    def update(self):
        if False:
            print('Hello World!')
        self.internal_map.update_to(tuple(self.get_effective_source_map().items()))

    def get_effective_source_map(self):
        if False:
            return 10
        'Returns a map (filename, lineno) -> (filename, lineno, function_name).'
        raise NotImplementedError('subclasses need to override this')
EMPTY_DICT = {}

class SentinelMapper(StackTraceMapper):

    def get_effective_source_map(self):
        if False:
            while True:
                i = 10
        return EMPTY_DICT

class StackTraceFilter(StackTraceTransform):
    """Allows filtering traceback information by removing superfluous frames."""
    _stack_dict = _source_filter_stacks

    def __init__(self):
        if False:
            while True:
                i = 10
        self.internal_set = _tf_stack.PyBindFileSet()

    def update(self):
        if False:
            return 10
        self.internal_set.update_to(set(self.get_filtered_filenames()))

    def get_filtered_filenames(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('subclasses need to override this')
EMPTY_SET = frozenset()

class SentinelFilter(StackTraceFilter):

    def get_filtered_filenames(self):
        if False:
            i = 10
            return i + 15
        return EMPTY_SET

class CurrentModuleFilter(StackTraceFilter):
    """Filters stack frames from the module where this is used (best effort)."""

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        filter_filename = None
        outer_f = None
        f = inspect.currentframe()
        try:
            if f is not None:
                outer_f = f.f_back
                if outer_f is not None:
                    filter_filename = inspect.getsourcefile(outer_f)
            self._filename = filter_filename
            self._cached_set = None
        finally:
            del f
            del outer_f

    def get_filtered_filenames(self):
        if False:
            print('Hello World!')
        if self._cached_set is not None:
            return self._cached_set
        filtered_filenames = frozenset((self._filename,))
        if self.parent is not None:
            filtered_filenames |= self.parent.get_filtered_filenames()
        self._cached_set = filtered_filenames
        return filtered_filenames

def extract_stack(stacklevel=1):
    if False:
        for i in range(10):
            print('nop')
    'An eager-friendly alternative to traceback.extract_stack.\n\n  Args:\n    stacklevel: number of initial frames to skip when producing the stack.\n\n  Returns:\n    A list-like FrameSummary containing StackFrame-like objects, which are\n    namedtuple-like objects with the following fields: filename, lineno, name,\n    line, meant to masquerade as traceback.FrameSummary objects.\n  '
    thread_key = _get_thread_key()
    return _tf_stack.extract_stack(_source_mapper_stacks[thread_key][-1].internal_map, _source_filter_stacks[thread_key][-1].internal_set, stacklevel)

def LoadTracesFromDebugInfo(debug_info):
    if False:
        return 10
    return _tf_stack.LoadTracesFromDebugInfo(debug_info.SerializeToString())

class GraphDebugInfoBuilder(_tf_stack.GraphDebugInfoBuilder):

    def AppendGraphDebugInfo(self, fn_name, fn_debug_info):
        if False:
            return 10
        debug_info_str = fn_debug_info.SerializeToString()
        super().AppendGraphDebugInfo(fn_name, debug_info_str)

    def Build(self):
        if False:
            while True:
                i = 10
        debug_info_str = super().Build()
        debug_info = graph_debug_info_pb2.GraphDebugInfo()
        debug_info.ParseFromString(debug_info_str)
        return debug_info
StackSummary = _tf_stack.StackTrace
FrameSummary = _tf_stack.StackFrame