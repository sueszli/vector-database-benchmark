"""Internal side input transforms and implementations.

For internal use only; no backwards-compatibility guarantees.

Important: this module is an implementation detail and should not be used
directly by pipeline writers. Instead, users should use the helper methods
AsSingleton, AsIter, AsList and AsDict in apache_beam.pvalue.
"""
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from apache_beam.transforms import window
if TYPE_CHECKING:
    from apache_beam import pvalue
WindowMappingFn = Callable[[window.BoundedWindow], window.BoundedWindow]
SIDE_INPUT_PREFIX = 'python_side_input'
SIDE_INPUT_REGEX = SIDE_INPUT_PREFIX + '([0-9]+)(-.*)?$'

def _global_window_mapping_fn(w, global_window=window.GlobalWindow()):
    if False:
        while True:
            i = 10
    return global_window

def default_window_mapping_fn(target_window_fn):
    if False:
        return 10
    if target_window_fn == window.GlobalWindows():
        return _global_window_mapping_fn
    if isinstance(target_window_fn, window.Sessions):
        raise RuntimeError('Sessions is not allowed in side inputs')

    def map_via_end(source_window):
        if False:
            print('Hello World!')
        return list(target_window_fn.assign(window.WindowFn.AssignContext(source_window.max_timestamp())))[-1]
    return map_via_end

def get_sideinput_index(tag):
    if False:
        for i in range(10):
            print('nop')
    match = re.match(SIDE_INPUT_REGEX, tag, re.DOTALL)
    if match:
        return int(match.group(1))
    else:
        raise RuntimeError('Invalid tag %r' % tag)

class SideInputMap(object):
    """Represents a mapping of windows to side input values."""

    def __init__(self, view_class, view_options, iterable):
        if False:
            while True:
                i = 10
        self._window_mapping_fn = view_options.get('window_mapping_fn', _global_window_mapping_fn)
        self._view_class = view_class
        self._view_options = view_options
        self._iterable = iterable
        self._cache = {}

    def __getitem__(self, window):
        if False:
            print('Hello World!')
        if window not in self._cache:
            target_window = self._window_mapping_fn(window)
            self._cache[window] = self._view_class._from_runtime_iterable(_FilteringIterable(self._iterable, target_window), self._view_options)
        return self._cache[window]

    def is_globally_windowed(self):
        if False:
            return 10
        return self._window_mapping_fn == _global_window_mapping_fn

class _FilteringIterable(object):
    """An iterable containing only those values in the given window.
  """

    def __init__(self, iterable, target_window):
        if False:
            print('Hello World!')
        self._iterable = iterable
        self._target_window = target_window

    def __iter__(self):
        if False:
            print('Hello World!')
        for wv in self._iterable:
            if self._target_window in wv.windows:
                yield wv.value

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return (list, (list(self),))