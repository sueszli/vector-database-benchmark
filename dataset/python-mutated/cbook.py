"""
A collection of utility functions and classes.  Originally, many
(but not all) were from the Python Cookbook -- hence the name cbook.
"""
import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
try:
    from numpy.exceptions import VisibleDeprecationWarning
except ImportError:
    from numpy import VisibleDeprecationWarning
import matplotlib
from matplotlib import _api, _c_internal_utils

def _get_running_interactive_framework():
    if False:
        print('Hello World!')
    '\n    Return the interactive framework whose event loop is currently running, if\n    any, or "headless" if no event loop can be started, or None.\n\n    Returns\n    -------\n    Optional[str]\n        One of the following values: "qt", "gtk3", "gtk4", "wx", "tk",\n        "macosx", "headless", ``None``.\n    '
    QtWidgets = sys.modules.get('PyQt6.QtWidgets') or sys.modules.get('PySide6.QtWidgets') or sys.modules.get('PyQt5.QtWidgets') or sys.modules.get('PySide2.QtWidgets')
    if QtWidgets and QtWidgets.QApplication.instance():
        return 'qt'
    Gtk = sys.modules.get('gi.repository.Gtk')
    if Gtk:
        if Gtk.MAJOR_VERSION == 4:
            from gi.repository import GLib
            if GLib.main_depth():
                return 'gtk4'
        if Gtk.MAJOR_VERSION == 3 and Gtk.main_level():
            return 'gtk3'
    wx = sys.modules.get('wx')
    if wx and wx.GetApp():
        return 'wx'
    tkinter = sys.modules.get('tkinter')
    if tkinter:
        codes = {tkinter.mainloop.__code__, tkinter.Misc.mainloop.__code__}
        for frame in sys._current_frames().values():
            while frame:
                if frame.f_code in codes:
                    return 'tk'
                frame = frame.f_back
        del frame
    macosx = sys.modules.get('matplotlib.backends._macosx')
    if macosx and macosx.event_loop_is_running():
        return 'macosx'
    if not _c_internal_utils.display_is_valid():
        return 'headless'
    return None

def _exception_printer(exc):
    if False:
        for i in range(10):
            print('nop')
    if _get_running_interactive_framework() in ['headless', None]:
        raise exc
    else:
        traceback.print_exc()

class _StrongRef:
    """
    Wrapper similar to a weakref, but keeping a strong reference to the object.
    """

    def __init__(self, obj):
        if False:
            print('Hello World!')
        self._obj = obj

    def __call__(self):
        if False:
            print('Hello World!')
        return self._obj

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, _StrongRef) and self._obj == other._obj

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self._obj)

def _weak_or_strong_ref(func, callback):
    if False:
        return 10
    '\n    Return a `WeakMethod` wrapping *func* if possible, else a `_StrongRef`.\n    '
    try:
        return weakref.WeakMethod(func, callback)
    except TypeError:
        return _StrongRef(func)

class CallbackRegistry:
    """
    Handle registering, processing, blocking, and disconnecting
    for a set of signals and callbacks:

        >>> def oneat(x):
        ...     print('eat', x)
        >>> def ondrink(x):
        ...     print('drink', x)

        >>> from matplotlib.cbook import CallbackRegistry
        >>> callbacks = CallbackRegistry()

        >>> id_eat = callbacks.connect('eat', oneat)
        >>> id_drink = callbacks.connect('drink', ondrink)

        >>> callbacks.process('drink', 123)
        drink 123
        >>> callbacks.process('eat', 456)
        eat 456
        >>> callbacks.process('be merry', 456)   # nothing will be called

        >>> callbacks.disconnect(id_eat)
        >>> callbacks.process('eat', 456)        # nothing will be called

        >>> with callbacks.blocked(signal='drink'):
        ...     callbacks.process('drink', 123)  # nothing will be called
        >>> callbacks.process('drink', 123)
        drink 123

    In practice, one should always disconnect all callbacks when they are
    no longer needed to avoid dangling references (and thus memory leaks).
    However, real code in Matplotlib rarely does so, and due to its design,
    it is rather difficult to place this kind of code.  To get around this,
    and prevent this class of memory leaks, we instead store weak references
    to bound methods only, so when the destination object needs to die, the
    CallbackRegistry won't keep it alive.

    Parameters
    ----------
    exception_handler : callable, optional
       If not None, *exception_handler* must be a function that takes an
       `Exception` as single parameter.  It gets called with any `Exception`
       raised by the callbacks during `CallbackRegistry.process`, and may
       either re-raise the exception or handle it in another manner.

       The default handler prints the exception (with `traceback.print_exc`) if
       an interactive event loop is running; it re-raises the exception if no
       interactive event loop is running.

    signals : list, optional
        If not None, *signals* is a list of signals that this registry handles:
        attempting to `process` or to `connect` to a signal not in the list
        throws a `ValueError`.  The default, None, does not restrict the
        handled signals.
    """

    def __init__(self, exception_handler=_exception_printer, *, signals=None):
        if False:
            for i in range(10):
                print('nop')
        self._signals = None if signals is None else list(signals)
        self.exception_handler = exception_handler
        self.callbacks = {}
        self._cid_gen = itertools.count()
        self._func_cid_map = {}
        self._pickled_cids = set()

    def __getstate__(self):
        if False:
            while True:
                i = 10
        return {**vars(self), 'callbacks': {s: {cid: proxy() for (cid, proxy) in d.items() if cid in self._pickled_cids} for (s, d) in self.callbacks.items()}, '_func_cid_map': None, '_cid_gen': next(self._cid_gen)}

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        cid_count = state.pop('_cid_gen')
        vars(self).update(state)
        self.callbacks = {s: {cid: _weak_or_strong_ref(func, self._remove_proxy) for (cid, func) in d.items()} for (s, d) in self.callbacks.items()}
        self._func_cid_map = {s: {proxy: cid for (cid, proxy) in d.items()} for (s, d) in self.callbacks.items()}
        self._cid_gen = itertools.count(cid_count)

    def connect(self, signal, func):
        if False:
            while True:
                i = 10
        'Register *func* to be called when signal *signal* is generated.'
        if self._signals is not None:
            _api.check_in_list(self._signals, signal=signal)
        self._func_cid_map.setdefault(signal, {})
        proxy = _weak_or_strong_ref(func, self._remove_proxy)
        if proxy in self._func_cid_map[signal]:
            return self._func_cid_map[signal][proxy]
        cid = next(self._cid_gen)
        self._func_cid_map[signal][proxy] = cid
        self.callbacks.setdefault(signal, {})
        self.callbacks[signal][cid] = proxy
        return cid

    def _connect_picklable(self, signal, func):
        if False:
            for i in range(10):
                print('nop')
        '\n        Like `.connect`, but the callback is kept when pickling/unpickling.\n\n        Currently internal-use only.\n        '
        cid = self.connect(signal, func)
        self._pickled_cids.add(cid)
        return cid

    def _remove_proxy(self, proxy, *, _is_finalizing=sys.is_finalizing):
        if False:
            while True:
                i = 10
        if _is_finalizing():
            return
        for (signal, proxy_to_cid) in list(self._func_cid_map.items()):
            cid = proxy_to_cid.pop(proxy, None)
            if cid is not None:
                del self.callbacks[signal][cid]
                self._pickled_cids.discard(cid)
                break
        else:
            return
        if len(self.callbacks[signal]) == 0:
            del self.callbacks[signal]
            del self._func_cid_map[signal]

    def disconnect(self, cid):
        if False:
            print('Hello World!')
        '\n        Disconnect the callback registered with callback id *cid*.\n\n        No error is raised if such a callback does not exist.\n        '
        self._pickled_cids.discard(cid)
        for (signal, cid_to_proxy) in list(self.callbacks.items()):
            proxy = cid_to_proxy.pop(cid, None)
            if proxy is not None:
                break
        else:
            return
        proxy_to_cid = self._func_cid_map[signal]
        for (current_proxy, current_cid) in list(proxy_to_cid.items()):
            if current_cid == cid:
                assert proxy is current_proxy
                del proxy_to_cid[current_proxy]
        if len(self.callbacks[signal]) == 0:
            del self.callbacks[signal]
            del self._func_cid_map[signal]

    def process(self, s, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Process signal *s*.\n\n        All of the functions registered to receive callbacks on *s* will be\n        called with ``*args`` and ``**kwargs``.\n        '
        if self._signals is not None:
            _api.check_in_list(self._signals, signal=s)
        for ref in list(self.callbacks.get(s, {}).values()):
            func = ref()
            if func is not None:
                try:
                    func(*args, **kwargs)
                except Exception as exc:
                    if self.exception_handler is not None:
                        self.exception_handler(exc)
                    else:
                        raise

    @contextlib.contextmanager
    def blocked(self, *, signal=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Block callback signals from being processed.\n\n        A context manager to temporarily block/disable callback signals\n        from being processed by the registered listeners.\n\n        Parameters\n        ----------\n        signal : str, optional\n            The callback signal to block. The default is to block all signals.\n        '
        orig = self.callbacks
        try:
            if signal is None:
                self.callbacks = {}
            else:
                self.callbacks = {k: orig[k] for k in orig if k != signal}
            yield
        finally:
            self.callbacks = orig

class silent_list(list):
    """
    A list with a short ``repr()``.

    This is meant to be used for a homogeneous list of artists, so that they
    don't cause long, meaningless output.

    Instead of ::

        [<matplotlib.lines.Line2D object at 0x7f5749fed3c8>,
         <matplotlib.lines.Line2D object at 0x7f5749fed4e0>,
         <matplotlib.lines.Line2D object at 0x7f5758016550>]

    one will get ::

        <a list of 3 Line2D objects>

    If ``self.type`` is None, the type name is obtained from the first item in
    the list (if any).
    """

    def __init__(self, type, seq=None):
        if False:
            while True:
                i = 10
        self.type = type
        if seq is not None:
            self.extend(seq)

    def __repr__(self):
        if False:
            return 10
        if self.type is not None or len(self) != 0:
            tp = self.type if self.type is not None else type(self[0]).__name__
            return f'<a list of {len(self)} {tp} objects>'
        else:
            return '<an empty list>'

def _local_over_kwdict(local_var, kwargs, *keys, warning_cls=_api.MatplotlibDeprecationWarning):
    if False:
        i = 10
        return i + 15
    out = local_var
    for key in keys:
        kwarg_val = kwargs.pop(key, None)
        if kwarg_val is not None:
            if out is None:
                out = kwarg_val
            else:
                _api.warn_external(f'"{key}" keyword argument will be ignored', warning_cls)
    return out

def strip_math(s):
    if False:
        while True:
            i = 10
    '\n    Remove latex formatting from mathtext.\n\n    Only handles fully math and fully non-math strings.\n    '
    if len(s) >= 2 and s[0] == s[-1] == '$':
        s = s[1:-1]
        for (tex, plain) in [('\\times', 'x'), ('\\mathdefault', ''), ('\\rm', ''), ('\\cal', ''), ('\\tt', ''), ('\\it', ''), ('\\', ''), ('{', ''), ('}', '')]:
            s = s.replace(tex, plain)
    return s

def _strip_comment(s):
    if False:
        print('Hello World!')
    'Strip everything from the first unquoted #.'
    pos = 0
    while True:
        quote_pos = s.find('"', pos)
        hash_pos = s.find('#', pos)
        if quote_pos < 0:
            without_comment = s if hash_pos < 0 else s[:hash_pos]
            return without_comment.strip()
        elif 0 <= hash_pos < quote_pos:
            return s[:hash_pos].strip()
        else:
            closing_quote_pos = s.find('"', quote_pos + 1)
            if closing_quote_pos < 0:
                raise ValueError(f'Missing closing quote in: {s!r}. If you need a double-quote inside a string, use escaping: e.g. "the " char"')
            pos = closing_quote_pos + 1

def is_writable_file_like(obj):
    if False:
        print('Hello World!')
    'Return whether *obj* looks like a file object with a *write* method.'
    return callable(getattr(obj, 'write', None))

def file_requires_unicode(x):
    if False:
        i = 10
        return i + 15
    '\n    Return whether the given writable file-like object requires Unicode to be\n    written to it.\n    '
    try:
        x.write(b'')
    except TypeError:
        return True
    else:
        return False

def to_filehandle(fname, flag='r', return_opened=False, encoding=None):
    if False:
        return 10
    "\n    Convert a path to an open file handle or pass-through a file-like object.\n\n    Consider using `open_file_cm` instead, as it allows one to properly close\n    newly created file objects more easily.\n\n    Parameters\n    ----------\n    fname : str or path-like or file-like\n        If `str` or `os.PathLike`, the file is opened using the flags specified\n        by *flag* and *encoding*.  If a file-like object, it is passed through.\n    flag : str, default: 'r'\n        Passed as the *mode* argument to `open` when *fname* is `str` or\n        `os.PathLike`; ignored if *fname* is file-like.\n    return_opened : bool, default: False\n        If True, return both the file object and a boolean indicating whether\n        this was a new file (that the caller needs to close).  If False, return\n        only the new file.\n    encoding : str or None, default: None\n        Passed as the *mode* argument to `open` when *fname* is `str` or\n        `os.PathLike`; ignored if *fname* is file-like.\n\n    Returns\n    -------\n    fh : file-like\n    opened : bool\n        *opened* is only returned if *return_opened* is True.\n    "
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    if isinstance(fname, str):
        if fname.endswith('.gz'):
            fh = gzip.open(fname, flag)
        elif fname.endswith('.bz2'):
            import bz2
            fh = bz2.BZ2File(fname, flag)
        else:
            fh = open(fname, flag, encoding=encoding)
        opened = True
    elif hasattr(fname, 'seek'):
        fh = fname
        opened = False
    else:
        raise ValueError('fname must be a PathLike or file handle')
    if return_opened:
        return (fh, opened)
    return fh

def open_file_cm(path_or_file, mode='r', encoding=None):
    if False:
        return 10
    'Pass through file objects and context-manage path-likes.'
    (fh, opened) = to_filehandle(path_or_file, mode, True, encoding)
    return fh if opened else contextlib.nullcontext(fh)

def is_scalar_or_string(val):
    if False:
        i = 10
        return i + 15
    'Return whether the given object is a scalar or string like.'
    return isinstance(val, str) or not np.iterable(val)

@_api.delete_parameter('3.8', 'np_load', alternative='open(get_sample_data(..., asfileobj=False))')
def get_sample_data(fname, asfileobj=True, *, np_load=True):
    if False:
        i = 10
        return i + 15
    "\n    Return a sample data file.  *fname* is a path relative to the\n    :file:`mpl-data/sample_data` directory.  If *asfileobj* is `True`\n    return a file object, otherwise just a file path.\n\n    Sample data files are stored in the 'mpl-data/sample_data' directory within\n    the Matplotlib package.\n\n    If the filename ends in .gz, the file is implicitly ungzipped.  If the\n    filename ends with .npy or .npz, and *asfileobj* is `True`, the file is\n    loaded with `numpy.load`.\n    "
    path = _get_data_path('sample_data', fname)
    if asfileobj:
        suffix = path.suffix.lower()
        if suffix == '.gz':
            return gzip.open(path)
        elif suffix in ['.npy', '.npz']:
            if np_load:
                return np.load(path)
            else:
                return path.open('rb')
        elif suffix in ['.csv', '.xrc', '.txt']:
            return path.open('r')
        else:
            return path.open('rb')
    else:
        return str(path)

def _get_data_path(*args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the `pathlib.Path` to a resource file provided by Matplotlib.\n\n    ``*args`` specify a path relative to the base data path.\n    '
    return Path(matplotlib.get_data_path(), *args)

def flatten(seq, scalarp=is_scalar_or_string):
    if False:
        i = 10
        return i + 15
    "\n    Return a generator of flattened nested containers.\n\n    For example:\n\n        >>> from matplotlib.cbook import flatten\n        >>> l = (('John', ['Hunter']), (1, 23), [[([42, (5, 23)], )]])\n        >>> print(list(flatten(l)))\n        ['John', 'Hunter', 1, 23, 42, 5, 23]\n\n    By: Composite of Holger Krekel and Luther Blissett\n    From: https://code.activestate.com/recipes/121294/\n    and Recipe 1.12 in cookbook\n    "
    for item in seq:
        if scalarp(item) or item is None:
            yield item
        else:
            yield from flatten(item, scalarp)

@_api.deprecated('3.8')
class Stack:
    """
    Stack of elements with a movable cursor.

    Mimics home/back/forward in a web browser.
    """

    def __init__(self, default=None):
        if False:
            i = 10
            return i + 15
        self.clear()
        self._default = default

    def __call__(self):
        if False:
            return 10
        'Return the current element, or None.'
        if not self._elements:
            return self._default
        else:
            return self._elements[self._pos]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._elements)

    def __getitem__(self, ind):
        if False:
            for i in range(10):
                print('nop')
        return self._elements[ind]

    def forward(self):
        if False:
            i = 10
            return i + 15
        'Move the position forward and return the current element.'
        self._pos = min(self._pos + 1, len(self._elements) - 1)
        return self()

    def back(self):
        if False:
            for i in range(10):
                print('nop')
        'Move the position back and return the current element.'
        if self._pos > 0:
            self._pos -= 1
        return self()

    def push(self, o):
        if False:
            while True:
                i = 10
        '\n        Push *o* to the stack at current position.  Discard all later elements.\n\n        *o* is returned.\n        '
        self._elements = self._elements[:self._pos + 1] + [o]
        self._pos = len(self._elements) - 1
        return self()

    def home(self):
        if False:
            while True:
                i = 10
        '\n        Push the first element onto the top of the stack.\n\n        The first element is returned.\n        '
        if not self._elements:
            return
        self.push(self._elements[0])
        return self()

    def empty(self):
        if False:
            for i in range(10):
                print('nop')
        'Return whether the stack is empty.'
        return len(self._elements) == 0

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        'Empty the stack.'
        self._pos = -1
        self._elements = []

    def bubble(self, o):
        if False:
            while True:
                i = 10
        '\n        Raise all references of *o* to the top of the stack, and return it.\n\n        Raises\n        ------\n        ValueError\n            If *o* is not in the stack.\n        '
        if o not in self._elements:
            raise ValueError('Given element not contained in the stack')
        old_elements = self._elements.copy()
        self.clear()
        top_elements = []
        for elem in old_elements:
            if elem == o:
                top_elements.append(elem)
            else:
                self.push(elem)
        for _ in top_elements:
            self.push(o)
        return o

    def remove(self, o):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove *o* from the stack.\n\n        Raises\n        ------\n        ValueError\n            If *o* is not in the stack.\n        '
        if o not in self._elements:
            raise ValueError('Given element not contained in the stack')
        old_elements = self._elements.copy()
        self.clear()
        for elem in old_elements:
            if elem != o:
                self.push(elem)

class _Stack:
    """
    Stack of elements with a movable cursor.

    Mimics home/back/forward in a web browser.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._pos = -1
        self._elements = []

    def clear(self):
        if False:
            return 10
        'Empty the stack.'
        self._pos = -1
        self._elements = []

    def __call__(self):
        if False:
            return 10
        'Return the current element, or None.'
        return self._elements[self._pos] if self._elements else None

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._elements)

    def __getitem__(self, ind):
        if False:
            return 10
        return self._elements[ind]

    def forward(self):
        if False:
            print('Hello World!')
        'Move the position forward and return the current element.'
        self._pos = min(self._pos + 1, len(self._elements) - 1)
        return self()

    def back(self):
        if False:
            for i in range(10):
                print('nop')
        'Move the position back and return the current element.'
        self._pos = max(self._pos - 1, 0)
        return self()

    def push(self, o):
        if False:
            while True:
                i = 10
        '\n        Push *o* to the stack after the current position, and return *o*.\n\n        Discard all later elements.\n        '
        self._elements[self._pos + 1:] = [o]
        self._pos = len(self._elements) - 1
        return o

    def home(self):
        if False:
            return 10
        '\n        Push the first element onto the top of the stack.\n\n        The first element is returned.\n        '
        return self.push(self._elements[0]) if self._elements else None

def safe_masked_invalid(x, copy=False):
    if False:
        i = 10
        return i + 15
    x = np.array(x, subok=True, copy=copy)
    if not x.dtype.isnative:
        x = x.byteswap(inplace=copy).view(x.dtype.newbyteorder('N'))
    try:
        xm = np.ma.masked_where(~np.isfinite(x), x, copy=False)
    except TypeError:
        return x
    return xm

def print_cycles(objects, outstream=sys.stdout, show_progress=False):
    if False:
        i = 10
        return i + 15
    '\n    Print loops of cyclic references in the given *objects*.\n\n    It is often useful to pass in ``gc.garbage`` to find the cycles that are\n    preventing some objects from being garbage collected.\n\n    Parameters\n    ----------\n    objects\n        A list of objects to find cycles in.\n    outstream\n        The stream for output.\n    show_progress : bool\n        If True, print the number of objects reached as they are found.\n    '
    import gc

    def print_path(path):
        if False:
            for i in range(10):
                print('nop')
        for (i, step) in enumerate(path):
            next = path[(i + 1) % len(path)]
            outstream.write('   %s -- ' % type(step))
            if isinstance(step, dict):
                for (key, val) in step.items():
                    if val is next:
                        outstream.write(f'[{key!r}]')
                        break
                    if key is next:
                        outstream.write(f'[key] = {val!r}')
                        break
            elif isinstance(step, list):
                outstream.write('[%d]' % step.index(next))
            elif isinstance(step, tuple):
                outstream.write('( tuple )')
            else:
                outstream.write(repr(step))
            outstream.write(' ->\n')
        outstream.write('\n')

    def recurse(obj, start, all, current_path):
        if False:
            for i in range(10):
                print('nop')
        if show_progress:
            outstream.write('%d\r' % len(all))
        all[id(obj)] = None
        referents = gc.get_referents(obj)
        for referent in referents:
            if referent is start:
                print_path(current_path)
            elif referent is objects or isinstance(referent, types.FrameType):
                continue
            elif id(referent) not in all:
                recurse(referent, start, all, current_path + [obj])
    for obj in objects:
        outstream.write(f'Examining: {obj!r}\n')
        recurse(obj, obj, {}, [])

class Grouper:
    """
    A disjoint-set data structure.

    Objects can be joined using :meth:`join`, tested for connectedness
    using :meth:`joined`, and all disjoint sets can be retrieved by
    using the object as an iterator.

    The objects being joined must be hashable and weak-referenceable.

    Examples
    --------
    >>> from matplotlib.cbook import Grouper
    >>> class Foo:
    ...     def __init__(self, s):
    ...         self.s = s
    ...     def __repr__(self):
    ...         return self.s
    ...
    >>> a, b, c, d, e, f = [Foo(x) for x in 'abcdef']
    >>> grp = Grouper()
    >>> grp.join(a, b)
    >>> grp.join(b, c)
    >>> grp.join(d, e)
    >>> list(grp)
    [[a, b, c], [d, e]]
    >>> grp.joined(a, b)
    True
    >>> grp.joined(a, c)
    True
    >>> grp.joined(a, d)
    False
    """

    def __init__(self, init=()):
        if False:
            for i in range(10):
                print('nop')
        self._mapping = weakref.WeakKeyDictionary({x: weakref.WeakSet([x]) for x in init})

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return {**vars(self), '_mapping': {k: set(v) for (k, v) in self._mapping.items()}}

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        vars(self).update(state)
        self._mapping = weakref.WeakKeyDictionary({k: weakref.WeakSet(v) for (k, v) in self._mapping.items()})

    def __contains__(self, item):
        if False:
            print('Hello World!')
        return item in self._mapping

    @_api.deprecated('3.8', alternative='none, you no longer need to clean a Grouper')
    def clean(self):
        if False:
            i = 10
            return i + 15
        'Clean dead weak references from the dictionary.'

    def join(self, a, *args):
        if False:
            print('Hello World!')
        '\n        Join given arguments into the same set.  Accepts one or more arguments.\n        '
        mapping = self._mapping
        set_a = mapping.setdefault(a, weakref.WeakSet([a]))
        for arg in args:
            set_b = mapping.get(arg, weakref.WeakSet([arg]))
            if set_b is not set_a:
                if len(set_b) > len(set_a):
                    (set_a, set_b) = (set_b, set_a)
                set_a.update(set_b)
                for elem in set_b:
                    mapping[elem] = set_a

    def joined(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Return whether *a* and *b* are members of the same set.'
        return self._mapping.get(a, object()) is self._mapping.get(b)

    def remove(self, a):
        if False:
            return 10
        'Remove *a* from the grouper, doing nothing if it is not there.'
        set_a = self._mapping.pop(a, None)
        if set_a:
            set_a.remove(a)

    def __iter__(self):
        if False:
            while True:
                i = 10
        '\n        Iterate over each of the disjoint sets as a list.\n\n        The iterator is invalid if interleaved with calls to join().\n        '
        unique_groups = {id(group): group for group in self._mapping.values()}
        for group in unique_groups.values():
            yield [x for x in group]

    def get_siblings(self, a):
        if False:
            print('Hello World!')
        'Return all of the items joined with *a*, including itself.'
        siblings = self._mapping.get(a, [a])
        return [x for x in siblings]

class GrouperView:
    """Immutable view over a `.Grouper`."""

    def __init__(self, grouper):
        if False:
            while True:
                i = 10
        self._grouper = grouper

    def __contains__(self, item):
        if False:
            i = 10
            return i + 15
        return item in self._grouper

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._grouper)

    def joined(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        return self._grouper.joined(a, b)

    def get_siblings(self, a):
        if False:
            return 10
        return self._grouper.get_siblings(a)

def simple_linear_interpolation(a, steps):
    if False:
        while True:
            i = 10
    '\n    Resample an array with ``steps - 1`` points between original point pairs.\n\n    Along each column of *a*, ``(steps - 1)`` points are introduced between\n    each original values; the values are linearly interpolated.\n\n    Parameters\n    ----------\n    a : array, shape (n, ...)\n    steps : int\n\n    Returns\n    -------\n    array\n        shape ``((n - 1) * steps + 1, ...)``\n    '
    fps = a.reshape((len(a), -1))
    xp = np.arange(len(a)) * steps
    x = np.arange((len(a) - 1) * steps + 1)
    return np.column_stack([np.interp(x, xp, fp) for fp in fps.T]).reshape((len(x),) + a.shape[1:])

def delete_masked_points(*args):
    if False:
        i = 10
        return i + 15
    '\n    Find all masked and/or non-finite points in a set of arguments,\n    and return the arguments with only the unmasked points remaining.\n\n    Arguments can be in any of 5 categories:\n\n    1) 1-D masked arrays\n    2) 1-D ndarrays\n    3) ndarrays with more than one dimension\n    4) other non-string iterables\n    5) anything else\n\n    The first argument must be in one of the first four categories;\n    any argument with a length differing from that of the first\n    argument (and hence anything in category 5) then will be\n    passed through unchanged.\n\n    Masks are obtained from all arguments of the correct length\n    in categories 1, 2, and 4; a point is bad if masked in a masked\n    array or if it is a nan or inf.  No attempt is made to\n    extract a mask from categories 2, 3, and 4 if `numpy.isfinite`\n    does not yield a Boolean array.\n\n    All input arguments that are not passed unchanged are returned\n    as ndarrays after removing the points or rows corresponding to\n    masks in any of the arguments.\n\n    A vastly simpler version of this function was originally\n    written as a helper for Axes.scatter().\n\n    '
    if not len(args):
        return ()
    if is_scalar_or_string(args[0]):
        raise ValueError('First argument must be a sequence')
    nrecs = len(args[0])
    margs = []
    seqlist = [False] * len(args)
    for (i, x) in enumerate(args):
        if not isinstance(x, str) and np.iterable(x) and (len(x) == nrecs):
            seqlist[i] = True
            if isinstance(x, np.ma.MaskedArray):
                if x.ndim > 1:
                    raise ValueError('Masked arrays must be 1-D')
            else:
                x = np.asarray(x)
        margs.append(x)
    masks = []
    for (i, x) in enumerate(margs):
        if seqlist[i]:
            if x.ndim > 1:
                continue
            if isinstance(x, np.ma.MaskedArray):
                masks.append(~np.ma.getmaskarray(x))
                xd = x.data
            else:
                xd = x
            try:
                mask = np.isfinite(xd)
                if isinstance(mask, np.ndarray):
                    masks.append(mask)
            except Exception:
                pass
    if len(masks):
        mask = np.logical_and.reduce(masks)
        igood = mask.nonzero()[0]
        if len(igood) < nrecs:
            for (i, x) in enumerate(margs):
                if seqlist[i]:
                    margs[i] = x[igood]
    for (i, x) in enumerate(margs):
        if seqlist[i] and isinstance(x, np.ma.MaskedArray):
            margs[i] = x.filled()
    return margs

def _combine_masks(*args):
    if False:
        i = 10
        return i + 15
    '\n    Find all masked and/or non-finite points in a set of arguments,\n    and return the arguments as masked arrays with a common mask.\n\n    Arguments can be in any of 5 categories:\n\n    1) 1-D masked arrays\n    2) 1-D ndarrays\n    3) ndarrays with more than one dimension\n    4) other non-string iterables\n    5) anything else\n\n    The first argument must be in one of the first four categories;\n    any argument with a length differing from that of the first\n    argument (and hence anything in category 5) then will be\n    passed through unchanged.\n\n    Masks are obtained from all arguments of the correct length\n    in categories 1, 2, and 4; a point is bad if masked in a masked\n    array or if it is a nan or inf.  No attempt is made to\n    extract a mask from categories 2 and 4 if `numpy.isfinite`\n    does not yield a Boolean array.  Category 3 is included to\n    support RGB or RGBA ndarrays, which are assumed to have only\n    valid values and which are passed through unchanged.\n\n    All input arguments that are not passed unchanged are returned\n    as masked arrays if any masked points are found, otherwise as\n    ndarrays.\n\n    '
    if not len(args):
        return ()
    if is_scalar_or_string(args[0]):
        raise ValueError('First argument must be a sequence')
    nrecs = len(args[0])
    margs = []
    seqlist = [False] * len(args)
    masks = []
    for (i, x) in enumerate(args):
        if is_scalar_or_string(x) or len(x) != nrecs:
            margs.append(x)
        else:
            if isinstance(x, np.ma.MaskedArray) and x.ndim > 1:
                raise ValueError('Masked arrays must be 1-D')
            try:
                x = np.asanyarray(x)
            except (VisibleDeprecationWarning, ValueError):
                x = np.asanyarray(x, dtype=object)
            if x.ndim == 1:
                x = safe_masked_invalid(x)
                seqlist[i] = True
                if np.ma.is_masked(x):
                    masks.append(np.ma.getmaskarray(x))
            margs.append(x)
    if len(masks):
        mask = np.logical_or.reduce(masks)
        for (i, x) in enumerate(margs):
            if seqlist[i]:
                margs[i] = np.ma.array(x, mask=mask)
    return margs

def _broadcast_with_masks(*args, compress=False):
    if False:
        return 10
    '\n    Broadcast inputs, combining all masked arrays.\n\n    Parameters\n    ----------\n    *args : array-like\n        The inputs to broadcast.\n    compress : bool, default: False\n        Whether to compress the masked arrays. If False, the masked values\n        are replaced by NaNs.\n\n    Returns\n    -------\n    list of array-like\n        The broadcasted and masked inputs.\n    '
    masks = [k.mask for k in args if isinstance(k, np.ma.MaskedArray)]
    bcast = np.broadcast_arrays(*args, *masks)
    inputs = bcast[:len(args)]
    masks = bcast[len(args):]
    if masks:
        mask = np.logical_or.reduce(masks)
        if compress:
            inputs = [np.ma.array(k, mask=mask).compressed() for k in inputs]
        else:
            inputs = [np.ma.array(k, mask=mask, dtype=float).filled(np.nan).ravel() for k in inputs]
    else:
        inputs = [np.ravel(k) for k in inputs]
    return inputs

def boxplot_stats(X, whis=1.5, bootstrap=None, labels=None, autorange=False):
    if False:
        print('Hello World!')
    '\n    Return a list of dictionaries of statistics used to draw a series of box\n    and whisker plots using `~.Axes.bxp`.\n\n    Parameters\n    ----------\n    X : array-like\n        Data that will be represented in the boxplots. Should have 2 or\n        fewer dimensions.\n\n    whis : float or (float, float), default: 1.5\n        The position of the whiskers.\n\n        If a float, the lower whisker is at the lowest datum above\n        ``Q1 - whis*(Q3-Q1)``, and the upper whisker at the highest datum below\n        ``Q3 + whis*(Q3-Q1)``, where Q1 and Q3 are the first and third\n        quartiles.  The default value of ``whis = 1.5`` corresponds to Tukey\'s\n        original definition of boxplots.\n\n        If a pair of floats, they indicate the percentiles at which to draw the\n        whiskers (e.g., (5, 95)).  In particular, setting this to (0, 100)\n        results in whiskers covering the whole range of the data.\n\n        In the edge case where ``Q1 == Q3``, *whis* is automatically set to\n        (0, 100) (cover the whole range of the data) if *autorange* is True.\n\n        Beyond the whiskers, data are considered outliers and are plotted as\n        individual points.\n\n    bootstrap : int, optional\n        Number of times the confidence intervals around the median\n        should be bootstrapped (percentile method).\n\n    labels : array-like, optional\n        Labels for each dataset. Length must be compatible with\n        dimensions of *X*.\n\n    autorange : bool, optional (False)\n        When `True` and the data are distributed such that the 25th and 75th\n        percentiles are equal, ``whis`` is set to (0, 100) such that the\n        whisker ends are at the minimum and maximum of the data.\n\n    Returns\n    -------\n    list of dict\n        A list of dictionaries containing the results for each column\n        of data. Keys of each dictionary are the following:\n\n        ========   ===================================\n        Key        Value Description\n        ========   ===================================\n        label      tick label for the boxplot\n        mean       arithmetic mean value\n        med        50th percentile\n        q1         first quartile (25th percentile)\n        q3         third quartile (75th percentile)\n        iqr        interquartile range\n        cilo       lower notch around the median\n        cihi       upper notch around the median\n        whislo     end of the lower whisker\n        whishi     end of the upper whisker\n        fliers     outliers\n        ========   ===================================\n\n    Notes\n    -----\n    Non-bootstrapping approach to confidence interval uses Gaussian-based\n    asymptotic approximation:\n\n    .. math::\n\n        \\mathrm{med} \\pm 1.57 \\times \\frac{\\mathrm{iqr}}{\\sqrt{N}}\n\n    General approach from:\n    McGill, R., Tukey, J.W., and Larsen, W.A. (1978) "Variations of\n    Boxplots", The American Statistician, 32:12-16.\n    '

    def _bootstrap_median(data, N=5000):
        if False:
            for i in range(10):
                print('nop')
        M = len(data)
        percentiles = [2.5, 97.5]
        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
        estimate = np.median(bsData, axis=1, overwrite_input=True)
        CI = np.percentile(estimate, percentiles)
        return CI

    def _compute_conf_interval(data, med, iqr, bootstrap):
        if False:
            i = 10
            return i + 15
        if bootstrap is not None:
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:
            N = len(data)
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)
        return (notch_min, notch_max)
    bxpstats = []
    X = _reshape_2D(X, 'X')
    ncols = len(X)
    if labels is None:
        labels = itertools.repeat(None)
    elif len(labels) != ncols:
        raise ValueError('Dimensions of labels and X must be compatible')
    input_whis = whis
    for (ii, (x, label)) in enumerate(zip(X, labels)):
        stats = {}
        if label is not None:
            stats['label'] = label
        whis = input_whis
        bxpstats.append(stats)
        if len(x) == 0:
            stats['fliers'] = np.array([])
            stats['mean'] = np.nan
            stats['med'] = np.nan
            stats['q1'] = np.nan
            stats['q3'] = np.nan
            stats['iqr'] = np.nan
            stats['cilo'] = np.nan
            stats['cihi'] = np.nan
            stats['whislo'] = np.nan
            stats['whishi'] = np.nan
            continue
        x = np.asarray(x)
        stats['mean'] = np.mean(x)
        (q1, med, q3) = np.percentile(x, [25, 50, 75])
        stats['iqr'] = q3 - q1
        if stats['iqr'] == 0 and autorange:
            whis = (0, 100)
        (stats['cilo'], stats['cihi']) = _compute_conf_interval(x, med, stats['iqr'], bootstrap)
        if np.iterable(whis) and (not isinstance(whis, str)):
            (loval, hival) = np.percentile(x, whis)
        elif np.isreal(whis):
            loval = q1 - whis * stats['iqr']
            hival = q3 + whis * stats['iqr']
        else:
            raise ValueError('whis must be a float or list of percentiles')
        wiskhi = x[x <= hival]
        if len(wiskhi) == 0 or np.max(wiskhi) < q3:
            stats['whishi'] = q3
        else:
            stats['whishi'] = np.max(wiskhi)
        wisklo = x[x >= loval]
        if len(wisklo) == 0 or np.min(wisklo) > q1:
            stats['whislo'] = q1
        else:
            stats['whislo'] = np.min(wisklo)
        stats['fliers'] = np.concatenate([x[x < stats['whislo']], x[x > stats['whishi']]])
        (stats['q1'], stats['med'], stats['q3']) = (q1, med, q3)
    return bxpstats
ls_mapper = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
ls_mapper_r = {v: k for (k, v) in ls_mapper.items()}

def contiguous_regions(mask):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of (ind0, ind1) such that ``mask[ind0:ind1].all()`` is\n    True and we cover all such regions.\n    '
    mask = np.asarray(mask, dtype=bool)
    if not mask.size:
        return []
    (idx,) = np.nonzero(mask[:-1] != mask[1:])
    idx += 1
    idx = idx.tolist()
    if mask[0]:
        idx = [0] + idx
    if mask[-1]:
        idx.append(len(mask))
    return list(zip(idx[::2], idx[1::2]))

def is_math_text(s):
    if False:
        return 10
    '\n    Return whether the string *s* contains math expressions.\n\n    This is done by checking whether *s* contains an even number of\n    non-escaped dollar signs.\n    '
    s = str(s)
    dollar_count = s.count('$') - s.count('\\$')
    even_dollars = dollar_count > 0 and dollar_count % 2 == 0
    return even_dollars

def _to_unmasked_float_array(x):
    if False:
        return 10
    '\n    Convert a sequence to a float array; if input was a masked array, masked\n    values are converted to nans.\n    '
    if hasattr(x, 'mask'):
        return np.ma.asarray(x, float).filled(np.nan)
    else:
        return np.asarray(x, float)

def _check_1d(x):
    if False:
        i = 10
        return i + 15
    'Convert scalars to 1D arrays; pass-through arrays as is.'
    x = _unpack_to_numpy(x)
    if not hasattr(x, 'shape') or not hasattr(x, 'ndim') or len(x.shape) < 1:
        return np.atleast_1d(x)
    else:
        return x

def _reshape_2D(X, name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Use Fortran ordering to convert ndarrays and lists of iterables to lists of\n    1D arrays.\n\n    Lists of iterables are converted by applying `numpy.asanyarray` to each of\n    their elements.  1D ndarrays are returned in a singleton list containing\n    them.  2D ndarrays are converted to the list of their *columns*.\n\n    *name* is used to generate the error message for invalid inputs.\n    '
    X = _unpack_to_numpy(X)
    if isinstance(X, np.ndarray):
        X = X.T
        if len(X) == 0:
            return [[]]
        elif X.ndim == 1 and np.ndim(X[0]) == 0:
            return [X]
        elif X.ndim in [1, 2]:
            return [np.reshape(x, -1) for x in X]
        else:
            raise ValueError(f'{name} must have 2 or fewer dimensions')
    if len(X) == 0:
        return [[]]
    result = []
    is_1d = True
    for xi in X:
        if not isinstance(xi, str):
            try:
                iter(xi)
            except TypeError:
                pass
            else:
                is_1d = False
        xi = np.asanyarray(xi)
        nd = np.ndim(xi)
        if nd > 1:
            raise ValueError(f'{name} must have 2 or fewer dimensions')
        result.append(xi.reshape(-1))
    if is_1d:
        return [np.reshape(result, -1)]
    else:
        return result

def violin_stats(X, method, points=100, quantiles=None):
    if False:
        i = 10
        return i + 15
    '\n    Return a list of dictionaries of data which can be used to draw a series\n    of violin plots.\n\n    See the ``Returns`` section below to view the required keys of the\n    dictionary.\n\n    Users can skip this function and pass a user-defined set of dictionaries\n    with the same keys to `~.axes.Axes.violinplot` instead of using Matplotlib\n    to do the calculations. See the *Returns* section below for the keys\n    that must be present in the dictionaries.\n\n    Parameters\n    ----------\n    X : array-like\n        Sample data that will be used to produce the gaussian kernel density\n        estimates. Must have 2 or fewer dimensions.\n\n    method : callable\n        The method used to calculate the kernel density estimate for each\n        column of data. When called via ``method(v, coords)``, it should\n        return a vector of the values of the KDE evaluated at the values\n        specified in coords.\n\n    points : int, default: 100\n        Defines the number of points to evaluate each of the gaussian kernel\n        density estimates at.\n\n    quantiles : array-like, default: None\n        Defines (if not None) a list of floats in interval [0, 1] for each\n        column of data, which represents the quantiles that will be rendered\n        for that column of data. Must have 2 or fewer dimensions. 1D array will\n        be treated as a singleton list containing them.\n\n    Returns\n    -------\n    list of dict\n        A list of dictionaries containing the results for each column of data.\n        The dictionaries contain at least the following:\n\n        - coords: A list of scalars containing the coordinates this particular\n          kernel density estimate was evaluated at.\n        - vals: A list of scalars containing the values of the kernel density\n          estimate at each of the coordinates given in *coords*.\n        - mean: The mean value for this column of data.\n        - median: The median value for this column of data.\n        - min: The minimum value for this column of data.\n        - max: The maximum value for this column of data.\n        - quantiles: The quantile values for this column of data.\n    '
    vpstats = []
    X = _reshape_2D(X, 'X')
    if quantiles is not None and len(quantiles) != 0:
        quantiles = _reshape_2D(quantiles, 'quantiles')
    else:
        quantiles = [[]] * len(X)
    if len(X) != len(quantiles):
        raise ValueError('List of violinplot statistics and quantiles values must have the same length')
    for (x, q) in zip(X, quantiles):
        stats = {}
        min_val = np.min(x)
        max_val = np.max(x)
        quantile_val = np.percentile(x, 100 * q)
        coords = np.linspace(min_val, max_val, points)
        stats['vals'] = method(x, coords)
        stats['coords'] = coords
        stats['mean'] = np.mean(x)
        stats['median'] = np.median(x)
        stats['min'] = min_val
        stats['max'] = max_val
        stats['quantiles'] = np.atleast_1d(quantile_val)
        vpstats.append(stats)
    return vpstats

def pts_to_prestep(x, *args):
    if False:
        return 10
    '\n    Convert continuous line to pre-steps.\n\n    Given a set of ``N`` points, convert to ``2N - 1`` points, which when\n    connected linearly give a step function which changes values at the\n    beginning of the intervals.\n\n    Parameters\n    ----------\n    x : array\n        The x location of the steps. May be empty.\n\n    y1, ..., yp : array\n        y arrays to be turned into steps; all must be the same length as ``x``.\n\n    Returns\n    -------\n    array\n        The x and y values converted to steps in the same order as the input;\n        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is\n        length ``N``, each of these arrays will be length ``2N + 1``. For\n        ``N=0``, the length will be 0.\n\n    Examples\n    --------\n    >>> x_s, y1_s, y2_s = pts_to_prestep(x, y1, y2)\n    '
    steps = np.zeros((1 + len(args), max(2 * len(x) - 1, 0)))
    steps[0, 0::2] = x
    steps[0, 1::2] = steps[0, 0:-2:2]
    steps[1:, 0::2] = args
    steps[1:, 1::2] = steps[1:, 2::2]
    return steps

def pts_to_poststep(x, *args):
    if False:
        while True:
            i = 10
    '\n    Convert continuous line to post-steps.\n\n    Given a set of ``N`` points convert to ``2N + 1`` points, which when\n    connected linearly give a step function which changes values at the end of\n    the intervals.\n\n    Parameters\n    ----------\n    x : array\n        The x location of the steps. May be empty.\n\n    y1, ..., yp : array\n        y arrays to be turned into steps; all must be the same length as ``x``.\n\n    Returns\n    -------\n    array\n        The x and y values converted to steps in the same order as the input;\n        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is\n        length ``N``, each of these arrays will be length ``2N + 1``. For\n        ``N=0``, the length will be 0.\n\n    Examples\n    --------\n    >>> x_s, y1_s, y2_s = pts_to_poststep(x, y1, y2)\n    '
    steps = np.zeros((1 + len(args), max(2 * len(x) - 1, 0)))
    steps[0, 0::2] = x
    steps[0, 1::2] = steps[0, 2::2]
    steps[1:, 0::2] = args
    steps[1:, 1::2] = steps[1:, 0:-2:2]
    return steps

def pts_to_midstep(x, *args):
    if False:
        print('Hello World!')
    '\n    Convert continuous line to mid-steps.\n\n    Given a set of ``N`` points convert to ``2N`` points which when connected\n    linearly give a step function which changes values at the middle of the\n    intervals.\n\n    Parameters\n    ----------\n    x : array\n        The x location of the steps. May be empty.\n\n    y1, ..., yp : array\n        y arrays to be turned into steps; all must be the same length as\n        ``x``.\n\n    Returns\n    -------\n    array\n        The x and y values converted to steps in the same order as the input;\n        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is\n        length ``N``, each of these arrays will be length ``2N``.\n\n    Examples\n    --------\n    >>> x_s, y1_s, y2_s = pts_to_midstep(x, y1, y2)\n    '
    steps = np.zeros((1 + len(args), 2 * len(x)))
    x = np.asanyarray(x)
    steps[0, 1:-1:2] = steps[0, 2::2] = (x[:-1] + x[1:]) / 2
    steps[0, :1] = x[:1]
    steps[0, -1:] = x[-1:]
    steps[1:, 0::2] = args
    steps[1:, 1::2] = steps[1:, 0::2]
    return steps
STEP_LOOKUP_MAP = {'default': lambda x, y: (x, y), 'steps': pts_to_prestep, 'steps-pre': pts_to_prestep, 'steps-post': pts_to_poststep, 'steps-mid': pts_to_midstep}

def index_of(y):
    if False:
        i = 10
        return i + 15
    '\n    A helper function to create reasonable x values for the given *y*.\n\n    This is used for plotting (x, y) if x values are not explicitly given.\n\n    First try ``y.index`` (assuming *y* is a `pandas.Series`), if that\n    fails, use ``range(len(y))``.\n\n    This will be extended in the future to deal with more types of\n    labeled data.\n\n    Parameters\n    ----------\n    y : float or array-like\n\n    Returns\n    -------\n    x, y : ndarray\n       The x and y values to plot.\n    '
    try:
        return (y.index.to_numpy(), y.to_numpy())
    except AttributeError:
        pass
    try:
        y = _check_1d(y)
    except (VisibleDeprecationWarning, ValueError):
        pass
    else:
        return (np.arange(y.shape[0], dtype=float), y)
    raise ValueError('Input could not be cast to an at-least-1D NumPy array')

def safe_first_element(obj):
    if False:
        i = 10
        return i + 15
    '\n    Return the first element in *obj*.\n\n    This is a type-independent way of obtaining the first element,\n    supporting both index access and the iterator protocol.\n    '
    if isinstance(obj, collections.abc.Iterator):
        try:
            return obj[0]
        except TypeError:
            pass
        raise RuntimeError('matplotlib does not support generators as input')
    return next(iter(obj))

def _safe_first_finite(obj):
    if False:
        print('Hello World!')
    '\n    Return the first finite element in *obj* if one is available and skip_nonfinite is\n    True. Otherwise, return the first element.\n\n    This is a method for internal use.\n\n    This is a type-independent way of obtaining the first finite element, supporting\n    both index access and the iterator protocol.\n    '

    def safe_isfinite(val):
        if False:
            while True:
                i = 10
        if val is None:
            return False
        try:
            return math.isfinite(val)
        except (TypeError, ValueError):
            pass
        try:
            return np.isfinite(val) if np.isscalar(val) else True
        except TypeError:
            return True
    if isinstance(obj, np.flatiter):
        return obj[0]
    elif isinstance(obj, collections.abc.Iterator):
        raise RuntimeError('matplotlib does not support generators as input')
    else:
        for val in obj:
            if safe_isfinite(val):
                return val
        return safe_first_element(obj)

def sanitize_sequence(data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert dictview objects to list. Other inputs are returned unchanged.\n    '
    return list(data) if isinstance(data, collections.abc.MappingView) else data

def normalize_kwargs(kw, alias_mapping=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to normalize kwarg inputs.\n\n    Parameters\n    ----------\n    kw : dict or None\n        A dict of keyword arguments.  None is explicitly supported and treated\n        as an empty dict, to support functions with an optional parameter of\n        the form ``props=None``.\n\n    alias_mapping : dict or Artist subclass or Artist instance, optional\n        A mapping between a canonical name to a list of aliases, in order of\n        precedence from lowest to highest.\n\n        If the canonical value is not in the list it is assumed to have the\n        highest priority.\n\n        If an Artist subclass or instance is passed, use its properties alias\n        mapping.\n\n    Raises\n    ------\n    TypeError\n        To match what Python raises if invalid arguments/keyword arguments are\n        passed to a callable.\n    '
    from matplotlib.artist import Artist
    if kw is None:
        return {}
    if alias_mapping is None:
        alias_mapping = {}
    elif isinstance(alias_mapping, type) and issubclass(alias_mapping, Artist) or isinstance(alias_mapping, Artist):
        alias_mapping = getattr(alias_mapping, '_alias_map', {})
    to_canonical = {alias: canonical for (canonical, alias_list) in alias_mapping.items() for alias in alias_list}
    canonical_to_seen = {}
    ret = {}
    for (k, v) in kw.items():
        canonical = to_canonical.get(k, k)
        if canonical in canonical_to_seen:
            raise TypeError(f'Got both {canonical_to_seen[canonical]!r} and {k!r}, which are aliases of one another')
        canonical_to_seen[canonical] = k
        ret[canonical] = v
    return ret

@contextlib.contextmanager
def _lock_path(path):
    if False:
        while True:
            i = 10
    '\n    Context manager for locking a path.\n\n    Usage::\n\n        with _lock_path(path):\n            ...\n\n    Another thread or process that attempts to lock the same path will wait\n    until this context manager is exited.\n\n    The lock is implemented by creating a temporary file in the parent\n    directory, so that directory must exist and be writable.\n    '
    path = Path(path)
    lock_path = path.with_name(path.name + '.matplotlib-lock')
    retries = 50
    sleeptime = 0.1
    for _ in range(retries):
        try:
            with lock_path.open('xb'):
                break
        except FileExistsError:
            time.sleep(sleeptime)
    else:
        raise TimeoutError('Lock error: Matplotlib failed to acquire the following lock file:\n    {}\nThis maybe due to another process holding this lock file.  If you are sure no\nother Matplotlib process is running, remove this file and try again.'.format(lock_path))
    try:
        yield
    finally:
        lock_path.unlink()

def _topmost_artist(artists, _cached_max=functools.partial(max, key=operator.attrgetter('zorder'))):
    if False:
        i = 10
        return i + 15
    '\n    Get the topmost artist of a list.\n\n    In case of a tie, return the *last* of the tied artists, as it will be\n    drawn on top of the others. `max` returns the first maximum in case of\n    ties, so we need to iterate over the list in reverse order.\n    '
    return _cached_max(reversed(artists))

def _str_equal(obj, s):
    if False:
        i = 10
        return i + 15
    '\n    Return whether *obj* is a string equal to string *s*.\n\n    This helper solely exists to handle the case where *obj* is a numpy array,\n    because in such cases, a naive ``obj == s`` would yield an array, which\n    cannot be used in a boolean context.\n    '
    return isinstance(obj, str) and obj == s

def _str_lower_equal(obj, s):
    if False:
        return 10
    '\n    Return whether *obj* is a string equal, when lowercased, to string *s*.\n\n    This helper solely exists to handle the case where *obj* is a numpy array,\n    because in such cases, a naive ``obj == s`` would yield an array, which\n    cannot be used in a boolean context.\n    '
    return isinstance(obj, str) and obj.lower() == s

def _array_perimeter(arr):
    if False:
        while True:
            i = 10
    '\n    Get the elements on the perimeter of *arr*.\n\n    Parameters\n    ----------\n    arr : ndarray, shape (M, N)\n        The input array.\n\n    Returns\n    -------\n    ndarray, shape (2*(M - 1) + 2*(N - 1),)\n        The elements on the perimeter of the array::\n\n           [arr[0, 0], ..., arr[0, -1], ..., arr[-1, -1], ..., arr[-1, 0], ...]\n\n    Examples\n    --------\n    >>> i, j = np.ogrid[:3, :4]\n    >>> a = i*10 + j\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [10, 11, 12, 13],\n           [20, 21, 22, 23]])\n    >>> _array_perimeter(a)\n    array([ 0,  1,  2,  3, 13, 23, 22, 21, 20, 10])\n    '
    forward = np.s_[0:-1]
    backward = np.s_[-1:0:-1]
    return np.concatenate((arr[0, forward], arr[forward, -1], arr[-1, backward], arr[backward, 0]))

def _unfold(arr, axis, size, step):
    if False:
        return 10
    '\n    Append an extra dimension containing sliding windows along *axis*.\n\n    All windows are of size *size* and begin with every *step* elements.\n\n    Parameters\n    ----------\n    arr : ndarray, shape (N_1, ..., N_k)\n        The input array\n    axis : int\n        Axis along which the windows are extracted\n    size : int\n        Size of the windows\n    step : int\n        Stride between first elements of subsequent windows.\n\n    Returns\n    -------\n    ndarray, shape (N_1, ..., 1 + (N_axis-size)/step, ..., N_k, size)\n\n    Examples\n    --------\n    >>> i, j = np.ogrid[:3, :7]\n    >>> a = i*10 + j\n    >>> a\n    array([[ 0,  1,  2,  3,  4,  5,  6],\n           [10, 11, 12, 13, 14, 15, 16],\n           [20, 21, 22, 23, 24, 25, 26]])\n    >>> _unfold(a, axis=1, size=3, step=2)\n    array([[[ 0,  1,  2],\n            [ 2,  3,  4],\n            [ 4,  5,  6]],\n           [[10, 11, 12],\n            [12, 13, 14],\n            [14, 15, 16]],\n           [[20, 21, 22],\n            [22, 23, 24],\n            [24, 25, 26]]])\n    '
    new_shape = [*arr.shape, size]
    new_strides = [*arr.strides, arr.strides[axis]]
    new_shape[axis] = (new_shape[axis] - size) // step + 1
    new_strides[axis] = new_strides[axis] * step
    return np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides, writeable=False)

def _array_patch_perimeters(x, rstride, cstride):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract perimeters of patches from *arr*.\n\n    Extracted patches are of size (*rstride* + 1) x (*cstride* + 1) and\n    share perimeters with their neighbors. The ordering of the vertices matches\n    that returned by ``_array_perimeter``.\n\n    Parameters\n    ----------\n    x : ndarray, shape (N, M)\n        Input array\n    rstride : int\n        Vertical (row) stride between corresponding elements of each patch\n    cstride : int\n        Horizontal (column) stride between corresponding elements of each patch\n\n    Returns\n    -------\n    ndarray, shape (N/rstride * M/cstride, 2 * (rstride + cstride))\n    '
    assert rstride > 0 and cstride > 0
    assert (x.shape[0] - 1) % rstride == 0
    assert (x.shape[1] - 1) % cstride == 0
    top = _unfold(x[:-1:rstride, :-1], 1, cstride, cstride)
    bottom = _unfold(x[rstride::rstride, 1:], 1, cstride, cstride)[..., ::-1]
    right = _unfold(x[:-1, cstride::cstride], 0, rstride, rstride)
    left = _unfold(x[1:, :-1:cstride], 0, rstride, rstride)[..., ::-1]
    return np.concatenate((top, right, bottom, left), axis=2).reshape(-1, 2 * (rstride + cstride))

@contextlib.contextmanager
def _setattr_cm(obj, **kwargs):
    if False:
        print('Hello World!')
    '\n    Temporarily set some attributes; restore original state at context exit.\n    '
    sentinel = object()
    origs = {}
    for attr in kwargs:
        orig = getattr(obj, attr, sentinel)
        if attr in obj.__dict__ or orig is sentinel:
            origs[attr] = orig
        else:
            cls_orig = getattr(type(obj), attr)
            if isinstance(cls_orig, property):
                origs[attr] = orig
            else:
                origs[attr] = sentinel
    try:
        for (attr, val) in kwargs.items():
            setattr(obj, attr, val)
        yield
    finally:
        for (attr, orig) in origs.items():
            if orig is sentinel:
                delattr(obj, attr)
            else:
                setattr(obj, attr, orig)

class _OrderedSet(collections.abc.MutableSet):

    def __init__(self):
        if False:
            print('Hello World!')
        self._od = collections.OrderedDict()

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        return key in self._od

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._od)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._od)

    def add(self, key):
        if False:
            return 10
        self._od.pop(key, None)
        self._od[key] = None

    def discard(self, key):
        if False:
            for i in range(10):
                print('nop')
        self._od.pop(key, None)

def _premultiplied_argb32_to_unmultiplied_rgba8888(buf):
    if False:
        return 10
    '\n    Convert a premultiplied ARGB32 buffer to an unmultiplied RGBA8888 buffer.\n    '
    rgba = np.take(buf, [2, 1, 0, 3] if sys.byteorder == 'little' else [1, 2, 3, 0], axis=2)
    rgb = rgba[..., :-1]
    alpha = rgba[..., -1]
    mask = alpha != 0
    for channel in np.rollaxis(rgb, -1):
        channel[mask] = (channel[mask].astype(int) * 255 + alpha[mask] // 2) // alpha[mask]
    return rgba

def _unmultiplied_rgba8888_to_premultiplied_argb32(rgba8888):
    if False:
        print('Hello World!')
    '\n    Convert an unmultiplied RGBA8888 buffer to a premultiplied ARGB32 buffer.\n    '
    if sys.byteorder == 'little':
        argb32 = np.take(rgba8888, [2, 1, 0, 3], axis=2)
        rgb24 = argb32[..., :-1]
        alpha8 = argb32[..., -1:]
    else:
        argb32 = np.take(rgba8888, [3, 0, 1, 2], axis=2)
        alpha8 = argb32[..., :1]
        rgb24 = argb32[..., 1:]
    if alpha8.min() != 255:
        np.multiply(rgb24, alpha8 / 255, out=rgb24, casting='unsafe')
    return argb32

def _get_nonzero_slices(buf):
    if False:
        print('Hello World!')
    '\n    Return the bounds of the nonzero region of a 2D array as a pair of slices.\n\n    ``buf[_get_nonzero_slices(buf)]`` is the smallest sub-rectangle in *buf*\n    that encloses all non-zero entries in *buf*.  If *buf* is fully zero, then\n    ``(slice(0, 0), slice(0, 0))`` is returned.\n    '
    (x_nz,) = buf.any(axis=0).nonzero()
    (y_nz,) = buf.any(axis=1).nonzero()
    if len(x_nz) and len(y_nz):
        (l, r) = x_nz[[0, -1]]
        (b, t) = y_nz[[0, -1]]
        return (slice(b, t + 1), slice(l, r + 1))
    else:
        return (slice(0, 0), slice(0, 0))

def _pformat_subprocess(command):
    if False:
        i = 10
        return i + 15
    'Pretty-format a subprocess command for printing/logging purposes.'
    return command if isinstance(command, str) else ' '.join((shlex.quote(os.fspath(arg)) for arg in command))

def _check_and_log_subprocess(command, logger, **kwargs):
    if False:
        return 10
    '\n    Run *command*, returning its stdout output if it succeeds.\n\n    If it fails (exits with nonzero return code), raise an exception whose text\n    includes the failed command and captured stdout and stderr output.\n\n    Regardless of the return code, the command is logged at DEBUG level on\n    *logger*.  In case of success, the output is likewise logged.\n    '
    logger.debug('%s', _pformat_subprocess(command))
    proc = subprocess.run(command, capture_output=True, **kwargs)
    if proc.returncode:
        stdout = proc.stdout
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        stderr = proc.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode()
        raise RuntimeError(f'The command\n    {_pformat_subprocess(command)}\nfailed and generated the following output:\n{stdout}\nand the following error:\n{stderr}')
    if proc.stdout:
        logger.debug('stdout:\n%s', proc.stdout)
    if proc.stderr:
        logger.debug('stderr:\n%s', proc.stderr)
    return proc.stdout

def _backend_module_name(name):
    if False:
        print('Hello World!')
    '\n    Convert a backend name (either a standard backend -- "Agg", "TkAgg", ... --\n    or a custom backend -- "module://...") to the corresponding module name).\n    '
    return name[9:] if name.startswith('module://') else f'matplotlib.backends.backend_{name.lower()}'

def _setup_new_guiapp():
    if False:
        while True:
            i = 10
    '\n    Perform OS-dependent setup when Matplotlib creates a new GUI application.\n    '
    try:
        _c_internal_utils.Win32_GetCurrentProcessExplicitAppUserModelID()
    except OSError:
        _c_internal_utils.Win32_SetCurrentProcessExplicitAppUserModelID('matplotlib')

def _format_approx(number, precision):
    if False:
        for i in range(10):
            print('nop')
    '\n    Format the number with at most the number of decimals given as precision.\n    Remove trailing zeros and possibly the decimal point.\n    '
    return f'{number:.{precision}f}'.rstrip('0').rstrip('.') or '0'

def _g_sig_digits(value, delta):
    if False:
        while True:
            i = 10
    '\n    Return the number of significant digits to %g-format *value*, assuming that\n    it is known with an error of *delta*.\n    '
    if delta == 0:
        delta = abs(np.spacing(value))
    return max(0, (math.floor(math.log10(abs(value))) + 1 if value else 1) - math.floor(math.log10(delta))) if math.isfinite(value) else 0

def _unikey_or_keysym_to_mplkey(unikey, keysym):
    if False:
        return 10
    '\n    Convert a Unicode key or X keysym to a Matplotlib key name.\n\n    The Unicode key is checked first; this avoids having to list most printable\n    keysyms such as ``EuroSign``.\n    '
    if unikey and unikey.isprintable():
        return unikey
    key = keysym.lower()
    if key.startswith('kp_'):
        key = key[3:]
    if key.startswith('page_'):
        key = key.replace('page_', 'page')
    if key.endswith(('_l', '_r')):
        key = key[:-2]
    if sys.platform == 'darwin' and key == 'meta':
        key = 'cmd'
    key = {'return': 'enter', 'prior': 'pageup', 'next': 'pagedown'}.get(key, key)
    return key

@functools.cache
def _make_class_factory(mixin_class, fmt, attr_name=None):
    if False:
        while True:
            i = 10
    '\n    Return a function that creates picklable classes inheriting from a mixin.\n\n    After ::\n\n        factory = _make_class_factory(FooMixin, fmt, attr_name)\n        FooAxes = factory(Axes)\n\n    ``Foo`` is a class that inherits from ``FooMixin`` and ``Axes`` and **is\n    picklable** (picklability is what differentiates this from a plain call to\n    `type`).  Its ``__name__`` is set to ``fmt.format(Axes.__name__)`` and the\n    base class is stored in the ``attr_name`` attribute, if not None.\n\n    Moreover, the return value of ``factory`` is memoized: calls with the same\n    ``Axes`` class always return the same subclass.\n    '

    @functools.cache
    def class_factory(axes_class):
        if False:
            for i in range(10):
                print('nop')
        if issubclass(axes_class, mixin_class):
            return axes_class
        base_class = axes_class

        class subcls(mixin_class, base_class):
            __module__ = mixin_class.__module__

            def __reduce__(self):
                if False:
                    while True:
                        i = 10
                return (_picklable_class_constructor, (mixin_class, fmt, attr_name, base_class), self.__getstate__())
        subcls.__name__ = subcls.__qualname__ = fmt.format(base_class.__name__)
        if attr_name is not None:
            setattr(subcls, attr_name, base_class)
        return subcls
    class_factory.__module__ = mixin_class.__module__
    return class_factory

def _picklable_class_constructor(mixin_class, fmt, attr_name, base_class):
    if False:
        for i in range(10):
            print('nop')
    'Internal helper for _make_class_factory.'
    factory = _make_class_factory(mixin_class, fmt, attr_name)
    cls = factory(base_class)
    return cls.__new__(cls)

def _unpack_to_numpy(x):
    if False:
        return 10
    'Internal helper to extract data from e.g. pandas and xarray objects.'
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'to_numpy'):
        return x.to_numpy()
    if hasattr(x, 'values'):
        xtmp = x.values
        if isinstance(xtmp, np.ndarray):
            return xtmp
    return x

def _auto_format_str(fmt, value):
    if False:
        return 10
    "\n    Apply *value* to the format string *fmt*.\n\n    This works both with unnamed %-style formatting and\n    unnamed {}-style formatting. %-style formatting has priority.\n    If *fmt* is %-style formattable that will be used. Otherwise,\n    {}-formatting is applied. Strings without formatting placeholders\n    are passed through as is.\n\n    Examples\n    --------\n    >>> _auto_format_str('%.2f m', 0.2)\n    '0.20 m'\n    >>> _auto_format_str('{} m', 0.2)\n    '0.2 m'\n    >>> _auto_format_str('const', 0.2)\n    'const'\n    >>> _auto_format_str('%d or {}', 0.2)\n    '0 or {}'\n    "
    try:
        return fmt % (value,)
    except (TypeError, ValueError):
        return fmt.format(value)