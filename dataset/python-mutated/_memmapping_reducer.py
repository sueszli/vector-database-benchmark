"""
Reducer using memory mapping for numpy arrays
"""
from mmap import mmap
import errno
import os
import stat
import threading
import atexit
import tempfile
import time
import warnings
import weakref
from uuid import uuid4
from multiprocessing import util
from pickle import whichmodule, loads, dumps, HIGHEST_PROTOCOL, PicklingError
try:
    WindowsError
except NameError:
    WindowsError = type(None)
try:
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
except ImportError:
    np = None
from .numpy_pickle import dump, load, load_temporary_memmap
from .backports import make_memmap
from .disk import delete_folder
from .externals.loky.backend import resource_tracker
SYSTEM_SHARED_MEM_FS = '/dev/shm'
SYSTEM_SHARED_MEM_FS_MIN_SIZE = int(2000000000.0)
FOLDER_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
FILE_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR
JOBLIB_MMAPS = set()

def _log_and_unlink(filename):
    if False:
        for i in range(10):
            print('nop')
    from .externals.loky.backend.resource_tracker import _resource_tracker
    util.debug('[FINALIZER CALL] object mapping to {} about to be deleted, decrementing the refcount of the file (pid: {})'.format(os.path.basename(filename), os.getpid()))
    _resource_tracker.maybe_unlink(filename, 'file')

def add_maybe_unlink_finalizer(memmap):
    if False:
        while True:
            i = 10
    util.debug('[FINALIZER ADD] adding finalizer to {} (id {}, filename {}, pid  {})'.format(type(memmap), id(memmap), os.path.basename(memmap.filename), os.getpid()))
    weakref.finalize(memmap, _log_and_unlink, memmap.filename)

def unlink_file(filename):
    if False:
        print('Hello World!')
    'Wrapper around os.unlink with a retry mechanism.\n\n    The retry mechanism has been implemented primarily to overcome a race\n    condition happening during the finalizer of a np.memmap: when a process\n    holding the last reference to a mmap-backed np.memmap/np.array is about to\n    delete this array (and close the reference), it sends a maybe_unlink\n    request to the resource_tracker. This request can be processed faster than\n    it takes for the last reference of the memmap to be closed, yielding (on\n    Windows) a PermissionError in the resource_tracker loop.\n    '
    NUM_RETRIES = 10
    for retry_no in range(1, NUM_RETRIES + 1):
        try:
            os.unlink(filename)
            break
        except PermissionError:
            util.debug('[ResourceTracker] tried to unlink {}, got PermissionError'.format(filename))
            if retry_no == NUM_RETRIES:
                raise
            else:
                time.sleep(0.2)
        except FileNotFoundError:
            pass
resource_tracker._CLEANUP_FUNCS['file'] = unlink_file

class _WeakArrayKeyMap:
    """A variant of weakref.WeakKeyDictionary for unhashable numpy arrays.

    This datastructure will be used with numpy arrays as obj keys, therefore we
    do not use the __get__ / __set__ methods to avoid any conflict with the
    numpy fancy indexing syntax.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._data = {}

    def get(self, obj):
        if False:
            while True:
                i = 10
        (ref, val) = self._data[id(obj)]
        if ref() is not obj:
            raise KeyError(obj)
        return val

    def set(self, obj, value):
        if False:
            while True:
                i = 10
        key = id(obj)
        try:
            (ref, _) = self._data[key]
            if ref() is not obj:
                raise KeyError(obj)
        except KeyError:

            def on_destroy(_):
                if False:
                    for i in range(10):
                        print('nop')
                del self._data[key]
            ref = weakref.ref(obj, on_destroy)
        self._data[key] = (ref, value)

    def __getstate__(self):
        if False:
            print('Hello World!')
        raise PicklingError('_WeakArrayKeyMap is not pickleable')

def _get_backing_memmap(a):
    if False:
        i = 10
        return i + 15
    'Recursively look up the original np.memmap instance base if any.'
    b = getattr(a, 'base', None)
    if b is None:
        return None
    elif isinstance(b, mmap):
        return a
    else:
        return _get_backing_memmap(b)

def _get_temp_dir(pool_folder_name, temp_folder=None):
    if False:
        return 10
    'Get the full path to a subfolder inside the temporary folder.\n\n    Parameters\n    ----------\n    pool_folder_name : str\n        Sub-folder name used for the serialization of a pool instance.\n\n    temp_folder: str, optional\n        Folder to be used by the pool for memmapping large arrays\n        for sharing memory with worker processes. If None, this will try in\n        order:\n\n        - a folder pointed by the JOBLIB_TEMP_FOLDER environment\n          variable,\n        - /dev/shm if the folder exists and is writable: this is a\n          RAMdisk filesystem available by default on modern Linux\n          distributions,\n        - the default system temporary folder that can be\n          overridden with TMP, TMPDIR or TEMP environment\n          variables, typically /tmp under Unix operating systems.\n\n    Returns\n    -------\n    pool_folder : str\n       full path to the temporary folder\n    use_shared_mem : bool\n       whether the temporary folder is written to the system shared memory\n       folder or some other temporary folder.\n    '
    use_shared_mem = False
    if temp_folder is None:
        temp_folder = os.environ.get('JOBLIB_TEMP_FOLDER', None)
    if temp_folder is None:
        if os.path.exists(SYSTEM_SHARED_MEM_FS) and hasattr(os, 'statvfs'):
            try:
                shm_stats = os.statvfs(SYSTEM_SHARED_MEM_FS)
                available_nbytes = shm_stats.f_bsize * shm_stats.f_bavail
                if available_nbytes > SYSTEM_SHARED_MEM_FS_MIN_SIZE:
                    temp_folder = SYSTEM_SHARED_MEM_FS
                    pool_folder = os.path.join(temp_folder, pool_folder_name)
                    if not os.path.exists(pool_folder):
                        os.makedirs(pool_folder)
                    use_shared_mem = True
            except (IOError, OSError):
                temp_folder = None
    if temp_folder is None:
        temp_folder = tempfile.gettempdir()
    temp_folder = os.path.abspath(os.path.expanduser(temp_folder))
    pool_folder = os.path.join(temp_folder, pool_folder_name)
    return (pool_folder, use_shared_mem)

def has_shareable_memory(a):
    if False:
        while True:
            i = 10
    'Return True if a is backed by some mmap buffer directly or not.'
    return _get_backing_memmap(a) is not None

def _strided_from_memmap(filename, dtype, mode, offset, order, shape, strides, total_buffer_len, unlink_on_gc_collect):
    if False:
        return 10
    'Reconstruct an array view on a memory mapped file.'
    if mode == 'w+':
        mode = 'r+'
    if strides is None:
        return make_memmap(filename, dtype=dtype, shape=shape, mode=mode, offset=offset, order=order, unlink_on_gc_collect=unlink_on_gc_collect)
    else:
        base = make_memmap(filename, dtype=dtype, shape=total_buffer_len, offset=offset, mode=mode, order=order, unlink_on_gc_collect=unlink_on_gc_collect)
        return as_strided(base, shape=shape, strides=strides)

def _reduce_memmap_backed(a, m):
    if False:
        for i in range(10):
            print('nop')
    'Pickling reduction for memmap backed arrays.\n\n    a is expected to be an instance of np.ndarray (or np.memmap)\n    m is expected to be an instance of np.memmap on the top of the ``base``\n    attribute ancestry of a. ``m.base`` should be the real python mmap object.\n    '
    util.debug('[MEMMAP REDUCE] reducing a memmap-backed array (shape, {}, pid: {})'.format(a.shape, os.getpid()))
    try:
        from numpy.lib.array_utils import byte_bounds
    except (ModuleNotFoundError, ImportError):
        from numpy import byte_bounds
    (a_start, a_end) = byte_bounds(a)
    m_start = byte_bounds(m)[0]
    offset = a_start - m_start
    offset += m.offset
    if m.flags['F_CONTIGUOUS']:
        order = 'F'
    else:
        order = 'C'
    if a.flags['F_CONTIGUOUS'] or a.flags['C_CONTIGUOUS']:
        strides = None
        total_buffer_len = None
    else:
        strides = a.strides
        total_buffer_len = (a_end - a_start) // a.itemsize
    return (_strided_from_memmap, (m.filename, a.dtype, m.mode, offset, order, a.shape, strides, total_buffer_len, False))

def reduce_array_memmap_backward(a):
    if False:
        return 10
    'reduce a np.array or a np.memmap from a child process'
    m = _get_backing_memmap(a)
    if isinstance(m, np.memmap) and m.filename not in JOBLIB_MMAPS:
        return _reduce_memmap_backed(a, m)
    else:
        return (loads, (dumps(np.asarray(a), protocol=HIGHEST_PROTOCOL),))

class ArrayMemmapForwardReducer(object):
    """Reducer callable to dump large arrays to memmap files.

    Parameters
    ----------
    max_nbytes: int
        Threshold to trigger memmapping of large arrays to files created
        a folder.
    temp_folder_resolver: callable
        An callable in charge of resolving a temporary folder name where files
        for backing memmapped arrays are created.
    mmap_mode: 'r', 'r+' or 'c'
        Mode for the created memmap datastructure. See the documentation of
        numpy.memmap for more details. Note: 'w+' is coerced to 'r+'
        automatically to avoid zeroing the data on unpickling.
    verbose: int, optional, 0 by default
        If verbose > 0, memmap creations are logged.
        If verbose > 1, both memmap creations, reuse and array pickling are
        logged.
    prewarm: bool, optional, False by default.
        Force a read on newly memmapped array to make sure that OS pre-cache it
        memory. This can be useful to avoid concurrent disk access when the
        same data array is passed to different worker processes.
    """

    def __init__(self, max_nbytes, temp_folder_resolver, mmap_mode, unlink_on_gc_collect, verbose=0, prewarm=True):
        if False:
            print('Hello World!')
        self._max_nbytes = max_nbytes
        self._temp_folder_resolver = temp_folder_resolver
        self._mmap_mode = mmap_mode
        self.verbose = int(verbose)
        if prewarm == 'auto':
            self._prewarm = not self._temp_folder.startswith(SYSTEM_SHARED_MEM_FS)
        else:
            self._prewarm = prewarm
        self._prewarm = prewarm
        self._memmaped_arrays = _WeakArrayKeyMap()
        self._temporary_memmaped_filenames = set()
        self._unlink_on_gc_collect = unlink_on_gc_collect

    @property
    def _temp_folder(self):
        if False:
            for i in range(10):
                print('nop')
        return self._temp_folder_resolver()

    def __reduce__(self):
        if False:
            print('Hello World!')
        args = (self._max_nbytes, None, self._mmap_mode, self._unlink_on_gc_collect)
        kwargs = {'verbose': self.verbose, 'prewarm': self._prewarm}
        return (ArrayMemmapForwardReducer, args, kwargs)

    def __call__(self, a):
        if False:
            i = 10
            return i + 15
        m = _get_backing_memmap(a)
        if m is not None and isinstance(m, np.memmap):
            return _reduce_memmap_backed(a, m)
        if not a.dtype.hasobject and self._max_nbytes is not None and (a.nbytes > self._max_nbytes):
            try:
                os.makedirs(self._temp_folder)
                os.chmod(self._temp_folder, FOLDER_PERMISSIONS)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise e
            try:
                basename = self._memmaped_arrays.get(a)
            except KeyError:
                basename = '{}-{}-{}.pkl'.format(os.getpid(), id(threading.current_thread()), uuid4().hex)
                self._memmaped_arrays.set(a, basename)
            filename = os.path.join(self._temp_folder, basename)
            is_new_memmap = filename not in self._temporary_memmaped_filenames
            self._temporary_memmaped_filenames.add(filename)
            if self._unlink_on_gc_collect:
                resource_tracker.register(filename, 'file')
            if is_new_memmap:
                resource_tracker.register(filename, 'file')
            if not os.path.exists(filename):
                util.debug('[ARRAY DUMP] Pickling new array (shape={}, dtype={}) creating a new memmap at {}'.format(a.shape, a.dtype, filename))
                for dumped_filename in dump(a, filename):
                    os.chmod(dumped_filename, FILE_PERMISSIONS)
                if self._prewarm:
                    load(filename, mmap_mode=self._mmap_mode).max()
            else:
                util.debug('[ARRAY DUMP] Pickling known array (shape={}, dtype={}) reusing memmap file: {}'.format(a.shape, a.dtype, os.path.basename(filename)))
            return (load_temporary_memmap, (filename, self._mmap_mode, self._unlink_on_gc_collect))
        else:
            util.debug('[ARRAY DUMP] Pickling array (NO MEMMAPPING) (shape={},  dtype={}).'.format(a.shape, a.dtype))
            return (loads, (dumps(a, protocol=HIGHEST_PROTOCOL),))

def get_memmapping_reducers(forward_reducers=None, backward_reducers=None, temp_folder_resolver=None, max_nbytes=1000000.0, mmap_mode='r', verbose=0, prewarm=False, unlink_on_gc_collect=True, **kwargs):
    if False:
        print('Hello World!')
    'Construct a pair of memmapping reducer linked to a tmpdir.\n\n    This function manage the creation and the clean up of the temporary folders\n    underlying the memory maps and should be use to get the reducers necessary\n    to construct joblib pool or executor.\n    '
    if forward_reducers is None:
        forward_reducers = dict()
    if backward_reducers is None:
        backward_reducers = dict()
    if np is not None:
        forward_reduce_ndarray = ArrayMemmapForwardReducer(max_nbytes, temp_folder_resolver, mmap_mode, unlink_on_gc_collect, verbose, prewarm=prewarm)
        forward_reducers[np.ndarray] = forward_reduce_ndarray
        forward_reducers[np.memmap] = forward_reduce_ndarray
        backward_reducers[np.ndarray] = reduce_array_memmap_backward
        backward_reducers[np.memmap] = reduce_array_memmap_backward
    return (forward_reducers, backward_reducers)

class TemporaryResourcesManager(object):
    """Stateful object able to manage temporary folder and pickles

    It exposes:
    - a per-context folder name resolving API that memmap-based reducers will
      rely on to know where to pickle the temporary memmaps
    - a temporary file/folder management API that internally uses the
      resource_tracker.
    """

    def __init__(self, temp_folder_root=None, context_id=None):
        if False:
            for i in range(10):
                print('nop')
        self._current_temp_folder = None
        self._temp_folder_root = temp_folder_root
        self._use_shared_mem = None
        self._cached_temp_folders = dict()
        self._id = uuid4().hex
        self._finalizers = {}
        if context_id is None:
            context_id = uuid4().hex
        self.set_current_context(context_id)

    def set_current_context(self, context_id):
        if False:
            while True:
                i = 10
        self._current_context_id = context_id
        self.register_new_context(context_id)

    def register_new_context(self, context_id):
        if False:
            return 10
        if context_id in self._cached_temp_folders:
            return
        else:
            new_folder_name = 'joblib_memmapping_folder_{}_{}_{}'.format(os.getpid(), self._id, context_id)
            (new_folder_path, _) = _get_temp_dir(new_folder_name, self._temp_folder_root)
            self.register_folder_finalizer(new_folder_path, context_id)
            self._cached_temp_folders[context_id] = new_folder_path

    def resolve_temp_folder_name(self):
        if False:
            return 10
        'Return a folder name specific to the currently activated context'
        return self._cached_temp_folders[self._current_context_id]

    def register_folder_finalizer(self, pool_subfolder, context_id):
        if False:
            i = 10
            return i + 15
        pool_module_name = whichmodule(delete_folder, 'delete_folder')
        resource_tracker.register(pool_subfolder, 'folder')

        def _cleanup():
            if False:
                i = 10
                return i + 15
            delete_folder = __import__(pool_module_name, fromlist=['delete_folder']).delete_folder
            try:
                delete_folder(pool_subfolder, allow_non_empty=True)
                resource_tracker.unregister(pool_subfolder, 'folder')
            except OSError:
                warnings.warn('Failed to delete temporary folder: {}'.format(pool_subfolder))
        self._finalizers[context_id] = atexit.register(_cleanup)

    def _clean_temporary_resources(self, context_id=None, force=False, allow_non_empty=False):
        if False:
            i = 10
            return i + 15
        'Clean temporary resources created by a process-based pool'
        if context_id is None:
            for context_id in list(self._cached_temp_folders):
                self._clean_temporary_resources(context_id, force=force, allow_non_empty=allow_non_empty)
        else:
            temp_folder = self._cached_temp_folders.get(context_id)
            if temp_folder and os.path.exists(temp_folder):
                for filename in os.listdir(temp_folder):
                    if force:
                        resource_tracker.unregister(os.path.join(temp_folder, filename), 'file')
                    else:
                        resource_tracker.maybe_unlink(os.path.join(temp_folder, filename), 'file')
                allow_non_empty |= force
                try:
                    delete_folder(temp_folder, allow_non_empty=allow_non_empty)
                    self._cached_temp_folders.pop(context_id, None)
                    resource_tracker.unregister(temp_folder, 'folder')
                    finalizer = self._finalizers.pop(context_id, None)
                    if finalizer is not None:
                        atexit.unregister(finalizer)
                except OSError:
                    pass