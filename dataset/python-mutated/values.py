import os
from threading import Lock
import warnings
from .mmap_dict import mmap_key, MmapedDict

class MutexValue:
    """A float protected by a mutex."""
    _multiprocess = False

    def __init__(self, typ, metric_name, name, labelnames, labelvalues, help_text, **kwargs):
        if False:
            i = 10
            return i + 15
        self._value = 0.0
        self._exemplar = None
        self._lock = Lock()

    def inc(self, amount):
        if False:
            while True:
                i = 10
        with self._lock:
            self._value += amount

    def set(self, value, timestamp=None):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            self._value = value

    def set_exemplar(self, exemplar):
        if False:
            while True:
                i = 10
        with self._lock:
            self._exemplar = exemplar

    def get(self):
        if False:
            while True:
                i = 10
        with self._lock:
            return self._value

    def get_exemplar(self):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            return self._exemplar

def MultiProcessValue(process_identifier=os.getpid):
    if False:
        return 10
    "Returns a MmapedValue class based on a process_identifier function.\n\n    The 'process_identifier' function MUST comply with this simple rule:\n    when called in simultaneously running processes it MUST return distinct values.\n\n    Using a different function than the default 'os.getpid' is at your own risk.\n    "
    files = {}
    values = []
    pid = {'value': process_identifier()}
    lock = Lock()

    class MmapedValue:
        """A float protected by a mutex backed by a per-process mmaped file."""
        _multiprocess = True

        def __init__(self, typ, metric_name, name, labelnames, labelvalues, help_text, multiprocess_mode='', **kwargs):
            if False:
                print('Hello World!')
            self._params = (typ, metric_name, name, labelnames, labelvalues, help_text, multiprocess_mode)
            if 'prometheus_multiproc_dir' in os.environ and 'PROMETHEUS_MULTIPROC_DIR' not in os.environ:
                os.environ['PROMETHEUS_MULTIPROC_DIR'] = os.environ['prometheus_multiproc_dir']
                warnings.warn('prometheus_multiproc_dir variable has been deprecated in favor of the upper case naming PROMETHEUS_MULTIPROC_DIR', DeprecationWarning)
            with lock:
                self.__check_for_pid_change()
                self.__reset()
                values.append(self)

        def __reset(self):
            if False:
                while True:
                    i = 10
            (typ, metric_name, name, labelnames, labelvalues, help_text, multiprocess_mode) = self._params
            if typ == 'gauge':
                file_prefix = typ + '_' + multiprocess_mode
            else:
                file_prefix = typ
            if file_prefix not in files:
                filename = os.path.join(os.environ.get('PROMETHEUS_MULTIPROC_DIR'), '{}_{}.db'.format(file_prefix, pid['value']))
                files[file_prefix] = MmapedDict(filename)
            self._file = files[file_prefix]
            self._key = mmap_key(metric_name, name, labelnames, labelvalues, help_text)
            (self._value, self._timestamp) = self._file.read_value(self._key)

        def __check_for_pid_change(self):
            if False:
                while True:
                    i = 10
            actual_pid = process_identifier()
            if pid['value'] != actual_pid:
                pid['value'] = actual_pid
                for f in files.values():
                    f.close()
                files.clear()
                for value in values:
                    value.__reset()

        def inc(self, amount):
            if False:
                print('Hello World!')
            with lock:
                self.__check_for_pid_change()
                self._value += amount
                self._timestamp = 0.0
                self._file.write_value(self._key, self._value, self._timestamp)

        def set(self, value, timestamp=None):
            if False:
                i = 10
                return i + 15
            with lock:
                self.__check_for_pid_change()
                self._value = value
                self._timestamp = timestamp or 0.0
                self._file.write_value(self._key, self._value, self._timestamp)

        def set_exemplar(self, exemplar):
            if False:
                print('Hello World!')
            return

        def get(self):
            if False:
                return 10
            with lock:
                self.__check_for_pid_change()
                return self._value

        def get_exemplar(self):
            if False:
                i = 10
                return i + 15
            return None
    return MmapedValue

def get_value_class():
    if False:
        print('Hello World!')
    if 'prometheus_multiproc_dir' in os.environ or 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
        return MultiProcessValue()
    else:
        return MutexValue
ValueClass = get_value_class()