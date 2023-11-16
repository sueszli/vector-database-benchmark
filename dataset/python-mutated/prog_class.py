"""
Sebastian Raschka 2014-2017
Python Progress Indicator Utility

Author: Sebastian Raschka <sebastianraschka.com>
License: BSD 3 clause

Contributors: https://github.com/rasbt/pyprind/graphs/contributors
Code Repository: https://github.com/rasbt/pyprind
PyPI: https://pypi.python.org/pypi/PyPrind
"""
import os
import sys
import time
from io import UnsupportedOperation
try:
    import psutil
    psutil_import = True
except ImportError:
    psutil_import = False

class Prog:

    def __init__(self, iterations, track_time, stream, title, monitor, update_interval=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes tracking object.'
        self.cnt = 0
        self.title = title
        self.max_iter = float(iterations)
        self.track = track_time
        self.start = time.time()
        self.end = None
        self.item_id = None
        self.eta = None
        self.total_time = 0.0
        self.last_time = self.start
        self.monitor = monitor
        self.stream = stream
        self.active = True
        self._stream_out = self._no_stream
        self._stream_flush = self._no_stream
        self._check_stream()
        self._print_title()
        self.update_interval = update_interval
        self._cached_output = ''
        sys.stdout.flush()
        sys.stderr.flush()
        if monitor:
            if not psutil_import:
                raise ValueError('psutil package is required when using the `monitor` option.')
            else:
                self.process = psutil.Process()
        if self.track:
            self.eta = 1

    def update(self, iterations=1, item_id=None, force_flush=False):
        if False:
            return 10
        '\n        Updates the progress bar / percentage indicator.\n\n        Parameters\n        ----------\n        iterations : int (default: 1)\n            default argument can be changed to integer values\n            >=1 in order to update the progress indicators more than once\n            per iteration.\n        item_id : str (default: None)\n            Print an item_id sring behind the progress bar\n        force_flush : bool (default: False)\n            If True, flushes the progress indicator to the output screen\n            in each iteration.\n\n        '
        self.item_id = item_id
        self.cnt += iterations
        self._print(force_flush=force_flush)
        self._finish()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        'Stops the progress bar / percentage indicator if necessary.'
        self.cnt = self.max_iter
        self._finish()

    def _check_stream(self):
        if False:
            while True:
                i = 10
        'Determines which output stream (stdout, stderr, or custom) to use'
        if self.stream:
            try:
                supported = 'PYCHARM_HOSTED' in os.environ or os.isatty(sys.stdout.fileno())
            except UnsupportedOperation:
                supported = True
            else:
                if self.stream is not None and hasattr(self.stream, 'write'):
                    self._stream_out = self.stream.write
                    self._stream_flush = self.stream.flush
            if supported:
                if self.stream == 1:
                    self._stream_out = sys.stdout.write
                    self._stream_flush = sys.stdout.flush
                elif self.stream == 2:
                    self._stream_out = sys.stderr.write
                    self._stream_flush = sys.stderr.flush
            elif self.stream is not None and hasattr(self.stream, 'write'):
                self._stream_out = self.stream.write
                self._stream_flush = self.stream.flush
            else:
                print('Warning: No valid output stream.')

    def _elapsed(self):
        if False:
            print('Hello World!')
        'Returns elapsed time at update.'
        self.last_time = time.time()
        return self.last_time - self.start

    def _calc_eta(self):
        if False:
            i = 10
            return i + 15
        'Calculates estimated time left until completion.'
        elapsed = self._elapsed()
        if self.cnt == 0 or elapsed < 0.001:
            return None
        rate = float(self.cnt) / elapsed
        self.eta = (float(self.max_iter) - float(self.cnt)) / rate

    def _calc_percent(self):
        if False:
            print('Hello World!')
        'Calculates the rel. progress in percent with 2 decimal points.'
        return round(self.cnt / self.max_iter * 100, 2)

    def _no_stream(self, text=None):
        if False:
            print('Hello World!')
        'Called when no valid output stream is available.'
        pass

    def _get_time(self, _time):
        if False:
            while True:
                i = 10
        if _time < 86400:
            return time.strftime('%H:%M:%S', time.gmtime(_time))
        else:
            s = str(int(_time // 3600)) + ':' + time.strftime('%M:%S', time.gmtime(_time))
            return s

    def _finish(self):
        if False:
            i = 10
            return i + 15
        'Determines if maximum number of iterations (seed) is reached.'
        if self.active and self.cnt >= self.max_iter:
            self.total_time = self._elapsed()
            self.end = time.time()
            self.last_progress -= 1
            self._print()
            if self.track:
                self._stream_out('\nTotal time elapsed: ' + self._get_time(self.total_time))
            self._stream_out('\n')
            self.active = False

    def _print_title(self):
        if False:
            for i in range(10):
                print('nop')
        'Prints tracking title at initialization.'
        if self.title:
            self._stream_out('{}\n'.format(self.title))
            self._stream_flush()

    def _cache_eta(self):
        if False:
            return 10
        'Prints the estimated time left.'
        self._calc_eta()
        self._cached_output += ' | ETA: ' + self._get_time(self.eta)

    def _cache_item_id(self):
        if False:
            for i in range(10):
                print('nop')
        'Prints an item id behind the tracking object.'
        self._cached_output += ' | Item ID: %s' % self.item_id

    def __repr__(self):
        if False:
            return 10
        str_start = time.strftime('%m/%d/%Y %H:%M:%S', time.localtime(self.start))
        str_end = time.strftime('%m/%d/%Y %H:%M:%S', time.localtime(self.end))
        self._stream_flush()
        time_info = 'Title: {}\n  Started: {}\n  Finished: {}\n  Total time elapsed: '.format(self.title, str_start, str_end) + self._get_time(self.total_time)
        if self.monitor:
            try:
                cpu_total = self.process.cpu_percent()
                mem_total = self.process.memory_percent()
            except AttributeError:
                cpu_total = self.process.get_cpu_percent()
                mem_total = self.process.get_memory_percent()
            cpu_mem_info = '  CPU %: {:.2f}\n  Memory %: {:.2f}'.format(cpu_total, mem_total)
            return time_info + '\n' + cpu_mem_info
        else:
            return time_info

    def __str__(self):
        if False:
            print('Hello World!')
        return self.__repr__()