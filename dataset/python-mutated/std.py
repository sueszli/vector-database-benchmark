"""
Customisable progressbar decorator for iterators.
Includes a default `range` iterator printing to `stderr`.

Usage:
>>> from tqdm import trange, tqdm
>>> for i in trange(10):
...     ...
"""
import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet
from ._monitor import TMonitor
from .utils import CallbackIOWrapper, Comparable, DisableOnWriteError, FormatReplace, SimpleTextIOWrapper, _is_ascii, _screen_shape_wrapper, _supports_unicode, _term_move_up, disp_len, disp_trim, envwrap
__author__ = 'https://github.com/tqdm/tqdm#contributions'
__all__ = ['tqdm', 'trange', 'TqdmTypeError', 'TqdmKeyError', 'TqdmWarning', 'TqdmExperimentalWarning', 'TqdmDeprecationWarning', 'TqdmMonitorWarning']

class TqdmTypeError(TypeError):
    pass

class TqdmKeyError(KeyError):
    pass

class TqdmWarning(Warning):
    """base class for all tqdm warnings.

    Used for non-external-code-breaking errors, such as garbled printing.
    """

    def __init__(self, msg, fp_write=None, *a, **k):
        if False:
            while True:
                i = 10
        if fp_write is not None:
            fp_write('\n' + self.__class__.__name__ + ': ' + str(msg).rstrip() + '\n')
        else:
            super(TqdmWarning, self).__init__(msg, *a, **k)

class TqdmExperimentalWarning(TqdmWarning, FutureWarning):
    """beta feature, unstable API and behaviour"""
    pass

class TqdmDeprecationWarning(TqdmWarning, DeprecationWarning):
    pass

class TqdmMonitorWarning(TqdmWarning, RuntimeWarning):
    """tqdm monitor errors which do not affect external functionality"""
    pass

def TRLock(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    'threading RLock'
    try:
        from threading import RLock
        return RLock(*args, **kwargs)
    except (ImportError, OSError):
        pass

class TqdmDefaultWriteLock(object):
    """
    Provide a default write lock for thread and multiprocessing safety.
    Works only on platforms supporting `fork` (so Windows is excluded).
    You must initialise a `tqdm` or `TqdmDefaultWriteLock` instance
    before forking in order for the write lock to work.
    On Windows, you need to supply the lock from the parent to the children as
    an argument to joblib or the parallelism lib you use.
    """
    th_lock = TRLock()

    def __init__(self):
        if False:
            print('Hello World!')
        cls = type(self)
        root_lock = cls.th_lock
        if root_lock is not None:
            root_lock.acquire()
        cls.create_mp_lock()
        self.locks = [lk for lk in [cls.mp_lock, cls.th_lock] if lk is not None]
        if root_lock is not None:
            root_lock.release()

    def acquire(self, *a, **k):
        if False:
            while True:
                i = 10
        for lock in self.locks:
            lock.acquire(*a, **k)

    def release(self):
        if False:
            while True:
                i = 10
        for lock in self.locks[::-1]:
            lock.release()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.acquire()

    def __exit__(self, *exc):
        if False:
            for i in range(10):
                print('nop')
        self.release()

    @classmethod
    def create_mp_lock(cls):
        if False:
            while True:
                i = 10
        if not hasattr(cls, 'mp_lock'):
            try:
                from multiprocessing import RLock
                cls.mp_lock = RLock()
            except (ImportError, OSError):
                cls.mp_lock = None

    @classmethod
    def create_th_lock(cls):
        if False:
            return 10
        assert hasattr(cls, 'th_lock')
        warn('create_th_lock not needed anymore', TqdmDeprecationWarning, stacklevel=2)

class Bar(object):
    """
    `str.format`-able bar with format specifiers: `[width][type]`

    - `width`
      + unspecified (default): use `self.default_len`
      + `int >= 0`: overrides `self.default_len`
      + `int < 0`: subtract from `self.default_len`
    - `type`
      + `a`: ascii (`charset=self.ASCII` override)
      + `u`: unicode (`charset=self.UTF` override)
      + `b`: blank (`charset="  "` override)
    """
    ASCII = ' 123456789#'
    UTF = u' ' + u''.join(map(chr, range(9615, 9607, -1)))
    BLANK = '  '
    COLOUR_RESET = '\x1b[0m'
    COLOUR_RGB = '\x1b[38;2;%d;%d;%dm'
    COLOURS = {'BLACK': '\x1b[30m', 'RED': '\x1b[31m', 'GREEN': '\x1b[32m', 'YELLOW': '\x1b[33m', 'BLUE': '\x1b[34m', 'MAGENTA': '\x1b[35m', 'CYAN': '\x1b[36m', 'WHITE': '\x1b[37m'}

    def __init__(self, frac, default_len=10, charset=UTF, colour=None):
        if False:
            for i in range(10):
                print('nop')
        if not 0 <= frac <= 1:
            warn('clamping frac to range [0, 1]', TqdmWarning, stacklevel=2)
            frac = max(0, min(1, frac))
        assert default_len > 0
        self.frac = frac
        self.default_len = default_len
        self.charset = charset
        self.colour = colour

    @property
    def colour(self):
        if False:
            while True:
                i = 10
        return self._colour

    @colour.setter
    def colour(self, value):
        if False:
            print('Hello World!')
        if not value:
            self._colour = None
            return
        try:
            if value.upper() in self.COLOURS:
                self._colour = self.COLOURS[value.upper()]
            elif value[0] == '#' and len(value) == 7:
                self._colour = self.COLOUR_RGB % tuple((int(i, 16) for i in (value[1:3], value[3:5], value[5:7])))
            else:
                raise KeyError
        except (KeyError, AttributeError):
            warn('Unknown colour (%s); valid choices: [hex (#00ff00), %s]' % (value, ', '.join(self.COLOURS)), TqdmWarning, stacklevel=2)
            self._colour = None

    def __format__(self, format_spec):
        if False:
            return 10
        if format_spec:
            _type = format_spec[-1].lower()
            try:
                charset = {'a': self.ASCII, 'u': self.UTF, 'b': self.BLANK}[_type]
            except KeyError:
                charset = self.charset
            else:
                format_spec = format_spec[:-1]
            if format_spec:
                N_BARS = int(format_spec)
                if N_BARS < 0:
                    N_BARS += self.default_len
            else:
                N_BARS = self.default_len
        else:
            charset = self.charset
            N_BARS = self.default_len
        nsyms = len(charset) - 1
        (bar_length, frac_bar_length) = divmod(int(self.frac * N_BARS * nsyms), nsyms)
        res = charset[-1] * bar_length
        if bar_length < N_BARS:
            res = res + charset[frac_bar_length] + charset[0] * (N_BARS - bar_length - 1)
        return self.colour + res + self.COLOUR_RESET if self.colour else res

class EMA(object):
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.

    Parameters
    ----------
    smoothing  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Increase to give more weight to recent values.
        Ranges from 0 (yields old value) to 1 (yields new value).
    """

    def __init__(self, smoothing=0.3):
        if False:
            for i in range(10):
                print('nop')
        self.alpha = smoothing
        self.last = 0
        self.calls = 0

    def __call__(self, x=None):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        x  : float\n            New value to include in EMA.\n        '
        beta = 1 - self.alpha
        if x is not None:
            self.last = self.alpha * x + beta * self.last
            self.calls += 1
        return self.last / (1 - beta ** self.calls) if self.calls else self.last

class tqdm(Comparable):
    """
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progressbar every time a value is requested.

    Parameters
    ----------
    iterable  : iterable, optional
        Iterable to decorate with a progressbar.
        Leave blank to manually manage the updates.
    desc  : str, optional
        Prefix for the progressbar.
    total  : int or float, optional
        The number of expected iterations. If unspecified,
        len(iterable) is used if possible. If float("inf") or as a last
        resort, only basic progress statistics are displayed
        (no ETA, no progressbar).
        If `gui` is True and this parameter needs subsequent updating,
        specify an initial arbitrary large positive number,
        e.g. 9e9.
    leave  : bool, optional
        If [default: True], keeps all traces of the progressbar
        upon termination of iteration.
        If `None`, will leave only if `position` is `0`.
    file  : `io.TextIOWrapper` or `io.StringIO`, optional
        Specifies where to output the progress messages
        (default: sys.stderr). Uses `file.write(str)` and `file.flush()`
        methods.  For encoding, see `write_bytes`.
    ncols  : int, optional
        The width of the entire output message. If specified,
        dynamically resizes the progressbar to stay within this bound.
        If unspecified, attempts to use environment width. The
        fallback is a meter width of 10 and no limit for the counter and
        statistics. If 0, will not print any meter (only stats).
    mininterval  : float, optional
        Minimum progress display update interval [default: 0.1] seconds.
    maxinterval  : float, optional
        Maximum progress display update interval [default: 10] seconds.
        Automatically adjusts `miniters` to correspond to `mininterval`
        after long display update lag. Only works if `dynamic_miniters`
        or monitor thread is enabled.
    miniters  : int or float, optional
        Minimum progress display update interval, in iterations.
        If 0 and `dynamic_miniters`, will automatically adjust to equal
        `mininterval` (more CPU efficient, good for tight loops).
        If > 0, will skip display of specified number of iterations.
        Tweak this and `mininterval` to get very efficient loops.
        If your progress is erratic with both fast and slow iterations
        (network, skipping items, etc) you should set miniters=1.
    ascii  : bool or str, optional
        If unspecified or False, use unicode (smooth blocks) to fill
        the meter. The fallback is to use ASCII characters " 123456789#".
    disable  : bool, optional
        Whether to disable the entire progressbar wrapper
        [default: False]. If set to None, disable on non-TTY.
    unit  : str, optional
        String that will be used to define the unit of each iteration
        [default: it].
    unit_scale  : bool or int or float, optional
        If 1 or True, the number of iterations will be reduced/scaled
        automatically and a metric prefix following the
        International System of Units standard will be added
        (kilo, mega, etc.) [default: False]. If any other non-zero
        number, will scale `total` and `n`.
    dynamic_ncols  : bool, optional
        If set, constantly alters `ncols` and `nrows` to the
        environment (allowing for window resizes) [default: False].
    smoothing  : float, optional
        Exponential moving average smoothing factor for speed estimates
        (ignored in GUI mode). Ranges from 0 (average speed) to 1
        (current/instantaneous speed) [default: 0.3].
    bar_format  : str, optional
        Specify a custom bar string formatting. May impact performance.
        [default: '{l_bar}{bar}{r_bar}'], where
        l_bar='{desc}: {percentage:3.0f}%|' and
        r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
            '{rate_fmt}{postfix}]'
        Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
            percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
            rate, rate_fmt, rate_noinv, rate_noinv_fmt,
            rate_inv, rate_inv_fmt, postfix, unit_divisor,
            remaining, remaining_s, eta.
        Note that a trailing ": " is automatically removed after {desc}
        if the latter is empty.
    initial  : int or float, optional
        The initial counter value. Useful when restarting a progress
        bar [default: 0]. If using float, consider specifying `{n:.3f}`
        or similar in `bar_format`, or specifying `unit_scale`.
    position  : int, optional
        Specify the line offset to print this bar (starting from 0)
        Automatic if unspecified.
        Useful to manage multiple bars at once (eg, from threads).
    postfix  : dict or *, optional
        Specify additional stats to display at the end of the bar.
        Calls `set_postfix(**postfix)` if possible (dict).
    unit_divisor  : float, optional
        [default: 1000], ignored unless `unit_scale` is True.
    write_bytes  : bool, optional
        Whether to write bytes. If (default: False) will write unicode.
    lock_args  : tuple, optional
        Passed to `refresh` for intermediate output
        (initialisation, iterating, and updating).
    nrows  : int, optional
        The screen height. If specified, hides nested bars outside this
        bound. If unspecified, attempts to use environment height.
        The fallback is 20.
    colour  : str, optional
        Bar colour (e.g. 'green', '#00ff00').
    delay  : float, optional
        Don't display until [default: 0] seconds have elapsed.
    gui  : bool, optional
        WARNING: internal parameter - do not use.
        Use tqdm.gui.tqdm(...) instead. If set, will attempt to use
        matplotlib animations for a graphical output [default: False].

    Returns
    -------
    out  : decorated iterator.
    """
    monitor_interval = 10
    monitor = None
    _instances = WeakSet()

    @staticmethod
    def format_sizeof(num, suffix='', divisor=1000):
        if False:
            while True:
                i = 10
        "\n        Formats a number (greater than unity) with SI Order of Magnitude\n        prefixes.\n\n        Parameters\n        ----------\n        num  : float\n            Number ( >= 1) to format.\n        suffix  : str, optional\n            Post-postfix [default: ''].\n        divisor  : float, optional\n            Divisor between prefixes [default: 1000].\n\n        Returns\n        -------\n        out  : str\n            Number with Order of Magnitude SI unit postfix.\n        "
        for unit in ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 999.5:
                if abs(num) < 99.95:
                    if abs(num) < 9.995:
                        return '{0:1.2f}'.format(num) + unit + suffix
                    return '{0:2.1f}'.format(num) + unit + suffix
                return '{0:3.0f}'.format(num) + unit + suffix
            num /= divisor
        return '{0:3.1f}Y'.format(num) + suffix

    @staticmethod
    def format_interval(t):
        if False:
            for i in range(10):
                print('nop')
        '\n        Formats a number of seconds as a clock time, [H:]MM:SS\n\n        Parameters\n        ----------\n        t  : int\n            Number of seconds.\n\n        Returns\n        -------\n        out  : str\n            [H:]MM:SS\n        '
        (mins, s) = divmod(int(t), 60)
        (h, m) = divmod(mins, 60)
        if h:
            return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
        else:
            return '{0:02d}:{1:02d}'.format(m, s)

    @staticmethod
    def format_num(n):
        if False:
            while True:
                i = 10
        '\n        Intelligent scientific notation (.3g).\n\n        Parameters\n        ----------\n        n  : int or float or Numeric\n            A Number.\n\n        Returns\n        -------\n        out  : str\n            Formatted number.\n        '
        f = '{0:.3g}'.format(n).replace('+0', '+').replace('-0', '-')
        n = str(n)
        return f if len(f) < len(n) else n

    @staticmethod
    def status_printer(file):
        if False:
            i = 10
            return i + 15
        '\n        Manage the printing and in-place updating of a line of characters.\n        Note that if the string is longer than a line, then in-place\n        updating may not work (it will print a new line at each refresh).\n        '
        fp = file
        fp_flush = getattr(fp, 'flush', lambda : None)
        if fp in (sys.stderr, sys.stdout):
            getattr(sys.stderr, 'flush', lambda : None)()
            getattr(sys.stdout, 'flush', lambda : None)()

        def fp_write(s):
            if False:
                print('Hello World!')
            fp.write(str(s))
            fp_flush()
        last_len = [0]

        def print_status(s):
            if False:
                return 10
            len_s = disp_len(s)
            fp_write('\r' + s + ' ' * max(last_len[0] - len_s, 0))
            last_len[0] = len_s
        return print_status

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='', ascii=False, unit='it', unit_scale=False, rate=None, bar_format=None, postfix=None, unit_divisor=1000, initial=0, colour=None, **extra_kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a string-based progress bar given some parameters\n\n        Parameters\n        ----------\n        n  : int or float\n            Number of finished iterations.\n        total  : int or float\n            The expected total number of iterations. If meaningless (None),\n            only basic progress statistics are displayed (no ETA).\n        elapsed  : float\n            Number of seconds passed since start.\n        ncols  : int, optional\n            The width of the entire output message. If specified,\n            dynamically resizes `{bar}` to stay within this bound\n            [default: None]. If `0`, will not print any bar (only stats).\n            The fallback is `{bar:10}`.\n        prefix  : str, optional\n            Prefix message (included in total width) [default: \'\'].\n            Use as {desc} in bar_format string.\n        ascii  : bool, optional or str, optional\n            If not set, use unicode (smooth blocks) to fill the meter\n            [default: False]. The fallback is to use ASCII characters\n            " 123456789#".\n        unit  : str, optional\n            The iteration unit [default: \'it\'].\n        unit_scale  : bool or int or float, optional\n            If 1 or True, the number of iterations will be printed with an\n            appropriate SI metric prefix (k = 10^3, M = 10^6, etc.)\n            [default: False]. If any other non-zero number, will scale\n            `total` and `n`.\n        rate  : float, optional\n            Manual override for iteration rate.\n            If [default: None], uses n/elapsed.\n        bar_format  : str, optional\n            Specify a custom bar string formatting. May impact performance.\n            [default: \'{l_bar}{bar}{r_bar}\'], where\n            l_bar=\'{desc}: {percentage:3.0f}%|\' and\n            r_bar=\'| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, \'\n              \'{rate_fmt}{postfix}]\'\n            Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,\n              percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,\n              rate, rate_fmt, rate_noinv, rate_noinv_fmt,\n              rate_inv, rate_inv_fmt, postfix, unit_divisor,\n              remaining, remaining_s, eta.\n            Note that a trailing ": " is automatically removed after {desc}\n            if the latter is empty.\n        postfix  : *, optional\n            Similar to `prefix`, but placed at the end\n            (e.g. for additional stats).\n            Note: postfix is usually a string (not a dict) for this method,\n            and will if possible be set to postfix = \', \' + postfix.\n            However other types are supported (#382).\n        unit_divisor  : float, optional\n            [default: 1000], ignored unless `unit_scale` is True.\n        initial  : int or float, optional\n            The initial counter value [default: 0].\n        colour  : str, optional\n            Bar colour (e.g. \'green\', \'#00ff00\').\n\n        Returns\n        -------\n        out  : Formatted meter and stats, ready to display.\n        '
        if total and n >= total + 0.5:
            total = None
        if unit_scale and unit_scale not in (True, 1):
            if total:
                total *= unit_scale
            n *= unit_scale
            if rate:
                rate *= unit_scale
            unit_scale = False
        elapsed_str = tqdm.format_interval(elapsed)
        if rate is None and elapsed:
            rate = (n - initial) / elapsed
        inv_rate = 1 / rate if rate else None
        format_sizeof = tqdm.format_sizeof
        rate_noinv_fmt = ((format_sizeof(rate) if unit_scale else '{0:5.2f}'.format(rate)) if rate else '?') + unit + '/s'
        rate_inv_fmt = ((format_sizeof(inv_rate) if unit_scale else '{0:5.2f}'.format(inv_rate)) if inv_rate else '?') + 's/' + unit
        rate_fmt = rate_inv_fmt if inv_rate and inv_rate > 1 else rate_noinv_fmt
        if unit_scale:
            n_fmt = format_sizeof(n, divisor=unit_divisor)
            total_fmt = format_sizeof(total, divisor=unit_divisor) if total is not None else '?'
        else:
            n_fmt = str(n)
            total_fmt = str(total) if total is not None else '?'
        try:
            postfix = ', ' + postfix if postfix else ''
        except TypeError:
            pass
        remaining = (total - n) / rate if rate and total else 0
        remaining_str = tqdm.format_interval(remaining) if rate else '?'
        try:
            eta_dt = datetime.now() + timedelta(seconds=remaining) if rate and total else datetime.utcfromtimestamp(0)
        except OverflowError:
            eta_dt = datetime.max
        if prefix:
            bool_prefix_colon_already = prefix[-2:] == ': '
            l_bar = prefix if bool_prefix_colon_already else prefix + ': '
        else:
            l_bar = ''
        r_bar = f'| {n_fmt}/{total_fmt} [{elapsed_str}<{remaining_str}, {rate_fmt}{postfix}]'
        format_dict = {'n': n, 'n_fmt': n_fmt, 'total': total, 'total_fmt': total_fmt, 'elapsed': elapsed_str, 'elapsed_s': elapsed, 'ncols': ncols, 'desc': prefix or '', 'unit': unit, 'rate': inv_rate if inv_rate and inv_rate > 1 else rate, 'rate_fmt': rate_fmt, 'rate_noinv': rate, 'rate_noinv_fmt': rate_noinv_fmt, 'rate_inv': inv_rate, 'rate_inv_fmt': rate_inv_fmt, 'postfix': postfix, 'unit_divisor': unit_divisor, 'colour': colour, 'remaining': remaining_str, 'remaining_s': remaining, 'l_bar': l_bar, 'r_bar': r_bar, 'eta': eta_dt, **extra_kwargs}
        if total:
            frac = n / total
            percentage = frac * 100
            l_bar += '{0:3.0f}%|'.format(percentage)
            if ncols == 0:
                return l_bar[:-1] + r_bar[1:]
            format_dict.update(l_bar=l_bar)
            if bar_format:
                format_dict.update(percentage=percentage)
                if not prefix:
                    bar_format = bar_format.replace('{desc}: ', '')
            else:
                bar_format = '{l_bar}{bar}{r_bar}'
            full_bar = FormatReplace()
            nobar = bar_format.format(bar=full_bar, **format_dict)
            if not full_bar.format_called:
                return nobar
            full_bar = Bar(frac, max(1, ncols - disp_len(nobar)) if ncols else 10, charset=Bar.ASCII if ascii is True else ascii or Bar.UTF, colour=colour)
            if not _is_ascii(full_bar.charset) and _is_ascii(bar_format):
                bar_format = str(bar_format)
            res = bar_format.format(bar=full_bar, **format_dict)
            return disp_trim(res, ncols) if ncols else res
        elif bar_format:
            l_bar += '|'
            format_dict.update(l_bar=l_bar, percentage=0)
            full_bar = FormatReplace()
            nobar = bar_format.format(bar=full_bar, **format_dict)
            if not full_bar.format_called:
                return nobar
            full_bar = Bar(0, max(1, ncols - disp_len(nobar)) if ncols else 10, charset=Bar.BLANK, colour=colour)
            res = bar_format.format(bar=full_bar, **format_dict)
            return disp_trim(res, ncols) if ncols else res
        else:
            return f"{(prefix + ': ' if prefix else '')}{n_fmt}{unit} [{elapsed_str}, {rate_fmt}{postfix}]"

    def __new__(cls, *_, **__):
        if False:
            i = 10
            return i + 15
        instance = object.__new__(cls)
        with cls.get_lock():
            cls._instances.add(instance)
            if cls.monitor_interval and (cls.monitor is None or not cls.monitor.report()):
                try:
                    cls.monitor = TMonitor(cls, cls.monitor_interval)
                except Exception as e:
                    warn('tqdm:disabling monitor support (monitor_interval = 0) due to:\n' + str(e), TqdmMonitorWarning, stacklevel=2)
                    cls.monitor_interval = 0
        return instance

    @classmethod
    def _get_free_pos(cls, instance=None):
        if False:
            for i in range(10):
                print('nop')
        'Skips specified instance.'
        positions = {abs(inst.pos) for inst in cls._instances if inst is not instance and hasattr(inst, 'pos')}
        return min(set(range(len(positions) + 1)).difference(positions))

    @classmethod
    def _decr_instances(cls, instance):
        if False:
            while True:
                i = 10
        '\n        Remove from list and reposition another unfixed bar\n        to fill the new gap.\n\n        This means that by default (where all nested bars are unfixed),\n        order is not maintained but screen flicker/blank space is minimised.\n        (tqdm<=4.44.1 moved ALL subsequent unfixed bars up.)\n        '
        with cls._lock:
            try:
                cls._instances.remove(instance)
            except KeyError:
                pass
            if not instance.gui:
                last = (instance.nrows or 20) - 1
                instances = list(filter(lambda i: hasattr(i, 'pos') and last <= i.pos, cls._instances))
                if instances:
                    inst = min(instances, key=lambda i: i.pos)
                    inst.clear(nolock=True)
                    inst.pos = abs(instance.pos)

    @classmethod
    def write(cls, s, file=None, end='\n', nolock=False):
        if False:
            i = 10
            return i + 15
        'Print a message via tqdm (without overlap with bars).'
        fp = file if file is not None else sys.stdout
        with cls.external_write_mode(file=file, nolock=nolock):
            fp.write(s)
            fp.write(end)

    @classmethod
    @contextmanager
    def external_write_mode(cls, file=None, nolock=False):
        if False:
            print('Hello World!')
        '\n        Disable tqdm within context and refresh tqdm when exits.\n        Useful when writing to standard output stream\n        '
        fp = file if file is not None else sys.stdout
        try:
            if not nolock:
                cls.get_lock().acquire()
            inst_cleared = []
            for inst in getattr(cls, '_instances', []):
                if hasattr(inst, 'start_t') and (inst.fp == fp or all((f in (sys.stdout, sys.stderr) for f in (fp, inst.fp)))):
                    inst.clear(nolock=True)
                    inst_cleared.append(inst)
            yield
            for inst in inst_cleared:
                inst.refresh(nolock=True)
        finally:
            if not nolock:
                cls._lock.release()

    @classmethod
    def set_lock(cls, lock):
        if False:
            while True:
                i = 10
        'Set the global lock.'
        cls._lock = lock

    @classmethod
    def get_lock(cls):
        if False:
            print('Hello World!')
        'Get the global lock. Construct it if it does not exist.'
        if not hasattr(cls, '_lock'):
            cls._lock = TqdmDefaultWriteLock()
        return cls._lock

    @classmethod
    def pandas(cls, **tqdm_kwargs):
        if False:
            return 10
        '\n        Registers the current `tqdm` class with\n            pandas.core.\n            ( frame.DataFrame\n            | series.Series\n            | groupby.(generic.)DataFrameGroupBy\n            | groupby.(generic.)SeriesGroupBy\n            ).progress_apply\n\n        A new instance will be created every time `progress_apply` is called,\n        and each instance will automatically `close()` upon completion.\n\n        Parameters\n        ----------\n        tqdm_kwargs  : arguments for the tqdm instance\n\n        Examples\n        --------\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>> from tqdm import tqdm\n        >>> from tqdm.gui import tqdm as tqdm_gui\n        >>>\n        >>> df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))\n        >>> tqdm.pandas(ncols=50)  # can use tqdm_gui, optional kwargs, etc\n        >>> # Now you can use `progress_apply` instead of `apply`\n        >>> df.groupby(0).progress_apply(lambda x: x**2)\n\n        References\n        ----------\n        <https://stackoverflow.com/questions/18603270/        progress-indicator-during-pandas-operations-python>\n        '
        from warnings import catch_warnings, simplefilter
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
        try:
            with catch_warnings():
                simplefilter('ignore', category=FutureWarning)
                from pandas import Panel
        except ImportError:
            Panel = None
        (Rolling, Expanding) = (None, None)
        try:
            from pandas.core.window.rolling import _Rolling_and_Expanding
        except ImportError:
            try:
                from pandas.core.window import _Rolling_and_Expanding
            except ImportError:
                try:
                    from pandas.core.window.expanding import Expanding
                    from pandas.core.window.rolling import Rolling
                    _Rolling_and_Expanding = (Rolling, Expanding)
                except ImportError:
                    _Rolling_and_Expanding = None
        try:
            from pandas.core.groupby.generic import SeriesGroupBy
            from pandas.core.groupby.generic import DataFrameGroupBy
        except ImportError:
            try:
                from pandas.core.groupby.groupby import DataFrameGroupBy, SeriesGroupBy
            except ImportError:
                from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
        try:
            from pandas.core.groupby.groupby import GroupBy
        except ImportError:
            from pandas.core.groupby import GroupBy
        try:
            from pandas.core.groupby.groupby import PanelGroupBy
        except ImportError:
            try:
                from pandas.core.groupby import PanelGroupBy
            except ImportError:
                PanelGroupBy = None
        tqdm_kwargs = tqdm_kwargs.copy()
        deprecated_t = [tqdm_kwargs.pop('deprecated_t', None)]

        def inner_generator(df_function='apply'):
            if False:
                i = 10
                return i + 15

            def inner(df, func, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Parameters\n                ----------\n                df  : (DataFrame|Series)[GroupBy]\n                    Data (may be grouped).\n                func  : function\n                    To be applied on the (grouped) data.\n                **kwargs  : optional\n                    Transmitted to `df.apply()`.\n                '
                total = tqdm_kwargs.pop('total', getattr(df, 'ngroups', None))
                if total is None:
                    if df_function == 'applymap':
                        total = df.size
                    elif isinstance(df, Series):
                        total = len(df)
                    elif _Rolling_and_Expanding is None or not isinstance(df, _Rolling_and_Expanding):
                        axis = kwargs.get('axis', 0)
                        if axis == 'index':
                            axis = 0
                        elif axis == 'columns':
                            axis = 1
                        total = df.size // df.shape[axis]
                if deprecated_t[0] is not None:
                    t = deprecated_t[0]
                    deprecated_t[0] = None
                else:
                    t = cls(total=total, **tqdm_kwargs)
                if len(args) > 0:
                    TqdmDeprecationWarning('Except func, normal arguments are intentionally' + ' not supported by' + ' `(DataFrame|Series|GroupBy).progress_apply`.' + ' Use keyword arguments instead.', fp_write=getattr(t.fp, 'write', sys.stderr.write))
                try:
                    from pandas.core.common import is_builtin_func
                except ImportError:
                    is_builtin_func = df._is_builtin_func
                try:
                    func = is_builtin_func(func)
                except TypeError:
                    pass

                def wrapper(*args, **kwargs):
                    if False:
                        i = 10
                        return i + 15
                    t.update(n=1 if not t.total or t.n < t.total else 0)
                    return func(*args, **kwargs)
                try:
                    return getattr(df, df_function)(wrapper, **kwargs)
                finally:
                    t.close()
            return inner
        Series.progress_apply = inner_generator()
        SeriesGroupBy.progress_apply = inner_generator()
        Series.progress_map = inner_generator('map')
        SeriesGroupBy.progress_map = inner_generator('map')
        DataFrame.progress_apply = inner_generator()
        DataFrameGroupBy.progress_apply = inner_generator()
        DataFrame.progress_applymap = inner_generator('applymap')
        if Panel is not None:
            Panel.progress_apply = inner_generator()
        if PanelGroupBy is not None:
            PanelGroupBy.progress_apply = inner_generator()
        GroupBy.progress_apply = inner_generator()
        GroupBy.progress_aggregate = inner_generator('aggregate')
        GroupBy.progress_transform = inner_generator('transform')
        if Rolling is not None and Expanding is not None:
            Rolling.progress_apply = inner_generator()
            Expanding.progress_apply = inner_generator()
        elif _Rolling_and_Expanding is not None:
            _Rolling_and_Expanding.progress_apply = inner_generator()

    @envwrap('TQDM_', is_method=True, types={'total': float, 'ncols': int, 'miniters': float, 'position': int, 'nrows': int})
    def __init__(self, iterable=None, desc=None, total=None, leave=True, file=None, ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None, ascii=None, disable=False, unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0, position=None, postfix=None, unit_divisor=1000, write_bytes=False, lock_args=None, nrows=None, colour=None, delay=0.0, gui=False, **kwargs):
        if False:
            while True:
                i = 10
        'see tqdm.tqdm for arguments'
        if file is None:
            file = sys.stderr
        if write_bytes:
            file = SimpleTextIOWrapper(file, encoding=getattr(file, 'encoding', None) or 'utf-8')
        file = DisableOnWriteError(file, tqdm_instance=self)
        if disable is None and hasattr(file, 'isatty') and (not file.isatty()):
            disable = True
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = None
        if total == float('inf'):
            total = None
        if disable:
            self.iterable = iterable
            self.disable = disable
            with self._lock:
                self.pos = self._get_free_pos(self)
                self._instances.remove(self)
            self.n = initial
            self.total = total
            self.leave = leave
            return
        if kwargs:
            self.disable = True
            with self._lock:
                self.pos = self._get_free_pos(self)
                self._instances.remove(self)
            raise TqdmDeprecationWarning('`nested` is deprecated and automated.\nUse `position` instead for manual control.\n', fp_write=getattr(file, 'write', sys.stderr.write)) if 'nested' in kwargs else TqdmKeyError('Unknown argument(s): ' + str(kwargs))
        if (ncols is None or nrows is None) and file in (sys.stderr, sys.stdout) or dynamic_ncols:
            if dynamic_ncols:
                dynamic_ncols = _screen_shape_wrapper()
                if dynamic_ncols:
                    (ncols, nrows) = dynamic_ncols(file)
            else:
                _dynamic_ncols = _screen_shape_wrapper()
                if _dynamic_ncols:
                    (_ncols, _nrows) = _dynamic_ncols(file)
                    if ncols is None:
                        ncols = _ncols
                    if nrows is None:
                        nrows = _nrows
        if miniters is None:
            miniters = 0
            dynamic_miniters = True
        else:
            dynamic_miniters = False
        if mininterval is None:
            mininterval = 0
        if maxinterval is None:
            maxinterval = 0
        if ascii is None:
            ascii = not _supports_unicode(file)
        if bar_format and ascii is not True and (not _is_ascii(ascii)):
            bar_format = str(bar_format)
        if smoothing is None:
            smoothing = 0
        self.iterable = iterable
        self.desc = desc or ''
        self.total = total
        self.leave = leave
        self.fp = file
        self.ncols = ncols
        self.nrows = nrows
        self.mininterval = mininterval
        self.maxinterval = maxinterval
        self.miniters = miniters
        self.dynamic_miniters = dynamic_miniters
        self.ascii = ascii
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        self.initial = initial
        self.lock_args = lock_args
        self.delay = delay
        self.gui = gui
        self.dynamic_ncols = dynamic_ncols
        self.smoothing = smoothing
        self._ema_dn = EMA(smoothing)
        self._ema_dt = EMA(smoothing)
        self._ema_miniters = EMA(smoothing)
        self.bar_format = bar_format
        self.postfix = None
        self.colour = colour
        self._time = time
        if postfix:
            try:
                self.set_postfix(refresh=False, **postfix)
            except TypeError:
                self.postfix = postfix
        self.last_print_n = initial
        self.n = initial
        with self._lock:
            self.pos = self._get_free_pos(self) if position is None else -position
        if not gui:
            self.sp = self.status_printer(self.fp)
            if delay <= 0:
                self.refresh(lock_args=self.lock_args)
        self.last_print_t = self._time()
        self.start_t = self.last_print_t

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.total is not None:
            return self.total > 0
        if self.iterable is None:
            raise TypeError('bool() undefined when iterable == total == None')
        return bool(self.iterable)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.total if self.iterable is None else self.iterable.shape[0] if hasattr(self.iterable, 'shape') else len(self.iterable) if hasattr(self.iterable, '__len__') else self.iterable.__length_hint__() if hasattr(self.iterable, '__length_hint__') else getattr(self, 'total', None)

    def __reversed__(self):
        if False:
            while True:
                i = 10
        try:
            orig = self.iterable
        except AttributeError:
            raise TypeError("'tqdm' object is not reversible")
        else:
            self.iterable = reversed(self.iterable)
            return self.__iter__()
        finally:
            self.iterable = orig

    def __contains__(self, item):
        if False:
            while True:
                i = 10
        contains = getattr(self.iterable, '__contains__', None)
        return contains(item) if contains is not None else item in self.__iter__()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            print('Hello World!')
        try:
            self.close()
        except AttributeError:
            if (exc_type, exc_value, traceback) == (None, None, None):
                raise
            warn('AttributeError ignored', TqdmWarning, stacklevel=2)

    def __del__(self):
        if False:
            print('Hello World!')
        self.close()

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.format_meter(**self.format_dict)

    @property
    def _comparable(self):
        if False:
            while True:
                i = 10
        return abs(getattr(self, 'pos', 1 << 31))

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return id(self)

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Backward-compatibility to use: for x in tqdm(iterable)'
        iterable = self.iterable
        if self.disable:
            for obj in iterable:
                yield obj
            return
        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time
        try:
            for obj in iterable:
                yield obj
                n += 1
                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()

    def update(self, n=1):
        if False:
            while True:
                i = 10
        '\n        Manually update the progress bar, useful for streams\n        such as reading files.\n        E.g.:\n        >>> t = tqdm(total=filesize) # Initialise\n        >>> for current_buffer in stream:\n        ...    ...\n        ...    t.update(len(current_buffer))\n        >>> t.close()\n        The last line is highly recommended, but possibly not necessary if\n        `t.update()` will be called in such a way that `filesize` will be\n        exactly reached and printed.\n\n        Parameters\n        ----------\n        n  : int or float, optional\n            Increment to add to the internal counter of iterations\n            [default: 1]. If using float, consider specifying `{n:.3f}`\n            or similar in `bar_format`, or specifying `unit_scale`.\n\n        Returns\n        -------\n        out  : bool or None\n            True if a `display()` was triggered.\n        '
        if self.disable:
            return
        if n < 0:
            self.last_print_n += n
        self.n += n
        if self.n - self.last_print_n >= self.miniters:
            cur_t = self._time()
            dt = cur_t - self.last_print_t
            if dt >= self.mininterval and cur_t >= self.start_t + self.delay:
                cur_t = self._time()
                dn = self.n - self.last_print_n
                if self.smoothing and dt and dn:
                    self._ema_dn(dn)
                    self._ema_dt(dt)
                self.refresh(lock_args=self.lock_args)
                if self.dynamic_miniters:
                    if self.maxinterval and dt >= self.maxinterval:
                        self.miniters = dn * (self.mininterval or self.maxinterval) / dt
                    elif self.smoothing:
                        self.miniters = self._ema_miniters(dn * (self.mininterval / dt if self.mininterval and dt else 1))
                    else:
                        self.miniters = max(self.miniters, dn)
                self.last_print_n = self.n
                self.last_print_t = cur_t
                return True

    def close(self):
        if False:
            while True:
                i = 10
        'Cleanup and (if leave=False) close the progressbar.'
        if self.disable:
            return
        self.disable = True
        pos = abs(self.pos)
        self._decr_instances(self)
        if self.last_print_t < self.start_t + self.delay:
            return
        if getattr(self, 'sp', None) is None:
            return

        def fp_write(s):
            if False:
                for i in range(10):
                    print('nop')
            self.fp.write(str(s))
        try:
            fp_write('')
        except ValueError as e:
            if 'closed' in str(e):
                return
            raise
        leave = pos == 0 if self.leave is None else self.leave
        with self._lock:
            if leave:
                self._ema_dt = lambda : None
                self.display(pos=0)
                fp_write('\n')
            elif self.display(msg='', pos=pos) and (not pos):
                fp_write('\r')

    def clear(self, nolock=False):
        if False:
            for i in range(10):
                print('nop')
        'Clear current bar display.'
        if self.disable:
            return
        if not nolock:
            self._lock.acquire()
        pos = abs(self.pos)
        if pos < (self.nrows or 20):
            self.moveto(pos)
            self.sp('')
            self.fp.write('\r')
            self.moveto(-pos)
        if not nolock:
            self._lock.release()

    def refresh(self, nolock=False, lock_args=None):
        if False:
            return 10
        "\n        Force refresh the display of this bar.\n\n        Parameters\n        ----------\n        nolock  : bool, optional\n            If `True`, does not lock.\n            If [default: `False`]: calls `acquire()` on internal lock.\n        lock_args  : tuple, optional\n            Passed to internal lock's `acquire()`.\n            If specified, will only `display()` if `acquire()` returns `True`.\n        "
        if self.disable:
            return
        if not nolock:
            if lock_args:
                if not self._lock.acquire(*lock_args):
                    return False
            else:
                self._lock.acquire()
        self.display()
        if not nolock:
            self._lock.release()
        return True

    def unpause(self):
        if False:
            for i in range(10):
                print('nop')
        'Restart tqdm timer from last print time.'
        if self.disable:
            return
        cur_t = self._time()
        self.start_t += cur_t - self.last_print_t
        self.last_print_t = cur_t

    def reset(self, total=None):
        if False:
            return 10
        '\n        Resets to 0 iterations for repeated use.\n\n        Consider combining with `leave=True`.\n\n        Parameters\n        ----------\n        total  : int or float, optional. Total to use for the new bar.\n        '
        self.n = 0
        if total is not None:
            self.total = total
        if self.disable:
            return
        self.last_print_n = 0
        self.last_print_t = self.start_t = self._time()
        self._ema_dn = EMA(self.smoothing)
        self._ema_dt = EMA(self.smoothing)
        self._ema_miniters = EMA(self.smoothing)
        self.refresh()

    def set_description(self, desc=None, refresh=True):
        if False:
            return 10
        '\n        Set/modify description of the progress bar.\n\n        Parameters\n        ----------\n        desc  : str, optional\n        refresh  : bool, optional\n            Forces refresh [default: True].\n        '
        self.desc = desc + ': ' if desc else ''
        if refresh:
            self.refresh()

    def set_description_str(self, desc=None, refresh=True):
        if False:
            return 10
        "Set/modify description without ': ' appended."
        self.desc = desc or ''
        if refresh:
            self.refresh()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Set/modify postfix (additional stats)\n        with automatic formatting based on datatype.\n\n        Parameters\n        ----------\n        ordered_dict  : dict or OrderedDict, optional\n        refresh  : bool, optional\n            Forces refresh [default: True].\n        kwargs  : dict, optional\n        '
        postfix = OrderedDict([] if ordered_dict is None else ordered_dict)
        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]
        for key in postfix.keys():
            if isinstance(postfix[key], Number):
                postfix[key] = self.format_num(postfix[key])
            elif not isinstance(postfix[key], str):
                postfix[key] = str(postfix[key])
        self.postfix = ', '.join((key + '=' + postfix[key].strip() for key in postfix.keys()))
        if refresh:
            self.refresh()

    def set_postfix_str(self, s='', refresh=True):
        if False:
            return 10
        '\n        Postfix without dictionary expansion, similar to prefix handling.\n        '
        self.postfix = str(s)
        if refresh:
            self.refresh()

    def moveto(self, n):
        if False:
            return 10
        self.fp.write('\n' * n + _term_move_up() * -n)
        getattr(self.fp, 'flush', lambda : None)()

    @property
    def format_dict(self):
        if False:
            print('Hello World!')
        'Public API for read-only member access.'
        if self.disable and (not hasattr(self, 'unit')):
            return defaultdict(lambda : None, {'n': self.n, 'total': self.total, 'elapsed': 0, 'unit': 'it'})
        if self.dynamic_ncols:
            (self.ncols, self.nrows) = self.dynamic_ncols(self.fp)
        return {'n': self.n, 'total': self.total, 'elapsed': self._time() - self.start_t if hasattr(self, 'start_t') else 0, 'ncols': self.ncols, 'nrows': self.nrows, 'prefix': self.desc, 'ascii': self.ascii, 'unit': self.unit, 'unit_scale': self.unit_scale, 'rate': self._ema_dn() / self._ema_dt() if self._ema_dt() else None, 'bar_format': self.bar_format, 'postfix': self.postfix, 'unit_divisor': self.unit_divisor, 'initial': self.initial, 'colour': self.colour}

    def display(self, msg=None, pos=None):
        if False:
            while True:
                i = 10
        '\n        Use `self.sp` to display `msg` in the specified `pos`.\n\n        Consider overloading this function when inheriting to use e.g.:\n        `self.some_frontend(**self.format_dict)` instead of `self.sp`.\n\n        Parameters\n        ----------\n        msg  : str, optional. What to display (default: `repr(self)`).\n        pos  : int, optional. Position to `moveto`\n          (default: `abs(self.pos)`).\n        '
        if pos is None:
            pos = abs(self.pos)
        nrows = self.nrows or 20
        if pos >= nrows - 1:
            if pos >= nrows:
                return False
            if msg or msg is None:
                msg = ' ... (more hidden) ...'
        if not hasattr(self, 'sp'):
            raise TqdmDeprecationWarning('Please use `tqdm.gui.tqdm(...)` instead of `tqdm(..., gui=True)`\n', fp_write=getattr(self.fp, 'write', sys.stderr.write))
        if pos:
            self.moveto(pos)
        self.sp(self.__str__() if msg is None else msg)
        if pos:
            self.moveto(-pos)
        return True

    @classmethod
    @contextmanager
    def wrapattr(cls, stream, method, total=None, bytes=True, **tqdm_kwargs):
        if False:
            while True:
                i = 10
        '\n        stream  : file-like object.\n        method  : str, "read" or "write". The result of `read()` and\n            the first argument of `write()` should have a `len()`.\n\n        >>> with tqdm.wrapattr(file_obj, "read", total=file_obj.size) as fobj:\n        ...     while True:\n        ...         chunk = fobj.read(chunk_size)\n        ...         if not chunk:\n        ...             break\n        '
        with cls(total=total, **tqdm_kwargs) as t:
            if bytes:
                t.unit = 'B'
                t.unit_scale = True
                t.unit_divisor = 1024
            yield CallbackIOWrapper(t.update, stream, method)

def trange(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Shortcut for tqdm(range(*args), **kwargs).'
    return tqdm(range(*args), **kwargs)