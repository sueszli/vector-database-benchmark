import csv
import os
import re
import sys
from contextlib import contextmanager
from functools import wraps
from warnings import catch_warnings, simplefilter
from pytest import importorskip, mark, raises, skip
from tqdm import TqdmDeprecationWarning, TqdmWarning, tqdm, trange
from tqdm.contrib import DummyTqdmFile
from tqdm.std import EMA, Bar
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from io import IOBase
from io import BytesIO

class DeprecationError(Exception):
    pass
if getattr(StringIO, '__exit__', False) and getattr(StringIO, '__enter__', False):

    def closing(arg):
        if False:
            print('Hello World!')
        return arg
else:
    from contextlib import closing
nt_and_no_colorama = False
if os.name == 'nt':
    try:
        import colorama
    except ImportError:
        nt_and_no_colorama = True
CTRLCHR = ['\\r', '\\n', '\\x1b\\[A']
RE_rate = re.compile('[^\\d](\\d[.\\d]+)it/s')
RE_ctrlchr = re.compile('(%s)' % '|'.join(CTRLCHR))
RE_ctrlchr_excl = re.compile('|'.join(CTRLCHR))
RE_pos = re.compile('([\\r\\n]+((pos\\d+) bar:\\s+\\d+%|\\s{3,6})?[^\\r\\n]*)')

def pos_line_diff(res_list, expected_list, raise_nonempty=True):
    if False:
        return 10
    '\n    Return differences between two bar output lists.\n    To be used with `RE_pos`\n    '
    res = [(r, e) for (r, e) in zip(res_list, expected_list) for pos in [len(e) - len(e.lstrip('\n'))] if r != e if not r.startswith(e) or not (any((r.endswith(end + i * '\x1b[A') for i in range(pos + 1) for end in [']', '  '])) or '100%' in r or r == '\n') or r[(-1 - pos) * len('\x1b[A'):] == '\x1b[A']
    if raise_nonempty and (res or len(res_list) != len(expected_list)):
        if len(res_list) < len(expected_list):
            res.extend([(None, e) for e in expected_list[len(res_list):]])
        elif len(res_list) > len(expected_list):
            res.extend([(r, None) for r in res_list[len(expected_list):]])
        raise AssertionError('Got => Expected\n' + '\n'.join(('%r => %r' % i for i in res)))
    return res

class DiscreteTimer(object):
    """Virtual discrete time manager, to precisely control time for tests"""

    def __init__(self):
        if False:
            return 10
        self.t = 0.0

    def sleep(self, t):
        if False:
            while True:
                i = 10
        'Sleep = increment the time counter (almost no CPU used)'
        self.t += t

    def time(self):
        if False:
            i = 10
            return i + 15
        'Get the current time'
        return self.t

def cpu_timify(t, timer=None):
    if False:
        i = 10
        return i + 15
    'Force tqdm to use the specified timer instead of system-wide time()'
    if timer is None:
        timer = DiscreteTimer()
    t._time = timer.time
    t._sleep = timer.sleep
    t.start_t = t.last_print_t = t._time()
    return timer

class UnicodeIO(IOBase):
    """Unicode version of StringIO"""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(UnicodeIO, self).__init__(*args, **kwargs)
        self.encoding = 'U8'
        self.text = ''
        self.cursor = 0

    def __len__(self):
        if False:
            return 10
        return len(self.text)

    def seek(self, offset):
        if False:
            print('Hello World!')
        self.cursor = offset

    def tell(self):
        if False:
            while True:
                i = 10
        return self.cursor

    def write(self, s):
        if False:
            for i in range(10):
                print('nop')
        self.text = self.text[:self.cursor] + s + self.text[self.cursor + len(s):]
        self.cursor += len(s)

    def read(self, n=-1):
        if False:
            while True:
                i = 10
        _cur = self.cursor
        self.cursor = len(self) if n < 0 else min(_cur + n, len(self))
        return self.text[_cur:self.cursor]

    def getvalue(self):
        if False:
            return 10
        return self.text

def get_bar(all_bars, i=None):
    if False:
        return 10
    'Get a specific update from a whole bar traceback'
    bars_split = RE_ctrlchr_excl.split(all_bars)
    bars_split = list(filter(None, bars_split))
    return bars_split if i is None else bars_split[i]

def progressbar_rate(bar_str):
    if False:
        return 10
    return float(RE_rate.search(bar_str).group(1))

def squash_ctrlchars(s):
    if False:
        for i in range(10):
            print('nop')
    'Apply control characters in a string just like a terminal display'
    curline = 0
    lines = ['']
    for nextctrl in filter(None, RE_ctrlchr.split(s)):
        if nextctrl == '\r':
            lines[curline] = ''
        elif nextctrl == '\n':
            if curline >= len(lines) - 1:
                lines.append('')
            curline += 1
        elif nextctrl == '\x1b[A':
            if curline > 0:
                curline -= 1
            else:
                raise ValueError('Cannot go further up')
        else:
            lines[curline] += nextctrl
    return lines

def test_format_interval():
    if False:
        print('Hello World!')
    'Test time interval format'
    format_interval = tqdm.format_interval
    assert format_interval(60) == '01:00'
    assert format_interval(6160) == '1:42:40'
    assert format_interval(238113) == '66:08:33'

def test_format_num():
    if False:
        return 10
    'Test number format'
    format_num = tqdm.format_num
    assert float(format_num(1337)) == 1337
    assert format_num(int(1000000.0)) == '1e+6'
    assert format_num(1239876) == '1239876'

def test_format_meter():
    if False:
        print('Hello World!')
    'Test statistics and progress bar formatting'
    try:
        unich = unichr
    except NameError:
        unich = chr
    format_meter = tqdm.format_meter
    assert format_meter(0, 1000, 13) == '  0%|          | 0/1000 [00:13<?, ?it/s]'
    assert format_meter(0, 1000, 13, ncols=68, prefix='desc: ') == 'desc:   0%|                                | 0/1000 [00:13<?, ?it/s]'
    assert format_meter(231, 1000, 392) == ' 23%|' + unich(9608) * 2 + unich(9614) + '       | 231/1000 [06:32<21:44,  1.70s/it]'
    assert format_meter(10000, 1000, 13) == '10000it [00:13, 769.23it/s]'
    assert format_meter(231, 1000, 392, ncols=56, ascii=True) == ' 23%|' + '#' * 3 + '6' + '            | 231/1000 [06:32<21:44,  1.70s/it]'
    assert format_meter(100000, 1000, 13, unit_scale=True, unit='iB') == '100kiB [00:13, 7.69kiB/s]'
    assert format_meter(100, 1000, 12, ncols=0, rate=7.33) == ' 10% 100/1000 [00:12<02:02,  7.33it/s]'
    assert format_meter(0, 1000, 13, ncols=10, bar_format='************{bar:10}$$$$$$$$$$') == '**********'
    assert format_meter(0, 1000, 13, ncols=20, bar_format='************{bar:10}$$$$$$$$$$') == '************        '
    assert format_meter(0, 1000, 13, ncols=30, bar_format='************{bar:10}$$$$$$$$$$') == '************          $$$$$$$$'
    assert format_meter(0, 1000, 13, ncols=10, bar_format='*****\x1b[22m****\x1b[0m***{bar:10}$$$$$$$$$$') == '*****\x1b[22m****\x1b[0m*\x1b[0m'
    assert format_meter(0, 1000, 13, ncols=10, bar_format='*****\x1b[22m*****\x1b[0m**{bar:10}$$$$$$$$$$') == '*****\x1b[22m*****\x1b[0m'
    assert format_meter(0, 1000, 13, ncols=10, bar_format='*****\x1b[22m******\x1b[0m*{bar:10}$$$$$$$$$$') == '*****\x1b[22m*****\x1b[0m'
    assert format_meter(20, 100, 12, ncols=13, rate=8.1, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}') == ' 20%|' + unich(9615) + '|20/100'
    assert format_meter(20, 100, 12, ncols=14, rate=8.1, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}') == ' 20%|' + unich(9613) + ' |20/100'
    assert format_meter(0, 1000, 13, ncols=68, prefix='ｆｕｌｌｗｉｄｔｈ: ') == 'ｆｕｌｌｗｉｄｔｈ:   0%|                  | 0/1000 [00:13<?, ?it/s]'
    assert format_meter(0, 1000, 13, ncols=68, prefix='ニッポン [ﾆｯﾎﾟﾝ]: ') == 'ニッポン [ﾆｯﾎﾟﾝ]:   0%|                    | 0/1000 [00:13<?, ?it/s]'
    assert format_meter(20, 100, 12, ncols=2, rate=8.1, bar_format='{bar}') == unich(9613) + ' '
    assert format_meter(20, 100, 12, ncols=7, rate=8.1, bar_format='{l_bar}{bar}') == ' 20%|' + unich(9613) + ' '
    assert format_meter(20, 100, 12, ncols=6, rate=8.1, bar_format='{bar}|test') == unich(9615) + '|test'

def test_ansi_escape_codes():
    if False:
        return 10
    'Test stripping of ANSI escape codes'
    ansi = {'BOLD': '\x1b[1m', 'RED': '\x1b[91m', 'END': '\x1b[0m'}
    desc_raw = '{BOLD}{RED}Colored{END} description'
    ncols = 123
    desc_stripped = desc_raw.format(BOLD='', RED='', END='')
    meter = tqdm.format_meter(0, 100, 0, ncols=ncols, prefix=desc_stripped)
    assert len(meter) == ncols
    desc = desc_raw.format(**ansi)
    meter = tqdm.format_meter(0, 100, 0, ncols=ncols, prefix=desc)
    ansi_len = len(desc) - len(desc_stripped) + len(ansi['END'])
    assert len(meter) == ncols + ansi_len

def test_si_format():
    if False:
        for i in range(10):
            print('nop')
    'Test SI unit prefixes'
    format_meter = tqdm.format_meter
    assert '9.00 ' in format_meter(1, 9, 1, unit_scale=True, unit='B')
    assert '99.0 ' in format_meter(1, 99, 1, unit_scale=True)
    assert '999 ' in format_meter(1, 999, 1, unit_scale=True)
    assert '9.99k ' in format_meter(1, 9994, 1, unit_scale=True)
    assert '10.0k ' in format_meter(1, 9999, 1, unit_scale=True)
    assert '99.5k ' in format_meter(1, 99499, 1, unit_scale=True)
    assert '100k ' in format_meter(1, 99999, 1, unit_scale=True)
    assert '1.00M ' in format_meter(1, 999999, 1, unit_scale=True)
    assert '1.00G ' in format_meter(1, 999999999, 1, unit_scale=True)
    assert '1.00T ' in format_meter(1, 999999999999, 1, unit_scale=True)
    assert '1.00P ' in format_meter(1, 999999999999999, 1, unit_scale=True)
    assert '1.00E ' in format_meter(1, 999999999999999999, 1, unit_scale=True)
    assert '1.00Z ' in format_meter(1, 999999999999999999999, 1, unit_scale=True)
    assert '1.0Y ' in format_meter(1, 999999999999999999999999, 1, unit_scale=True)
    assert '10.0Y ' in format_meter(1, 9999999999999999999999999, 1, unit_scale=True)
    assert '100.0Y ' in format_meter(1, 99999999999999999999999999, 1, unit_scale=True)
    assert '1000.0Y ' in format_meter(1, 999999999999999999999999999, 1, unit_scale=True)

def test_bar_formatspec():
    if False:
        return 10
    'Test Bar.__format__ spec'
    assert '{0:5a}'.format(Bar(0.3)) == '#5   '
    assert '{0:2}'.format(Bar(0.5, charset=' .oO0')) == '0 '
    assert '{0:2a}'.format(Bar(0.5, charset=' .oO0')) == '# '
    assert '{0:-6a}'.format(Bar(0.5, 10)) == '##  '
    assert '{0:2b}'.format(Bar(0.5, 10)) == '  '

def test_all_defaults():
    if False:
        while True:
            i = 10
    'Test default kwargs'
    with closing(UnicodeIO()) as our_file:
        with tqdm(range(10), file=our_file) as progressbar:
            assert len(progressbar) == 10
            for _ in progressbar:
                pass
    sys.stderr.write('\rTest default kwargs ... ')

class WriteTypeChecker(BytesIO):
    """File-like to assert the expected type is written"""

    def __init__(self, expected_type):
        if False:
            for i in range(10):
                print('nop')
        super(WriteTypeChecker, self).__init__()
        self.expected_type = expected_type

    def write(self, s):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(s, self.expected_type)

def test_native_string_io_for_default_file():
    if False:
        return 10
    'Native strings written to unspecified files'
    stderr = sys.stderr
    try:
        sys.stderr = WriteTypeChecker(expected_type=type(''))
        for _ in tqdm(range(3)):
            pass
        sys.stderr.encoding = None
        for _ in tqdm(range(3)):
            pass
    finally:
        sys.stderr = stderr

def test_unicode_string_io_for_specified_file():
    if False:
        print('Hello World!')
    'Unicode strings written to specified files'
    for _ in tqdm(range(3), file=WriteTypeChecker(expected_type=type(u''))):
        pass

def test_write_bytes():
    if False:
        for i in range(10):
            print('nop')
    'Test write_bytes argument with and without `file`'
    for _ in tqdm(range(3), file=WriteTypeChecker(expected_type=type(b'')), write_bytes=True):
        pass
    stderr = sys.stderr
    try:
        sys.stderr = WriteTypeChecker(expected_type=type(u''))
        for _ in tqdm(range(3), write_bytes=False):
            pass
    finally:
        sys.stderr = stderr

def test_iterate_over_csv_rows():
    if False:
        for i in range(10):
            print('nop')
    'Test csv iterator'
    with closing(StringIO()) as test_csv_file:
        writer = csv.writer(test_csv_file)
        for _ in range(3):
            writer.writerow(['test'] * 3)
        test_csv_file.seek(0)
        reader = csv.DictReader(test_csv_file, fieldnames=('row1', 'row2', 'row3'))
        with closing(StringIO()) as our_file:
            for _ in tqdm(reader, file=our_file):
                pass

def test_file_output():
    if False:
        for i in range(10):
            print('nop')
    'Test output to arbitrary file-like objects'
    with closing(StringIO()) as our_file:
        for i in tqdm(range(3), file=our_file):
            if i == 1:
                our_file.seek(0)
                assert '0/3' in our_file.read()

def test_leave_option():
    if False:
        print('Hello World!')
    'Test `leave=True` always prints info about the last iteration'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(3), file=our_file, leave=True):
            pass
        res = our_file.getvalue()
        assert '| 3/3 ' in res
        assert '\n' == res[-1]
    with closing(StringIO()) as our_file2:
        for _ in tqdm(range(3), file=our_file2, leave=False):
            pass
        assert '| 3/3 ' not in our_file2.getvalue()

def test_trange():
    if False:
        while True:
            i = 10
    'Test trange'
    with closing(StringIO()) as our_file:
        for _ in trange(3, file=our_file, leave=True):
            pass
        assert '| 3/3 ' in our_file.getvalue()
    with closing(StringIO()) as our_file2:
        for _ in trange(3, file=our_file2, leave=False):
            pass
        assert '| 3/3 ' not in our_file2.getvalue()

def test_min_interval():
    if False:
        for i in range(10):
            print('nop')
    'Test mininterval'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(3), file=our_file, mininterval=1e-10):
            pass
        assert '  0%|          | 0/3 [00:00<' in our_file.getvalue()

def test_max_interval():
    if False:
        print('Hello World!')
    'Test maxinterval'
    total = 100
    bigstep = 10
    smallstep = 5
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        with closing(StringIO()) as our_file2:
            t = tqdm(total=total, file=our_file, miniters=None, mininterval=0, smoothing=1, maxinterval=0.01)
            cpu_timify(t, timer)
            t2 = tqdm(total=total, file=our_file2, miniters=None, mininterval=0, smoothing=1, maxinterval=None)
            cpu_timify(t2, timer)
            assert t.dynamic_miniters
            assert t2.dynamic_miniters
            t.update(bigstep)
            t2.update(bigstep)
            for _ in range(4):
                t.update(smallstep)
                t2.update(smallstep)
                timer.sleep(1e-05)
            t.close()
            t2.close()
            assert '25%' not in our_file2.getvalue()
        assert '25%' not in our_file.getvalue()
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        with tqdm(total=total, file=our_file, miniters=None, mininterval=0, smoothing=1, maxinterval=0.0001) as t:
            cpu_timify(t, timer)
            t.update(bigstep)
            for _ in range(4):
                t.update(smallstep)
                timer.sleep(0.01)
            assert '25%' in our_file.getvalue()
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        with tqdm(range(total), file=our_file, miniters=None, mininterval=1e-05, smoothing=1, maxinterval=0.0001) as t2:
            cpu_timify(t2, timer)
            for i in t2:
                if i >= bigstep - 1 and (i - (bigstep - 1)) % smallstep == 0:
                    timer.sleep(0.01)
                if i >= 3 * bigstep:
                    break
        assert '15%' in our_file.getvalue()
    timer = DiscreteTimer()
    total = 1000
    mininterval = 0.1
    maxinterval = 10
    with closing(StringIO()) as our_file:
        with tqdm(total=total, file=our_file, miniters=None, smoothing=1, mininterval=mininterval, maxinterval=maxinterval) as tm1:
            with tqdm(total=total, file=our_file, miniters=None, smoothing=1, mininterval=0, maxinterval=maxinterval) as tm2:
                cpu_timify(tm1, timer)
                cpu_timify(tm2, timer)
                timer.sleep(mininterval)
                tm1.update(total / 2)
                tm2.update(total / 2)
                assert int(tm1.miniters) == tm2.miniters == total / 2
                timer.sleep(maxinterval * 2)
                tm1.update(total / 2)
                tm2.update(total / 2)
                res = [tm1.miniters, tm2.miniters]
                assert res == [total / 2 * mininterval / (maxinterval * 2), total / 2 * maxinterval / (maxinterval * 2)]
    timer1 = DiscreteTimer()
    timer2 = DiscreteTimer()
    total = 100
    mininterval = 0.1
    maxinterval = 10
    with closing(StringIO()) as our_file:
        t1 = tqdm(range(total), file=our_file, miniters=None, smoothing=1, mininterval=mininterval, maxinterval=maxinterval)
        t2 = tqdm(range(total), file=our_file, miniters=None, smoothing=1, mininterval=0, maxinterval=maxinterval)
        cpu_timify(t1, timer1)
        cpu_timify(t2, timer2)
        for i in t1:
            if i == total / 2 - 2:
                timer1.sleep(mininterval)
            if i == total - 1:
                timer1.sleep(maxinterval * 2)
        for i in t2:
            if i == total / 2 - 2:
                timer2.sleep(mininterval)
            if i == total - 1:
                timer2.sleep(maxinterval * 2)
        assert t1.miniters == 0.255
        assert t2.miniters == 0.5
        t1.close()
        t2.close()

def test_delay():
    if False:
        for i in range(10):
            print('nop')
    'Test delay'
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        t = tqdm(total=2, file=our_file, leave=True, delay=3)
        cpu_timify(t, timer)
        timer.sleep(2)
        t.update(1)
        assert not our_file.getvalue()
        timer.sleep(2)
        t.update(1)
        assert our_file.getvalue()
        t.close()

def test_min_iters():
    if False:
        print('Hello World!')
    'Test miniters'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(3), file=our_file, leave=True, mininterval=0, miniters=2):
            pass
        out = our_file.getvalue()
        assert '| 0/3 ' in out
        assert '| 1/3 ' not in out
        assert '| 2/3 ' in out
        assert '| 3/3 ' in out
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(3), file=our_file, leave=True, mininterval=0, miniters=1):
            pass
        out = our_file.getvalue()
        assert '| 0/3 ' in out
        assert '| 1/3 ' in out
        assert '| 2/3 ' in out
        assert '| 3/3 ' in out

def test_dynamic_min_iters():
    if False:
        print('Hello World!')
    'Test purely dynamic miniters (and manual updates and __del__)'
    with closing(StringIO()) as our_file:
        total = 10
        t = tqdm(total=total, file=our_file, miniters=None, mininterval=0, smoothing=1)
        t.update()
        t.update(3)
        t.update()
        t.update()
        t.update()
        out = our_file.getvalue()
        assert t.dynamic_miniters
        t.__del__()
    assert '  0%|          | 0/10 [00:00<' in out
    assert '40%' in out
    assert '50%' not in out
    assert '60%' not in out
    assert '70%' in out
    with closing(StringIO()) as our_file:
        total = 10
        t = tqdm(total=total, file=our_file, miniters=None, mininterval=0, smoothing=0)
        t.update()
        t.update(2)
        t.update(5)
        t.update(1)
        out = our_file.getvalue()
        assert all((i in out for i in ('0/10', '1/10', '3/10')))
        assert '2/10' not in out
        assert t.dynamic_miniters and (not t.smoothing)
        assert t.miniters == 5
        t.close()
    with closing(StringIO()) as our_file:
        t = tqdm(range(10), file=our_file, miniters=None, mininterval=None, smoothing=0.5)
        for _ in t:
            pass
        assert t.dynamic_miniters
    with closing(StringIO()) as our_file:
        t = tqdm(range(10), file=our_file, miniters=None, mininterval=None, smoothing=0)
        for _ in t:
            pass
        assert t.dynamic_miniters
    with closing(StringIO()) as our_file:
        t = tqdm(range(10), file=our_file, miniters=1, mininterval=None)
        for _ in t:
            pass
        assert not t.dynamic_miniters

def test_big_min_interval():
    if False:
        i = 10
        return i + 15
    'Test large mininterval'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(2), file=our_file, mininterval=10000000000.0):
            pass
        assert '50%' not in our_file.getvalue()
    with closing(StringIO()) as our_file:
        with tqdm(range(2), file=our_file, mininterval=10000000000.0) as t:
            t.update()
            t.update()
            assert '50%' not in our_file.getvalue()

def test_smoothed_dynamic_min_iters():
    if False:
        return 10
    'Test smoothed dynamic miniters'
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        with tqdm(total=100, file=our_file, miniters=None, mininterval=1, smoothing=0.5, maxinterval=0) as t:
            cpu_timify(t, timer)
            timer.sleep(1)
            t.update(10)
            for _ in range(2):
                timer.sleep(1)
                t.update(4)
            for _ in range(20):
                timer.sleep(1)
                t.update()
            assert t.dynamic_miniters
        out = our_file.getvalue()
    assert '  0%|          | 0/100 [00:00<' in out
    assert '20%' in out
    assert '23%' not in out
    assert '25%' in out
    assert '26%' not in out
    assert '28%' in out

def test_smoothed_dynamic_min_iters_with_min_interval():
    if False:
        for i in range(10):
            print('nop')
    'Test smoothed dynamic miniters with mininterval'
    timer = DiscreteTimer()
    total = 100
    with closing(StringIO()) as our_file:
        with tqdm(total=total, file=our_file, miniters=None, mininterval=0.001, smoothing=1, maxinterval=0) as t:
            cpu_timify(t, timer)
            t.update(10)
            timer.sleep(0.01)
            for _ in range(4):
                t.update()
                timer.sleep(0.01)
            out = our_file.getvalue()
            assert t.dynamic_miniters
    with closing(StringIO()) as our_file:
        with tqdm(range(total), file=our_file, miniters=None, mininterval=0.01, smoothing=1, maxinterval=0) as t2:
            cpu_timify(t2, timer)
            for i in t2:
                if i >= 10:
                    timer.sleep(0.1)
                if i >= 14:
                    break
            out2 = our_file.getvalue()
    assert t.dynamic_miniters
    assert '  0%|          | 0/100 [00:00<' in out
    assert '11%' in out and '11%' in out2
    assert '13%' in out and '13%' in out2
    assert '14%' in out and '14%' in out2

@mark.slow
def test_rlock_creation():
    if False:
        return 10
    'Test that importing tqdm does not create multiprocessing objects.'
    mp = importorskip('multiprocessing')
    if not hasattr(mp, 'get_context'):
        skip('missing multiprocessing.get_context')
    ctx = mp.get_context('spawn')
    with ctx.Pool(1) as pool:
        pool.apply(_rlock_creation_target)

def _rlock_creation_target():
    if False:
        print('Hello World!')
    'Check that the RLock has not been constructed.'
    import multiprocessing as mp
    patch = importorskip('unittest.mock').patch
    with patch('multiprocessing.RLock', wraps=mp.RLock) as rlock_mock:
        from tqdm import tqdm
        assert rlock_mock.call_count == 0
        with closing(StringIO()) as our_file:
            with tqdm(file=our_file) as _:
                pass
        assert rlock_mock.call_count == 1
        with closing(StringIO()) as our_file:
            with tqdm(file=our_file) as _:
                pass
        assert rlock_mock.call_count == 1

def test_disable():
    if False:
        i = 10
        return i + 15
    'Test disable'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(3), file=our_file, disable=True):
            pass
        assert our_file.getvalue() == ''
    with closing(StringIO()) as our_file:
        progressbar = tqdm(total=3, file=our_file, miniters=1, disable=True)
        progressbar.update(3)
        progressbar.close()
        assert our_file.getvalue() == ''

def test_infinite_total():
    if False:
        print('Hello World!')
    'Test treatment of infinite total'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(3), file=our_file, total=float('inf')):
            pass

def test_nototal():
    if False:
        return 10
    'Test unknown total length'
    with closing(StringIO()) as our_file:
        for _ in tqdm(iter(range(10)), file=our_file, unit_scale=10):
            pass
        assert '100it' in our_file.getvalue()
    with closing(StringIO()) as our_file:
        for _ in tqdm(iter(range(10)), file=our_file, bar_format='{l_bar}{bar}{r_bar}'):
            pass
        assert '10/?' in our_file.getvalue()

def test_unit():
    if False:
        i = 10
        return i + 15
    'Test SI unit prefix'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(3), file=our_file, miniters=1, unit='bytes'):
            pass
        assert 'bytes/s' in our_file.getvalue()

def test_ascii():
    if False:
        for i in range(10):
            print('nop')
    'Test ascii/unicode bar'
    with closing(StringIO()) as our_file:
        with tqdm(total=10, file=our_file, ascii=None) as t:
            assert t.ascii
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(3), total=15, file=our_file, miniters=1, mininterval=0, ascii=True):
            pass
        res = our_file.getvalue().strip('\r').split('\r')
    assert '7%|6' in res[1]
    assert '13%|#3' in res[2]
    assert '20%|##' in res[3]
    with closing(UnicodeIO()) as our_file:
        with tqdm(total=15, file=our_file, ascii=False, mininterval=0) as t:
            for _ in range(3):
                t.update()
        res = our_file.getvalue().strip('\r').split('\r')
    assert u'7%|▋' in res[1]
    assert u'13%|█▎' in res[2]
    assert u'20%|██' in res[3]
    for bars in [' .oO0', ' #']:
        with closing(StringIO()) as our_file:
            for _ in tqdm(range(len(bars) - 1), file=our_file, miniters=1, mininterval=0, ascii=bars, ncols=27):
                pass
            res = our_file.getvalue().strip('\r').split('\r')
        for (b, line) in zip(bars, res):
            assert '|' + b + '|' in line

def test_update():
    if False:
        i = 10
        return i + 15
    'Test manual creation and updates'
    res = None
    with closing(StringIO()) as our_file:
        with tqdm(total=2, file=our_file, miniters=1, mininterval=0) as progressbar:
            assert len(progressbar) == 2
            progressbar.update(2)
            assert '| 2/2' in our_file.getvalue()
            progressbar.desc = 'dynamically notify of 4 increments in total'
            progressbar.total = 4
            progressbar.update(-1)
            progressbar.update(2)
        res = our_file.getvalue()
    assert '| 3/4 ' in res
    assert 'dynamically notify of 4 increments in total' in res

def test_close():
    if False:
        i = 10
        return i + 15
    'Test manual creation and closure and n_instances'
    with closing(StringIO()) as our_file:
        progressbar = tqdm(total=3, file=our_file, miniters=10)
        progressbar.update(3)
        assert '| 3/3 ' not in our_file.getvalue()
        assert len(tqdm._instances) == 1
        progressbar.close()
        assert len(tqdm._instances) == 0
        assert '| 3/3 ' in our_file.getvalue()
    with closing(StringIO()) as our_file:
        progressbar = tqdm(total=3, file=our_file, miniters=10, leave=False)
        progressbar.update(3)
        progressbar.close()
        assert '| 3/3 ' not in our_file.getvalue()
    with closing(StringIO()) as our_file:
        assert len(tqdm._instances) == 0
        with tqdm(total=3, file=our_file, miniters=0, mininterval=0, leave=True) as progressbar:
            assert len(tqdm._instances) == 1
            progressbar.update(3)
            res = our_file.getvalue()
            assert '| 3/3 ' in res
            assert '\n' not in res
        assert len(tqdm._instances) == 0
        exres = res.rsplit(', ', 1)[0]
        res = our_file.getvalue()
        assert res[-1] == '\n'
        if not res.startswith(exres):
            raise AssertionError(f'\n<<< Expected:\n{exres}, ...it/s]\n>>> Got:\n{res}\n===')
    with closing(StringIO()) as our_file:
        t = tqdm(total=2, file=our_file)
        t.update()
        t.update()
    t.close()

def test_ema():
    if False:
        print('Hello World!')
    'Test exponential weighted average'
    ema = EMA(0.01)
    assert round(ema(10), 2) == 10
    assert round(ema(1), 2) == 5.48
    assert round(ema(), 2) == 5.48
    assert round(ema(1), 2) == 3.97
    assert round(ema(1), 2) == 3.22

def test_smoothing():
    if False:
        for i in range(10):
            print('nop')
    'Test exponential weighted average smoothing'
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        with tqdm(range(3), file=our_file, smoothing=None, leave=True) as t:
            cpu_timify(t, timer)
            for _ in t:
                pass
        assert '| 3/3 ' in our_file.getvalue()
    with closing(StringIO()) as our_file2:
        with closing(StringIO()) as our_file:
            t = tqdm(range(3), file=our_file2, smoothing=None, leave=True, miniters=1, mininterval=0)
            cpu_timify(t, timer)
            with tqdm(range(3), file=our_file, smoothing=None, leave=True, miniters=1, mininterval=0) as t2:
                cpu_timify(t2, timer)
                for i in t2:
                    if i == 0:
                        timer.sleep(0.01)
                    else:
                        timer.sleep(0.001)
                    t.update()
            n_old = len(tqdm._instances)
            t.close()
            assert len(tqdm._instances) == n_old - 1
            a = progressbar_rate(get_bar(our_file.getvalue(), 3))
        a2 = progressbar_rate(get_bar(our_file2.getvalue(), 3))
    with closing(StringIO()) as our_file2:
        with closing(StringIO()) as our_file:
            t = tqdm(range(3), file=our_file2, smoothing=1, leave=True, miniters=1, mininterval=0)
            cpu_timify(t, timer)
            with tqdm(range(3), file=our_file, smoothing=1, leave=True, miniters=1, mininterval=0) as t2:
                cpu_timify(t2, timer)
                for i in t2:
                    if i == 0:
                        timer.sleep(0.01)
                    else:
                        timer.sleep(0.001)
                    t.update()
            t.close()
            b = progressbar_rate(get_bar(our_file.getvalue(), 3))
        b2 = progressbar_rate(get_bar(our_file2.getvalue(), 3))
    with closing(StringIO()) as our_file2:
        with closing(StringIO()) as our_file:
            t = tqdm(range(3), file=our_file2, smoothing=0.5, leave=True, miniters=1, mininterval=0)
            cpu_timify(t, timer)
            t2 = tqdm(range(3), file=our_file, smoothing=0.5, leave=True, miniters=1, mininterval=0)
            cpu_timify(t2, timer)
            for i in t2:
                if i == 0:
                    timer.sleep(0.01)
                else:
                    timer.sleep(0.001)
                t.update()
            t2.close()
            t.close()
            c = progressbar_rate(get_bar(our_file.getvalue(), 3))
        c2 = progressbar_rate(get_bar(our_file2.getvalue(), 3))
    assert a <= c <= b
    assert a2 <= c2 <= b2

@mark.skipif(nt_and_no_colorama, reason='Windows without colorama')
def test_deprecated_nested():
    if False:
        i = 10
        return i + 15
    'Test nested progress bars'
    our_file = StringIO()
    try:
        tqdm(total=2, file=our_file, nested=True)
    except TqdmDeprecationWarning:
        if '`nested` is deprecated and automated.\nUse `position` instead for manual control.' not in our_file.getvalue():
            raise
    else:
        raise DeprecationError('Should not allow nested kwarg')

def test_bar_format():
    if False:
        return 10
    'Test custom bar formatting'
    with closing(StringIO()) as our_file:
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt}-{n}/{total}{percentage}{rate}{rate_fmt}{elapsed}{remaining}'
        for _ in trange(2, file=our_file, leave=True, bar_format=bar_format):
            pass
        out = our_file.getvalue()
    assert '\r  0%|          |0/2-0/20.0None?it/s00:00?\r' in out
    with closing(StringIO()) as our_file:
        bar_format = 'hello world'
        with tqdm(ascii=False, bar_format=bar_format, file=our_file) as t:
            assert isinstance(t.bar_format, str)

def test_custom_format():
    if False:
        return 10
    'Test adding additional derived format arguments'

    class TqdmExtraFormat(tqdm):
        """Provides a `total_time` format parameter"""

        @property
        def format_dict(self):
            if False:
                for i in range(10):
                    print('nop')
            d = super(TqdmExtraFormat, self).format_dict
            total_time = d['elapsed'] * (d['total'] or 0) / max(d['n'], 1)
            d.update(total_time=self.format_interval(total_time) + ' in total')
            return d
    with closing(StringIO()) as our_file:
        for _ in TqdmExtraFormat(range(10), file=our_file, bar_format='{total_time}: {percentage:.0f}%|{bar}{r_bar}'):
            pass
        assert '00:00 in total' in our_file.getvalue()

def test_eta(capsys):
    if False:
        return 10
    'Test eta bar_format'
    from datetime import datetime as dt
    for _ in trange(999, miniters=1, mininterval=0, leave=True, bar_format='{l_bar}{eta:%Y-%m-%d}'):
        pass
    (_, err) = capsys.readouterr()
    assert '\r100%|{eta:%Y-%m-%d}\n'.format(eta=dt.now()) in err

def test_unpause():
    if False:
        for i in range(10):
            print('nop')
    'Test unpause'
    timer = DiscreteTimer()
    with closing(StringIO()) as our_file:
        t = trange(10, file=our_file, leave=True, mininterval=0)
        cpu_timify(t, timer)
        timer.sleep(0.01)
        t.update()
        timer.sleep(0.01)
        t.update()
        timer.sleep(0.1)
        t.unpause()
        timer.sleep(0.01)
        t.update()
        timer.sleep(0.01)
        t.update()
        t.close()
        r_before = progressbar_rate(get_bar(our_file.getvalue(), 2))
        r_after = progressbar_rate(get_bar(our_file.getvalue(), 3))
    assert r_before == r_after

def test_disabled_unpause(capsys):
    if False:
        return 10
    'Test disabled unpause'
    with tqdm(total=10, disable=True) as t:
        t.update()
        t.unpause()
        t.update()
        print(t)
    (out, err) = capsys.readouterr()
    assert not err
    assert out == '  0%|          | 0/10 [00:00<?, ?it/s]\n'

def test_reset():
    if False:
        for i in range(10):
            print('nop')
    'Test resetting a bar for re-use'
    with closing(StringIO()) as our_file:
        with tqdm(total=10, file=our_file, miniters=1, mininterval=0, maxinterval=0) as t:
            t.update(9)
            t.reset()
            t.update()
            t.reset(total=12)
            t.update(10)
        assert '| 1/10' in our_file.getvalue()
        assert '| 10/12' in our_file.getvalue()

def test_disabled_reset(capsys):
    if False:
        print('Hello World!')
    'Test disabled reset'
    with tqdm(total=10, disable=True) as t:
        t.update(9)
        t.reset()
        t.update()
        t.reset(total=12)
        t.update(10)
        print(t)
    (out, err) = capsys.readouterr()
    assert not err
    assert out == '  0%|          | 0/12 [00:00<?, ?it/s]\n'

@mark.skipif(nt_and_no_colorama, reason='Windows without colorama')
def test_position():
    if False:
        i = 10
        return i + 15
    'Test positioned progress bars'
    our_file = StringIO()
    kwargs = {'file': our_file, 'miniters': 1, 'mininterval': 0, 'maxinterval': 0}
    t = tqdm(total=2, desc='pos2 bar', leave=False, position=2, **kwargs)
    t.update()
    t.close()
    out = our_file.getvalue()
    res = [m[0] for m in RE_pos.findall(out)]
    exres = ['\n\n\rpos2 bar:   0%', '\n\n\rpos2 bar:  50%', '\n\n\r      ']
    pos_line_diff(res, exres)
    our_file = StringIO()
    kwargs['file'] = our_file
    for _ in trange(2, desc='pos0 bar', position=0, **kwargs):
        for _ in trange(2, desc='pos1 bar', position=1, **kwargs):
            for _ in trange(2, desc='pos2 bar', position=2, **kwargs):
                pass
    out = our_file.getvalue()
    res = [m[0] for m in RE_pos.findall(out)]
    exres = ['\rpos0 bar:   0%', '\n\rpos1 bar:   0%', '\n\n\rpos2 bar:   0%', '\n\n\rpos2 bar:  50%', '\n\n\rpos2 bar: 100%', '\rpos2 bar: 100%', '\n\n\rpos1 bar:  50%', '\n\n\rpos2 bar:   0%', '\n\n\rpos2 bar:  50%', '\n\n\rpos2 bar: 100%', '\rpos2 bar: 100%', '\n\n\rpos1 bar: 100%', '\rpos1 bar: 100%', '\n\rpos0 bar:  50%', '\n\rpos1 bar:   0%', '\n\n\rpos2 bar:   0%', '\n\n\rpos2 bar:  50%', '\n\n\rpos2 bar: 100%', '\rpos2 bar: 100%', '\n\n\rpos1 bar:  50%', '\n\n\rpos2 bar:   0%', '\n\n\rpos2 bar:  50%', '\n\n\rpos2 bar: 100%', '\rpos2 bar: 100%', '\n\n\rpos1 bar: 100%', '\rpos1 bar: 100%', '\n\rpos0 bar: 100%', '\rpos0 bar: 100%', '\n']
    pos_line_diff(res, exres)
    our_file = StringIO()
    kwargs['file'] = our_file
    kwargs['total'] = 2
    t1 = tqdm(desc='pos0 bar', position=0, **kwargs)
    t2 = tqdm(desc='pos1 bar', position=1, **kwargs)
    t3 = tqdm(desc='pos2 bar', position=2, **kwargs)
    for _ in range(2):
        t1.update()
        t3.update()
        t2.update()
    out = our_file.getvalue()
    res = [m[0] for m in RE_pos.findall(out)]
    exres = ['\rpos0 bar:   0%', '\n\rpos1 bar:   0%', '\n\n\rpos2 bar:   0%', '\rpos0 bar:  50%', '\n\n\rpos2 bar:  50%', '\n\rpos1 bar:  50%', '\rpos0 bar: 100%', '\n\n\rpos2 bar: 100%', '\n\rpos1 bar: 100%']
    pos_line_diff(res, exres)
    t1.close()
    t2.close()
    t3.close()
    with closing(StringIO()) as our_file:
        t1 = tqdm(total=10, file=our_file, desc='1.pos0 bar', mininterval=0)
        t2 = tqdm(total=10, file=our_file, desc='2.pos1 bar', mininterval=0)
        t3 = tqdm(total=10, file=our_file, desc='3.pos2 bar', mininterval=0)
        res = [m[0] for m in RE_pos.findall(our_file.getvalue())]
        exres = ['\r1.pos0 bar:   0%', '\n\r2.pos1 bar:   0%', '\n\n\r3.pos2 bar:   0%']
        pos_line_diff(res, exres)
        t2.close()
        t4 = tqdm(total=10, file=our_file, desc='4.pos2 bar', mininterval=0)
        t1.update(1)
        t3.update(1)
        t4.update(1)
        res = [m[0] for m in RE_pos.findall(our_file.getvalue())]
        exres = ['\r1.pos0 bar:   0%', '\n\r2.pos1 bar:   0%', '\n\n\r3.pos2 bar:   0%', '\r2.pos1 bar:   0%', '\n\n\r4.pos2 bar:   0%', '\r1.pos0 bar:  10%', '\n\n\r3.pos2 bar:  10%', '\n\r4.pos2 bar:  10%']
        pos_line_diff(res, exres)
        t4.close()
        t3.close()
        t1.close()

def test_set_description():
    if False:
        i = 10
        return i + 15
    'Test set description'
    with closing(StringIO()) as our_file:
        with tqdm(desc='Hello', file=our_file) as t:
            assert t.desc == 'Hello'
            t.set_description_str('World')
            assert t.desc == 'World'
            t.set_description()
            assert t.desc == ''
            t.set_description('Bye')
            assert t.desc == 'Bye: '
        assert 'World' in our_file.getvalue()
    with closing(StringIO()) as our_file:
        with tqdm(desc='Hello', file=our_file) as t:
            assert t.desc == 'Hello'
            t.set_description_str('World', False)
            assert t.desc == 'World'
            t.set_description(None, False)
            assert t.desc == ''
        assert 'World' not in our_file.getvalue()
    with closing(StringIO()) as our_file:
        with tqdm(total=10, file=our_file) as t:
            t.set_description(u'áéíóú')

def test_deprecated_gui():
    if False:
        i = 10
        return i + 15
    'Test internal GUI properties'
    with closing(StringIO()) as our_file:
        t = tqdm(total=2, gui=True, file=our_file, miniters=1, mininterval=0)
        assert not hasattr(t, 'sp')
        try:
            t.update(1)
        except TqdmDeprecationWarning as e:
            if 'Please use `tqdm.gui.tqdm(...)` instead of `tqdm(..., gui=True)`' not in our_file.getvalue():
                raise e
        else:
            raise DeprecationError('Should not allow manual gui=True without overriding __iter__() and update()')
        finally:
            t._instances.clear()
        t = tqdm(range(3), gui=True, file=our_file, miniters=1, mininterval=0)
        try:
            for _ in t:
                pass
        except TqdmDeprecationWarning as e:
            if 'Please use `tqdm.gui.tqdm(...)` instead of `tqdm(..., gui=True)`' not in our_file.getvalue():
                raise e
        else:
            raise DeprecationError('Should not allow manual gui=True without overriding __iter__() and update()')
        finally:
            t._instances.clear()
        with tqdm(total=1, gui=False, file=our_file) as t:
            assert hasattr(t, 'sp')

def test_cmp():
    if False:
        i = 10
        return i + 15
    'Test comparison functions'
    with closing(StringIO()) as our_file:
        t0 = tqdm(total=10, file=our_file)
        t1 = tqdm(total=10, file=our_file)
        t2 = tqdm(total=10, file=our_file)
        assert t0 < t1
        assert t2 >= t0
        assert t0 <= t2
        t3 = tqdm(total=10, file=our_file)
        t4 = tqdm(total=10, file=our_file)
        t5 = tqdm(total=10, file=our_file)
        t5.close()
        t6 = tqdm(total=10, file=our_file)
        assert t3 != t4
        assert t3 > t2
        assert t5 == t6
        t6.close()
        t4.close()
        t3.close()
        t2.close()
        t1.close()
        t0.close()

def test_repr():
    if False:
        i = 10
        return i + 15
    'Test representation'
    with closing(StringIO()) as our_file:
        with tqdm(total=10, ascii=True, file=our_file) as t:
            assert str(t) == '  0%|          | 0/10 [00:00<?, ?it/s]'

def test_clear():
    if False:
        while True:
            i = 10
    'Test clearing bar display'
    with closing(StringIO()) as our_file:
        t1 = tqdm(total=10, file=our_file, desc='pos0 bar', bar_format='{l_bar}')
        t2 = trange(10, file=our_file, desc='pos1 bar', bar_format='{l_bar}')
        before = squash_ctrlchars(our_file.getvalue())
        t2.clear()
        t1.clear()
        after = squash_ctrlchars(our_file.getvalue())
        t1.close()
        t2.close()
        assert before == ['pos0 bar:   0%|', 'pos1 bar:   0%|']
        assert after == ['', '']

def test_clear_disabled():
    if False:
        return 10
    'Test disabled clear'
    with closing(StringIO()) as our_file:
        with tqdm(total=10, file=our_file, desc='pos0 bar', disable=True, bar_format='{l_bar}') as t:
            t.clear()
        assert our_file.getvalue() == ''

def test_refresh():
    if False:
        for i in range(10):
            print('nop')
    'Test refresh bar display'
    with closing(StringIO()) as our_file:
        t1 = tqdm(total=10, file=our_file, desc='pos0 bar', bar_format='{l_bar}', mininterval=999, miniters=999)
        t2 = tqdm(total=10, file=our_file, desc='pos1 bar', bar_format='{l_bar}', mininterval=999, miniters=999)
        t1.update()
        t2.update()
        before = squash_ctrlchars(our_file.getvalue())
        t1.refresh()
        t2.refresh()
        after = squash_ctrlchars(our_file.getvalue())
        t1.close()
        t2.close()
        assert before == [u'pos0 bar:   0%|', u'pos1 bar:   0%|']
        assert after == [u'pos0 bar:  10%|', u'pos1 bar:  10%|']

def test_disabled_repr(capsys):
    if False:
        return 10
    'Test disabled repr'
    with tqdm(total=10, disable=True) as t:
        str(t)
        t.update()
        print(t)
    (out, err) = capsys.readouterr()
    assert not err
    assert out == '  0%|          | 0/10 [00:00<?, ?it/s]\n'

def test_disabled_refresh():
    if False:
        i = 10
        return i + 15
    'Test disabled refresh'
    with closing(StringIO()) as our_file:
        with tqdm(total=10, file=our_file, desc='pos0 bar', disable=True, bar_format='{l_bar}', mininterval=999, miniters=999) as t:
            t.update()
            t.refresh()
        assert our_file.getvalue() == ''

def test_write():
    if False:
        while True:
            i = 10
    'Test write messages'
    s = 'Hello world'
    with closing(StringIO()) as our_file:
        t1 = tqdm(total=10, file=our_file, desc='pos0 bar', bar_format='{l_bar}', mininterval=0, miniters=1)
        t2 = trange(10, file=our_file, desc='pos1 bar', bar_format='{l_bar}', mininterval=0, miniters=1)
        t3 = tqdm(total=10, file=our_file, desc='pos2 bar', bar_format='{l_bar}', mininterval=0, miniters=1)
        t1.update()
        t2.update()
        t3.update()
        before = our_file.getvalue()
        t1.write(s, file=our_file)
        tqdm.write(s, file=our_file)
        after = our_file.getvalue()
        t1.close()
        t2.close()
        t3.close()
        before_squashed = squash_ctrlchars(before)
        after_squashed = squash_ctrlchars(after)
        assert after_squashed == [s, s] + before_squashed
    with closing(StringIO()) as our_file_bar:
        with closing(StringIO()) as our_file_write:
            t1 = tqdm(total=10, file=our_file_bar, desc='pos0 bar', bar_format='{l_bar}', mininterval=0, miniters=1)
            t1.update()
            before_bar = our_file_bar.getvalue()
            tqdm.write(s, file=our_file_write)
            after_bar = our_file_bar.getvalue()
            t1.close()
            assert before_bar == after_bar
    stde = sys.stderr
    stdo = sys.stdout
    with closing(StringIO()) as our_stderr:
        with closing(StringIO()) as our_stdout:
            sys.stderr = our_stderr
            sys.stdout = our_stdout
            t1 = tqdm(total=10, file=sys.stderr, desc='pos0 bar', bar_format='{l_bar}', mininterval=0, miniters=1)
            t1.update()
            before_err = sys.stderr.getvalue()
            before_out = sys.stdout.getvalue()
            tqdm.write(s, file=sys.stdout)
            after_err = sys.stderr.getvalue()
            after_out = sys.stdout.getvalue()
            t1.close()
            assert before_err == '\rpos0 bar:   0%|\rpos0 bar:  10%|'
            assert before_out == ''
            after_err_res = [m[0] for m in RE_pos.findall(after_err)]
            exres = ['\rpos0 bar:   0%|', '\rpos0 bar:  10%|', '\r               ', '\r\rpos0 bar:  10%|']
            pos_line_diff(after_err_res, exres)
            assert after_out == s + '\n'
    sys.stderr = stde
    sys.stdout = stdo

def test_len():
    if False:
        for i in range(10):
            print('nop')
    'Test advance len (numpy array shape)'
    np = importorskip('numpy')
    with closing(StringIO()) as f:
        with tqdm(np.zeros((3, 4)), file=f) as t:
            assert len(t) == 3

def test_autodisable_disable():
    if False:
        print('Hello World!')
    'Test autodisable will disable on non-TTY'
    with closing(StringIO()) as our_file:
        with tqdm(total=10, disable=None, file=our_file) as t:
            t.update(3)
        assert our_file.getvalue() == ''

def test_autodisable_enable():
    if False:
        return 10
    'Test autodisable will not disable on TTY'
    with closing(StringIO()) as our_file:
        our_file.isatty = lambda : True
        with tqdm(total=10, disable=None, file=our_file) as t:
            t.update()
        assert our_file.getvalue() != ''

def test_deprecation_exception():
    if False:
        i = 10
        return i + 15

    def test_TqdmDeprecationWarning():
        if False:
            i = 10
            return i + 15
        with closing(StringIO()) as our_file:
            raise TqdmDeprecationWarning('Test!', fp_write=getattr(our_file, 'write', sys.stderr.write))

    def test_TqdmDeprecationWarning_nofpwrite():
        if False:
            while True:
                i = 10
        raise TqdmDeprecationWarning('Test!', fp_write=None)
    raises(TqdmDeprecationWarning, test_TqdmDeprecationWarning)
    raises(Exception, test_TqdmDeprecationWarning_nofpwrite)

def test_postfix():
    if False:
        for i in range(10):
            print('nop')
    'Test postfix'
    postfix = {'float': 0.321034, 'gen': 543, 'str': 'h', 'lst': [2]}
    postfix_order = (('w', 'w'), ('a', 0))
    expected = ['float=0.321', 'gen=543', 'lst=[2]', 'str=h']
    expected_order = ['w=w', 'a=0', 'float=0.321', 'gen=543', 'lst=[2]', 'str=h']
    with closing(StringIO()) as our_file:
        with tqdm(total=10, file=our_file, desc='pos0 bar', bar_format='{r_bar}', postfix=postfix) as t1:
            t1.refresh()
            out = our_file.getvalue()
    with closing(StringIO()) as our_file:
        with trange(10, file=our_file, desc='pos1 bar', bar_format='{r_bar}', postfix=None) as t2:
            t2.set_postfix(**postfix)
            t2.refresh()
            out2 = our_file.getvalue()
    for res in expected:
        assert res in out
        assert res in out2
    with closing(StringIO()) as our_file:
        with trange(10, file=our_file, desc='pos2 bar', bar_format='{r_bar}', postfix=None) as t3:
            t3.set_postfix(postfix_order, False, **postfix)
            t3.refresh()
            out3 = our_file.getvalue()
    out3 = out3[1:-1].split(', ')[3:]
    assert out3 == expected_order
    with closing(StringIO()) as our_file:
        with trange(10, file=our_file, desc='pos2 bar', bar_format='{r_bar}', postfix=None) as t4:
            t4.set_postfix(postfix_order, True, **postfix)
            t4.refresh()
            out4 = our_file.getvalue()
    assert out4.count('\r') > out3.count('\r')
    assert out4.count(', '.join(expected_order)) == 2
    with closing(StringIO()) as our_file:
        with trange(10, file=our_file, desc='pos2 bar', bar_format='{r_bar}', postfix=None) as t5:
            t5.set_postfix_str('Hello', False)
            t5.set_postfix_str('World')
            out5 = our_file.getvalue()
    assert 'Hello' not in out5
    out5 = out5[1:-1].split(', ')[3:]
    assert out5 == ['World']

def test_postfix_direct():
    if False:
        print('Hello World!')
    'Test directly assigning non-str objects to postfix'
    with closing(StringIO()) as our_file:
        with tqdm(total=10, file=our_file, miniters=1, mininterval=0, bar_format='{postfix[0][name]} {postfix[1]:>5.2f}', postfix=[{'name': 'foo'}, 42]) as t:
            for i in range(10):
                if i % 2:
                    t.postfix[0]['name'] = 'abcdefghij'[i]
                else:
                    t.postfix[1] = i
                t.update()
        res = our_file.getvalue()
        assert 'f  6.00' in res
        assert 'h  6.00' in res
        assert 'h  8.00' in res
        assert 'j  8.00' in res

@contextmanager
def std_out_err_redirect_tqdm(tqdm_file=sys.stderr):
    if False:
        for i in range(10):
            print('nop')
    orig_out_err = (sys.stdout, sys.stderr)
    try:
        sys.stdout = sys.stderr = DummyTqdmFile(tqdm_file)
        yield orig_out_err[0]
    except Exception as exc:
        raise exc
    finally:
        (sys.stdout, sys.stderr) = orig_out_err

def test_file_redirection():
    if False:
        i = 10
        return i + 15
    'Test redirection of output'
    with closing(StringIO()) as our_file:
        with std_out_err_redirect_tqdm(tqdm_file=our_file):
            with tqdm(total=3) as pbar:
                print('Such fun')
                pbar.update(1)
                print('Such', 'fun')
                pbar.update(1)
                print('Such ', end='')
                print('fun')
                pbar.update(1)
        res = our_file.getvalue()
        assert res.count('Such fun\n') == 3
        assert '0/3' in res
        assert '3/3' in res

def test_external_write():
    if False:
        i = 10
        return i + 15
    'Test external write mode'
    with closing(StringIO()) as our_file:
        for _ in trange(3, file=our_file):
            del tqdm._lock
            with tqdm.external_write_mode(file=our_file):
                our_file.write('Such fun\n')
        res = our_file.getvalue()
        assert res.count('Such fun\n') == 3
        assert '0/3' in res
        assert '3/3' in res

def test_unit_scale():
    if False:
        i = 10
        return i + 15
    'Test numeric `unit_scale`'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(9), unit_scale=9, file=our_file, miniters=1, mininterval=0):
            pass
        out = our_file.getvalue()
        assert '81/81' in out

def patch_lock(thread=True):
    if False:
        return 10
    "decorator replacing tqdm's lock with vanilla threading/multiprocessing"
    try:
        if thread:
            from threading import RLock
        else:
            from multiprocessing import RLock
        lock = RLock()
    except (ImportError, OSError) as err:
        skip(str(err))

    def outer(func):
        if False:
            print('Hello World!')
        'actual decorator'

        @wraps(func)
        def inner(*args, **kwargs):
            if False:
                print('Hello World!')
            'set & reset lock even if exceptions occur'
            default_lock = tqdm.get_lock()
            try:
                tqdm.set_lock(lock)
                return func(*args, **kwargs)
            finally:
                tqdm.set_lock(default_lock)
        return inner
    return outer

@patch_lock(thread=False)
def test_threading():
    if False:
        print('Hello World!')
    'Test multiprocess/thread-realted features'
    pass

def test_bool():
    if False:
        return 10
    'Test boolean cast'

    def internal(our_file, disable):
        if False:
            while True:
                i = 10
        kwargs = {'file': our_file, 'disable': disable}
        with trange(10, **kwargs) as t:
            assert t
        with trange(0, **kwargs) as t:
            assert not t
        with tqdm(total=10, **kwargs) as t:
            assert bool(t)
        with tqdm(total=0, **kwargs) as t:
            assert not bool(t)
        with tqdm([], **kwargs) as t:
            assert not t
        with tqdm([0], **kwargs) as t:
            assert t
        with tqdm(iter([]), **kwargs) as t:
            assert t
        with tqdm(iter([1, 2, 3]), **kwargs) as t:
            assert t
        with tqdm(**kwargs) as t:
            try:
                print(bool(t))
            except TypeError:
                pass
            else:
                raise TypeError('Expected bool(tqdm()) to fail')
    with closing(StringIO()) as our_file:
        internal(our_file, False)
        internal(our_file, True)

def backendCheck(module):
    if False:
        for i in range(10):
            print('nop')
    'Test tqdm-like module fallback'
    tn = module.tqdm
    tr = module.trange
    with closing(StringIO()) as our_file:
        with tn(total=10, file=our_file) as t:
            assert len(t) == 10
        with tr(1337) as t:
            assert len(t) == 1337

def test_auto():
    if False:
        print('Hello World!')
    'Test auto fallback'
    from tqdm import auto, autonotebook
    backendCheck(autonotebook)
    backendCheck(auto)

def test_wrapattr():
    if False:
        print('Hello World!')
    'Test wrapping file-like objects'
    data = 'a twenty-char string'
    with closing(StringIO()) as our_file:
        with closing(StringIO()) as writer:
            with tqdm.wrapattr(writer, 'write', file=our_file, bytes=True) as wrap:
                wrap.write(data)
            res = writer.getvalue()
            assert data == res
        res = our_file.getvalue()
        assert '%.1fB [' % len(data) in res
    with closing(StringIO()) as our_file:
        with closing(StringIO()) as writer:
            with tqdm.wrapattr(writer, 'write', file=our_file, bytes=False) as wrap:
                wrap.write(data)
        res = our_file.getvalue()
        assert '%dit [' % len(data) in res

def test_float_progress():
    if False:
        while True:
            i = 10
    'Test float totals'
    with closing(StringIO()) as our_file:
        with trange(10, total=9.6, file=our_file) as t:
            with catch_warnings(record=True) as w:
                simplefilter('always', category=TqdmWarning)
                for i in t:
                    if i < 9:
                        assert not w
                assert w
                assert 'clamping frac' in str(w[-1].message)

def test_screen_shape():
    if False:
        i = 10
        return i + 15
    'Test screen shape'
    with closing(StringIO()) as our_file:
        with trange(10, file=our_file, ncols=50) as t:
            list(t)
        res = our_file.getvalue()
        assert all((len(i) == 50 for i in get_bar(res)))
    with closing(StringIO()) as our_file:
        kwargs = {'file': our_file, 'ncols': 50, 'nrows': 2, 'miniters': 0, 'mininterval': 0, 'leave': False}
        with trange(10, desc='one', **kwargs) as t1:
            with trange(10, desc='two', **kwargs) as t2:
                with trange(10, desc='three', **kwargs) as t3:
                    list(t3)
                list(t2)
            list(t1)
        res = our_file.getvalue()
        assert 'one' in res
        assert 'two' not in res
        assert 'three' not in res
        assert '\n\n' not in res
        assert 'more hidden' in res
        assert all((len(i) == 50 for i in get_bar(res) if i.strip() and 'more hidden' not in i))
    with closing(StringIO()) as our_file:
        kwargs = {'file': our_file, 'ncols': 50, 'nrows': 2, 'miniters': 0, 'mininterval': 0}
        with trange(10, desc='one', **kwargs) as t1:
            with trange(10, desc='two', **kwargs) as t2:
                assert 'two' not in our_file.getvalue()
                with trange(10, desc='three', **kwargs) as t3:
                    assert 'three' not in our_file.getvalue()
                    list(t3)
                list(t2)
            list(t1)
        res = our_file.getvalue()
        assert 'one' in res
        assert 'two' in res
        assert 'three' in res
        assert '\n\n' not in res
        assert 'more hidden' in res
        assert all((len(i) == 50 for i in get_bar(res) if i.strip() and 'more hidden' not in i))
    with closing(StringIO()) as our_file:
        kwargs = {'file': our_file, 'ncols': 50, 'nrows': 2, 'miniters': 0, 'mininterval': 0, 'leave': False}
        t1 = tqdm(total=10, desc='one', **kwargs)
        with tqdm(total=10, desc='two', **kwargs) as t2:
            t1.update()
            t2.update()
            t1.close()
            res = our_file.getvalue()
            assert 'one' in res
            assert 'two' not in res
            assert 'more hidden' in res
            t2.update()
        res = our_file.getvalue()
        assert 'two' in res

def test_initial():
    if False:
        print('Hello World!')
    'Test `initial`'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(9), initial=10, total=19, file=our_file, miniters=1, mininterval=0):
            pass
        out = our_file.getvalue()
        assert '10/19' in out
        assert '19/19' in out

def test_colour():
    if False:
        while True:
            i = 10
    'Test `colour`'
    with closing(StringIO()) as our_file:
        for _ in tqdm(range(9), file=our_file, colour='#beefed'):
            pass
        out = our_file.getvalue()
        assert '\x1b[38;2;%d;%d;%dm' % (190, 239, 237) in out
        with catch_warnings(record=True) as w:
            simplefilter('always', category=TqdmWarning)
            with tqdm(total=1, file=our_file, colour='charm') as t:
                assert w
                t.update()
            assert 'Unknown colour' in str(w[-1].message)
    with closing(StringIO()) as our_file2:
        for _ in tqdm(range(9), file=our_file2, colour='blue'):
            pass
        out = our_file2.getvalue()
        assert '\x1b[34m' in out

def test_closed():
    if False:
        for i in range(10):
            print('nop')
    'Test writing to closed file'
    with closing(StringIO()) as our_file:
        for i in trange(9, file=our_file, miniters=1, mininterval=0):
            if i == 5:
                our_file.close()

def test_reversed(capsys):
    if False:
        while True:
            i = 10
    'Test reversed()'
    for _ in reversed(tqdm(range(9))):
        pass
    (out, err) = capsys.readouterr()
    assert not out
    assert '  0%' in err
    assert '100%' in err

def test_contains(capsys):
    if False:
        return 10
    "Test __contains__ doesn't iterate"
    with tqdm(list(range(9))) as t:
        assert 9 not in t
        assert all((i in t for i in range(9)))
    (out, err) = capsys.readouterr()
    assert not out
    assert '  0%' in err
    assert '100%' not in err