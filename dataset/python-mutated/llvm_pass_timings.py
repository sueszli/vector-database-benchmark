import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm

class RecordLLVMPassTimings:
    """A helper context manager to track LLVM pass timings.
    """
    __slots__ = ['_data']

    def __enter__(self):
        if False:
            return 10
        'Enables the pass timing in LLVM.\n        '
        llvm.set_time_passes(True)
        return self

    def __exit__(self, exc_val, exc_type, exc_tb):
        if False:
            return 10
        'Reset timings and save report internally.\n        '
        self._data = llvm.report_and_reset_timings()
        llvm.set_time_passes(False)
        return

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve timing data for processing.\n\n        Returns\n        -------\n        timings: ProcessedPassTimings\n        '
        return ProcessedPassTimings(self._data)
PassTimingRecord = namedtuple('PassTimingRecord', ['user_time', 'user_percent', 'system_time', 'system_percent', 'user_system_time', 'user_system_percent', 'wall_time', 'wall_percent', 'pass_name', 'instruction'])

def _adjust_timings(records):
    if False:
        print('Hello World!')
    'Adjust timing records because of truncated information.\n\n    Details: The percent information can be used to improve the timing\n    information.\n\n    Returns\n    -------\n    res: List[PassTimingRecord]\n    '
    total_rec = records[-1]
    assert total_rec.pass_name == 'Total'

    def make_adjuster(attr):
        if False:
            print('Hello World!')
        time_attr = f'{attr}_time'
        percent_attr = f'{attr}_percent'
        time_getter = operator.attrgetter(time_attr)

        def adjust(d):
            if False:
                i = 10
                return i + 15
            'Compute percent x total_time = adjusted'
            total = time_getter(total_rec)
            adjusted = total * d[percent_attr] * 0.01
            d[time_attr] = adjusted
            return d
        return adjust
    adj_fns = [make_adjuster(x) for x in ['user', 'system', 'user_system', 'wall']]
    dicts = map(lambda x: x._asdict(), records)

    def chained(d):
        if False:
            while True:
                i = 10
        for fn in adj_fns:
            d = fn(d)
        return PassTimingRecord(**d)
    return list(map(chained, dicts))

class ProcessedPassTimings:
    """A class for processing raw timing report from LLVM.

    The processing is done lazily so we don't waste time processing unused
    timing information.
    """

    def __init__(self, raw_data):
        if False:
            for i in range(10):
                print('nop')
        self._raw_data = raw_data

    def __bool__(self):
        if False:
            while True:
                i = 10
        return bool(self._raw_data)

    def get_raw_data(self):
        if False:
            while True:
                i = 10
        'Returns the raw string data.\n\n        Returns\n        -------\n        res: str\n        '
        return self._raw_data

    def get_total_time(self):
        if False:
            i = 10
            return i + 15
        'Compute the total time spend in all passes.\n\n        Returns\n        -------\n        res: float\n        '
        return self.list_records()[-1].wall_time

    def list_records(self):
        if False:
            print('Hello World!')
        'Get the processed data for the timing report.\n\n        Returns\n        -------\n        res: List[PassTimingRecord]\n        '
        return self._processed

    def list_top(self, n):
        if False:
            for i in range(10):
                print('nop')
        'Returns the top(n) most time-consuming (by wall-time) passes.\n\n        Parameters\n        ----------\n        n: int\n            This limits the maximum number of items to show.\n            This function will show the ``n`` most time-consuming passes.\n\n        Returns\n        -------\n        res: List[PassTimingRecord]\n            Returns the top(n) most time-consuming passes in descending order.\n        '
        records = self.list_records()
        key = operator.attrgetter('wall_time')
        return heapq.nlargest(n, records[:-1], key)

    def summary(self, topn=5, indent=0):
        if False:
            print('Hello World!')
        'Return a string summarizing the timing information.\n\n        Parameters\n        ----------\n        topn: int; optional\n            This limits the maximum number of items to show.\n            This function will show the ``topn`` most time-consuming passes.\n        indent: int; optional\n            Set the indentation level. Defaults to 0 for no indentation.\n\n        Returns\n        -------\n        res: str\n        '
        buf = []
        prefix = ' ' * indent

        def ap(arg):
            if False:
                return 10
            buf.append(f'{prefix}{arg}')
        ap(f'Total {self.get_total_time():.4f}s')
        ap('Top timings:')
        for p in self.list_top(topn):
            ap(f'  {p.wall_time:.4f}s ({p.wall_percent:5}%) {p.pass_name}')
        return '\n'.join(buf)

    @cached_property
    def _processed(self):
        if False:
            print('Hello World!')
        'A cached property for lazily processing the data and returning it.\n\n        See ``_process()`` for details.\n        '
        return self._process()

    def _process(self):
        if False:
            print('Hello World!')
        'Parses the raw string data from LLVM timing report and attempts\n        to improve the data by recomputing the times\n        (See `_adjust_timings()``).\n        '

        def parse(raw_data):
            if False:
                for i in range(10):
                    print('nop')
            'A generator that parses the raw_data line-by-line to extract\n            timing information for each pass.\n            '
            lines = raw_data.splitlines()
            colheader = '[a-zA-Z+ ]+'
            multicolheaders = f'(?:\\s*-+{colheader}-+)+'
            line_iter = iter(lines)
            header_map = {'User Time': 'user', 'System Time': 'system', 'User+System': 'user_system', 'Wall Time': 'wall', 'Instr': 'instruction', 'Name': 'pass_name'}
            for ln in line_iter:
                m = re.match(multicolheaders, ln)
                if m:
                    raw_headers = re.findall('[a-zA-Z][a-zA-Z+ ]+', ln)
                    headers = [header_map[k.strip()] for k in raw_headers]
                    break
            assert headers[-1] == 'pass_name'
            attrs = []
            n = '\\s*((?:[0-9]+\\.)?[0-9]+)'
            pat = ''
            for k in headers[:-1]:
                if k == 'instruction':
                    pat += n
                else:
                    attrs.append(f'{k}_time')
                    attrs.append(f'{k}_percent')
                    pat += f'\\s+(?:{n}\\s*\\({n}%\\)|-+)'
            missing = {}
            for k in PassTimingRecord._fields:
                if k not in attrs and k != 'pass_name':
                    missing[k] = 0.0
            pat += '\\s*(.*)'
            for ln in line_iter:
                m = re.match(pat, ln)
                if m is not None:
                    raw_data = list(m.groups())
                    data = {k: float(v) if v is not None else 0.0 for (k, v) in zip(attrs, raw_data)}
                    data.update(missing)
                    pass_name = raw_data[-1]
                    rec = PassTimingRecord(pass_name=pass_name, **data)
                    yield rec
                    if rec.pass_name == 'Total':
                        break
            remaining = '\n'.join(line_iter)
            if remaining:
                raise ValueError(f'unexpected text after parser finished:\n{remaining}')
        records = list(parse(self._raw_data))
        return _adjust_timings(records)
NamedTimings = namedtuple('NamedTimings', ['name', 'timings'])

class PassTimingsCollection(Sequence):
    """A collection of pass timings.

    This class implements the ``Sequence`` protocol for accessing the
    individual timing records.
    """

    def __init__(self, name):
        if False:
            print('Hello World!')
        self._name = name
        self._records = []

    @contextmanager
    def record(self, name):
        if False:
            i = 10
            return i + 15
        'Record new timings and append to this collection.\n\n        Note: this is mainly for internal use inside the compiler pipeline.\n\n        See also ``RecordLLVMPassTimings``\n\n        Parameters\n        ----------\n        name: str\n            Name for the records.\n        '
        if config.LLVM_PASS_TIMINGS:
            with RecordLLVMPassTimings() as timings:
                yield
            rec = timings.get()
            if rec:
                self._append(name, rec)
        else:
            yield

    def _append(self, name, timings):
        if False:
            return 10
        'Append timing records\n\n        Parameters\n        ----------\n        name: str\n            Name for the records.\n        timings: ProcessedPassTimings\n            the timing records.\n        '
        self._records.append(NamedTimings(name, timings))

    def get_total_time(self):
        if False:
            while True:
                i = 10
        'Computes the sum of the total time across all contained timings.\n\n        Returns\n        -------\n        res: float or None\n            Returns the total number of seconds or None if no timings were\n            recorded\n        '
        if self._records:
            return sum((r.timings.get_total_time() for r in self._records))
        else:
            return None

    def list_longest_first(self):
        if False:
            print('Hello World!')
        'Returns the timings in descending order of total time duration.\n\n        Returns\n        -------\n        res: List[ProcessedPassTimings]\n        '
        return sorted(self._records, key=lambda x: x.timings.get_total_time(), reverse=True)

    @property
    def is_empty(self):
        if False:
            return 10
        '\n        '
        return not self._records

    def summary(self, topn=5):
        if False:
            while True:
                i = 10
        'Return a string representing the summary of the timings.\n\n        Parameters\n        ----------\n        topn: int; optional, default=5.\n            This limits the maximum number of items to show.\n            This function will show the ``topn`` most time-consuming passes.\n\n        Returns\n        -------\n        res: str\n\n        See also ``ProcessedPassTimings.summary()``\n        '
        if self.is_empty:
            return 'No pass timings were recorded'
        else:
            buf = []
            ap = buf.append
            ap(f'Printing pass timings for {self._name}')
            overall_time = self.get_total_time()
            ap(f'Total time: {overall_time:.4f}')
            for (i, r) in enumerate(self._records):
                ap(f'== #{i} {r.name}')
                percent = r.timings.get_total_time() / overall_time * 100
                ap(f' Percent: {percent:.1f}%')
                ap(r.timings.summary(topn=topn, indent=1))
            return '\n'.join(buf)

    def __getitem__(self, i):
        if False:
            print('Hello World!')
        'Get the i-th timing record.\n\n        Returns\n        -------\n        res: (name, timings)\n            A named tuple with two fields:\n\n            - name: str\n            - timings: ProcessedPassTimings\n        '
        return self._records[i]

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Length of this collection.\n        '
        return len(self._records)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.summary()