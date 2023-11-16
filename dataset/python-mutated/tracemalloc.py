from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces

def _format_size(size, sign):
    if False:
        for i in range(10):
            print('nop')
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(size) < 100 and unit != 'B':
            if sign:
                return '%+.1f %s' % (size, unit)
            else:
                return '%.1f %s' % (size, unit)
        if abs(size) < 10 * 1024 or unit == 'TiB':
            if sign:
                return '%+.0f %s' % (size, unit)
            else:
                return '%.0f %s' % (size, unit)
        size /= 1024

class Statistic:
    """
    Statistic difference on memory allocations between two Snapshot instance.
    """
    __slots__ = ('traceback', 'size', 'count')

    def __init__(self, traceback, size, count):
        if False:
            return 10
        self.traceback = traceback
        self.size = size
        self.count = count

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((self.traceback, self.size, self.count))

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Statistic):
            return NotImplemented
        return self.traceback == other.traceback and self.size == other.size and (self.count == other.count)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        text = '%s: size=%s, count=%i' % (self.traceback, _format_size(self.size, False), self.count)
        if self.count:
            average = self.size / self.count
            text += ', average=%s' % _format_size(average, False)
        return text

    def __repr__(self):
        if False:
            return 10
        return '<Statistic traceback=%r size=%i count=%i>' % (self.traceback, self.size, self.count)

    def _sort_key(self):
        if False:
            i = 10
            return i + 15
        return (self.size, self.count, self.traceback)

class StatisticDiff:
    """
    Statistic difference on memory allocations between an old and a new
    Snapshot instance.
    """
    __slots__ = ('traceback', 'size', 'size_diff', 'count', 'count_diff')

    def __init__(self, traceback, size, size_diff, count, count_diff):
        if False:
            while True:
                i = 10
        self.traceback = traceback
        self.size = size
        self.size_diff = size_diff
        self.count = count
        self.count_diff = count_diff

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.traceback, self.size, self.size_diff, self.count, self.count_diff))

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, StatisticDiff):
            return NotImplemented
        return self.traceback == other.traceback and self.size == other.size and (self.size_diff == other.size_diff) and (self.count == other.count) and (self.count_diff == other.count_diff)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        text = '%s: size=%s (%s), count=%i (%+i)' % (self.traceback, _format_size(self.size, False), _format_size(self.size_diff, True), self.count, self.count_diff)
        if self.count:
            average = self.size / self.count
            text += ', average=%s' % _format_size(average, False)
        return text

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<StatisticDiff traceback=%r size=%i (%+i) count=%i (%+i)>' % (self.traceback, self.size, self.size_diff, self.count, self.count_diff)

    def _sort_key(self):
        if False:
            return 10
        return (abs(self.size_diff), self.size, abs(self.count_diff), self.count, self.traceback)

def _compare_grouped_stats(old_group, new_group):
    if False:
        print('Hello World!')
    statistics = []
    for (traceback, stat) in new_group.items():
        previous = old_group.pop(traceback, None)
        if previous is not None:
            stat = StatisticDiff(traceback, stat.size, stat.size - previous.size, stat.count, stat.count - previous.count)
        else:
            stat = StatisticDiff(traceback, stat.size, stat.size, stat.count, stat.count)
        statistics.append(stat)
    for (traceback, stat) in old_group.items():
        stat = StatisticDiff(traceback, 0, -stat.size, 0, -stat.count)
        statistics.append(stat)
    return statistics

@total_ordering
class Frame:
    """
    Frame of a traceback.
    """
    __slots__ = ('_frame',)

    def __init__(self, frame):
        if False:
            return 10
        self._frame = frame

    @property
    def filename(self):
        if False:
            i = 10
            return i + 15
        return self._frame[0]

    @property
    def lineno(self):
        if False:
            print('Hello World!')
        return self._frame[1]

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Frame):
            return NotImplemented
        return self._frame == other._frame

    def __lt__(self, other):
        if False:
            return 10
        if not isinstance(other, Frame):
            return NotImplemented
        return self._frame < other._frame

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self._frame)

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s:%s' % (self.filename, self.lineno)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<Frame filename=%r lineno=%r>' % (self.filename, self.lineno)

@total_ordering
class Traceback(Sequence):
    """
    Sequence of Frame instances sorted from the oldest frame
    to the most recent frame.
    """
    __slots__ = ('_frames', '_total_nframe')

    def __init__(self, frames, total_nframe=None):
        if False:
            print('Hello World!')
        Sequence.__init__(self)
        self._frames = tuple(reversed(frames))
        self._total_nframe = total_nframe

    @property
    def total_nframe(self):
        if False:
            for i in range(10):
                print('nop')
        return self._total_nframe

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._frames)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        if isinstance(index, slice):
            return tuple((Frame(trace) for trace in self._frames[index]))
        else:
            return Frame(self._frames[index])

    def __contains__(self, frame):
        if False:
            return 10
        return frame._frame in self._frames

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self._frames)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Traceback):
            return NotImplemented
        return self._frames == other._frames

    def __lt__(self, other):
        if False:
            return 10
        if not isinstance(other, Traceback):
            return NotImplemented
        return self._frames < other._frames

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self[0])

    def __repr__(self):
        if False:
            return 10
        s = f'<Traceback {tuple(self)}'
        if self._total_nframe is None:
            s += '>'
        else:
            s += f' total_nframe={self.total_nframe}>'
        return s

    def format(self, limit=None, most_recent_first=False):
        if False:
            return 10
        lines = []
        if limit is not None:
            if limit > 0:
                frame_slice = self[-limit:]
            else:
                frame_slice = self[:limit]
        else:
            frame_slice = self
        if most_recent_first:
            frame_slice = reversed(frame_slice)
        for frame in frame_slice:
            lines.append('  File "%s", line %s' % (frame.filename, frame.lineno))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                lines.append('    %s' % line)
        return lines

def get_object_traceback(obj):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the traceback where the Python object *obj* was allocated.\n    Return a Traceback instance.\n\n    Return None if the tracemalloc module is not tracing memory allocations or\n    did not trace the allocation of the object.\n    '
    frames = _get_object_traceback(obj)
    if frames is not None:
        return Traceback(frames)
    else:
        return None

class Trace:
    """
    Trace of a memory block.
    """
    __slots__ = ('_trace',)

    def __init__(self, trace):
        if False:
            print('Hello World!')
        self._trace = trace

    @property
    def domain(self):
        if False:
            return 10
        return self._trace[0]

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return self._trace[1]

    @property
    def traceback(self):
        if False:
            for i in range(10):
                print('nop')
        return Traceback(*self._trace[2:])

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Trace):
            return NotImplemented
        return self._trace == other._trace

    def __hash__(self):
        if False:
            return 10
        return hash(self._trace)

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s: %s' % (self.traceback, _format_size(self.size, False))

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Trace domain=%s size=%s, traceback=%r>' % (self.domain, _format_size(self.size, False), self.traceback)

class _Traces(Sequence):

    def __init__(self, traces):
        if False:
            for i in range(10):
                print('nop')
        Sequence.__init__(self)
        self._traces = traces

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._traces)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        if isinstance(index, slice):
            return tuple((Trace(trace) for trace in self._traces[index]))
        else:
            return Trace(self._traces[index])

    def __contains__(self, trace):
        if False:
            i = 10
            return i + 15
        return trace._trace in self._traces

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, _Traces):
            return NotImplemented
        return self._traces == other._traces

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Traces len=%s>' % len(self)

def _normalize_filename(filename):
    if False:
        return 10
    filename = os.path.normcase(filename)
    if filename.endswith('.pyc'):
        filename = filename[:-1]
    return filename

class BaseFilter:

    def __init__(self, inclusive):
        if False:
            print('Hello World!')
        self.inclusive = inclusive

    def _match(self, trace):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class Filter(BaseFilter):

    def __init__(self, inclusive, filename_pattern, lineno=None, all_frames=False, domain=None):
        if False:
            i = 10
            return i + 15
        super().__init__(inclusive)
        self.inclusive = inclusive
        self._filename_pattern = _normalize_filename(filename_pattern)
        self.lineno = lineno
        self.all_frames = all_frames
        self.domain = domain

    @property
    def filename_pattern(self):
        if False:
            for i in range(10):
                print('nop')
        return self._filename_pattern

    def _match_frame_impl(self, filename, lineno):
        if False:
            while True:
                i = 10
        filename = _normalize_filename(filename)
        if not fnmatch.fnmatch(filename, self._filename_pattern):
            return False
        if self.lineno is None:
            return True
        else:
            return lineno == self.lineno

    def _match_frame(self, filename, lineno):
        if False:
            i = 10
            return i + 15
        return self._match_frame_impl(filename, lineno) ^ (not self.inclusive)

    def _match_traceback(self, traceback):
        if False:
            i = 10
            return i + 15
        if self.all_frames:
            if any((self._match_frame_impl(filename, lineno) for (filename, lineno) in traceback)):
                return self.inclusive
            else:
                return not self.inclusive
        else:
            (filename, lineno) = traceback[0]
            return self._match_frame(filename, lineno)

    def _match(self, trace):
        if False:
            print('Hello World!')
        (domain, size, traceback, total_nframe) = trace
        res = self._match_traceback(traceback)
        if self.domain is not None:
            if self.inclusive:
                return res and domain == self.domain
            else:
                return res or domain != self.domain
        return res

class DomainFilter(BaseFilter):

    def __init__(self, inclusive, domain):
        if False:
            while True:
                i = 10
        super().__init__(inclusive)
        self._domain = domain

    @property
    def domain(self):
        if False:
            print('Hello World!')
        return self._domain

    def _match(self, trace):
        if False:
            print('Hello World!')
        (domain, size, traceback, total_nframe) = trace
        return (domain == self.domain) ^ (not self.inclusive)

class Snapshot:
    """
    Snapshot of traces of memory blocks allocated by Python.
    """

    def __init__(self, traces, traceback_limit):
        if False:
            for i in range(10):
                print('nop')
        self.traces = _Traces(traces)
        self.traceback_limit = traceback_limit

    def dump(self, filename):
        if False:
            i = 10
            return i + 15
        '\n        Write the snapshot into a file.\n        '
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        if False:
            print('Hello World!')
        '\n        Load a snapshot from a file.\n        '
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    def _filter_trace(self, include_filters, exclude_filters, trace):
        if False:
            return 10
        if include_filters:
            if not any((trace_filter._match(trace) for trace_filter in include_filters)):
                return False
        if exclude_filters:
            if any((not trace_filter._match(trace) for trace_filter in exclude_filters)):
                return False
        return True

    def filter_traces(self, filters):
        if False:
            return 10
        '\n        Create a new Snapshot instance with a filtered traces sequence, filters\n        is a list of Filter or DomainFilter instances.  If filters is an empty\n        list, return a new Snapshot instance with a copy of the traces.\n        '
        if not isinstance(filters, Iterable):
            raise TypeError('filters must be a list of filters, not %s' % type(filters).__name__)
        if filters:
            include_filters = []
            exclude_filters = []
            for trace_filter in filters:
                if trace_filter.inclusive:
                    include_filters.append(trace_filter)
                else:
                    exclude_filters.append(trace_filter)
            new_traces = [trace for trace in self.traces._traces if self._filter_trace(include_filters, exclude_filters, trace)]
        else:
            new_traces = self.traces._traces.copy()
        return Snapshot(new_traces, self.traceback_limit)

    def _group_by(self, key_type, cumulative):
        if False:
            return 10
        if key_type not in ('traceback', 'filename', 'lineno'):
            raise ValueError('unknown key_type: %r' % (key_type,))
        if cumulative and key_type not in ('lineno', 'filename'):
            raise ValueError('cumulative mode cannot by used with key type %r' % key_type)
        stats = {}
        tracebacks = {}
        if not cumulative:
            for trace in self.traces._traces:
                (domain, size, trace_traceback, total_nframe) = trace
                try:
                    traceback = tracebacks[trace_traceback]
                except KeyError:
                    if key_type == 'traceback':
                        frames = trace_traceback
                    elif key_type == 'lineno':
                        frames = trace_traceback[:1]
                    else:
                        frames = ((trace_traceback[0][0], 0),)
                    traceback = Traceback(frames)
                    tracebacks[trace_traceback] = traceback
                try:
                    stat = stats[traceback]
                    stat.size += size
                    stat.count += 1
                except KeyError:
                    stats[traceback] = Statistic(traceback, size, 1)
        else:
            for trace in self.traces._traces:
                (domain, size, trace_traceback, total_nframe) = trace
                for frame in trace_traceback:
                    try:
                        traceback = tracebacks[frame]
                    except KeyError:
                        if key_type == 'lineno':
                            frames = (frame,)
                        else:
                            frames = ((frame[0], 0),)
                        traceback = Traceback(frames)
                        tracebacks[frame] = traceback
                    try:
                        stat = stats[traceback]
                        stat.size += size
                        stat.count += 1
                    except KeyError:
                        stats[traceback] = Statistic(traceback, size, 1)
        return stats

    def statistics(self, key_type, cumulative=False):
        if False:
            while True:
                i = 10
        '\n        Group statistics by key_type. Return a sorted list of Statistic\n        instances.\n        '
        grouped = self._group_by(key_type, cumulative)
        statistics = list(grouped.values())
        statistics.sort(reverse=True, key=Statistic._sort_key)
        return statistics

    def compare_to(self, old_snapshot, key_type, cumulative=False):
        if False:
            i = 10
            return i + 15
        '\n        Compute the differences with an old snapshot old_snapshot. Get\n        statistics as a sorted list of StatisticDiff instances, grouped by\n        group_by.\n        '
        new_group = self._group_by(key_type, cumulative)
        old_group = old_snapshot._group_by(key_type, cumulative)
        statistics = _compare_grouped_stats(old_group, new_group)
        statistics.sort(reverse=True, key=StatisticDiff._sort_key)
        return statistics

def take_snapshot():
    if False:
        while True:
            i = 10
    '\n    Take a snapshot of traces of memory blocks allocated by Python.\n    '
    if not is_tracing():
        raise RuntimeError('the tracemalloc module must be tracing memory allocations to take a snapshot')
    traces = _get_traces()
    traceback_limit = get_traceback_limit()
    return Snapshot(traces, traceback_limit)