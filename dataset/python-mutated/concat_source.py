"""For internal use only; no backwards-compatibility guarantees.

Concat Source, which reads the union of several other sources.
"""
import bisect
import threading
from apache_beam.io import iobase

class ConcatSource(iobase.BoundedSource):
    """For internal use only; no backwards-compatibility guarantees.

  A ``BoundedSource`` that can group a set of ``BoundedSources``.

  Primarily for internal use, use the ``apache_beam.Flatten`` transform
  to create the union of several reads.
  """

    def __init__(self, sources):
        if False:
            i = 10
            return i + 15
        self._source_bundles = [source if isinstance(source, iobase.SourceBundle) else iobase.SourceBundle(None, source, None, None) for source in sources]

    @property
    def sources(self):
        if False:
            return 10
        return [s.source for s in self._source_bundles]

    def estimate_size(self):
        if False:
            for i in range(10):
                print('nop')
        return sum((s.source.estimate_size() for s in self._source_bundles))

    def split(self, desired_bundle_size=None, start_position=None, stop_position=None):
        if False:
            return 10
        if start_position or stop_position:
            raise ValueError('Multi-level initial splitting is not supported. Expected start and stop positions to be None. Received %r and %r respectively.' % (start_position, stop_position))
        for source in self._source_bundles:
            for bundle in source.source.split(desired_bundle_size, source.start_position, source.stop_position):
                yield bundle

    def get_range_tracker(self, start_position=None, stop_position=None):
        if False:
            print('Hello World!')
        if start_position is None:
            start_position = (0, None)
        if stop_position is None:
            stop_position = (len(self._source_bundles), None)
        return ConcatRangeTracker(start_position, stop_position, self._source_bundles)

    def read(self, range_tracker):
        if False:
            while True:
                i = 10
        (start_source, _) = range_tracker.start_position()
        (stop_source, stop_pos) = range_tracker.stop_position()
        if stop_pos is not None:
            stop_source += 1
        for source_ix in range(start_source, stop_source):
            if not range_tracker.try_claim((source_ix, None)):
                break
            for record in self._source_bundles[source_ix].source.read(range_tracker.sub_range_tracker(source_ix)):
                yield record

    def default_output_coder(self):
        if False:
            while True:
                i = 10
        if self._source_bundles:
            return self._source_bundles[0].source.default_output_coder()
        else:
            return super().default_output_coder()

class ConcatRangeTracker(iobase.RangeTracker):
    """For internal use only; no backwards-compatibility guarantees.

  Range tracker for ConcatSource"""

    def __init__(self, start, end, source_bundles):
        if False:
            while True:
                i = 10
        'Initializes ``ConcatRangeTracker``\n\n    Args:\n      start: start position, a tuple of (source_index, source_position)\n      end: end position, a tuple of (source_index, source_position)\n      source_bundles: the list of source bundles in the ConcatSource\n    '
        super().__init__()
        self._start = start
        self._end = end
        self._source_bundles = source_bundles
        self._lock = threading.RLock()
        self._range_trackers = [None] * len(source_bundles)
        self._claimed_source_ix = self._start[0]
        last = end[0] if end[1] is None else end[0] + 1
        self._cumulative_weights = [0] * start[0] + self._compute_cumulative_weights(source_bundles[start[0]:last]) + [1] * (len(source_bundles) - last - start[0])

    @staticmethod
    def _compute_cumulative_weights(source_bundles):
        if False:
            print('Hello World!')
        min_diff = 1e-05
        known = [s.weight for s in source_bundles if s.weight is not None]
        avg = sum(known) / len(known) if known else 1.0
        weights = [s.weight or avg for s in source_bundles]
        total = float(sum(weights))
        running_total = [0]
        for w in weights:
            running_total.append(max(min_diff, min(1, running_total[-1] + w / total)))
        running_total[-1] = 1
        for k in range(1, len(running_total)):
            if running_total[k] == running_total[k - 1]:
                for j in range(k):
                    running_total[j] *= 1 - min_diff
        return running_total

    def start_position(self):
        if False:
            i = 10
            return i + 15
        return self._start

    def stop_position(self):
        if False:
            while True:
                i = 10
        return self._end

    def try_claim(self, pos):
        if False:
            while True:
                i = 10
        (source_ix, source_pos) = pos
        with self._lock:
            if source_ix > self._end[0]:
                return False
            elif source_ix == self._end[0] and self._end[1] is None:
                return False
            else:
                assert source_ix >= self._claimed_source_ix
                self._claimed_source_ix = source_ix
                if source_pos is None:
                    return True
                else:
                    return self.sub_range_tracker(source_ix).try_claim(source_pos)

    def try_split(self, pos):
        if False:
            i = 10
            return i + 15
        (source_ix, source_pos) = pos
        with self._lock:
            if source_ix < self._claimed_source_ix:
                return None
            elif source_ix > self._end[0]:
                return None
            elif source_ix == self._end[0] and self._end[1] is None:
                return None
            else:
                if source_ix > self._claimed_source_ix:
                    split_pos = None
                    ratio = self._cumulative_weights[source_ix]
                else:
                    split = self.sub_range_tracker(source_ix).try_split(source_pos)
                    if not split:
                        return None
                    (split_pos, frac) = split
                    ratio = self.local_to_global(source_ix, frac)
                self._end = (source_ix, split_pos)
                self._cumulative_weights = [min(w / ratio, 1) for w in self._cumulative_weights]
                return ((source_ix, split_pos), ratio)

    def set_current_position(self, pos):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Should only be called on sub-trackers')

    def position_at_fraction(self, fraction):
        if False:
            print('Hello World!')
        (source_ix, source_frac) = self.global_to_local(fraction)
        last = self._end[0] if self._end[1] is None else self._end[0] + 1
        if source_ix == last:
            return (source_ix, None)
        else:
            return (source_ix, self.sub_range_tracker(source_ix).position_at_fraction(source_frac))

    def fraction_consumed(self):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            if self._claimed_source_ix == len(self._source_bundles):
                return 1.0
            else:
                return self.local_to_global(self._claimed_source_ix, self.sub_range_tracker(self._claimed_source_ix).fraction_consumed())

    def local_to_global(self, source_ix, source_frac):
        if False:
            for i in range(10):
                print('nop')
        cw = self._cumulative_weights
        return cw[source_ix] + source_frac * (cw[source_ix + 1] - cw[source_ix])

    def global_to_local(self, frac):
        if False:
            for i in range(10):
                print('nop')
        if frac == 1:
            last = self._end[0] if self._end[1] is None else self._end[0] + 1
            return (last, None)
        else:
            cw = self._cumulative_weights
            source_ix = bisect.bisect(cw, frac) - 1
            return (source_ix, (frac - cw[source_ix]) / (cw[source_ix + 1] - cw[source_ix]))

    def sub_range_tracker(self, source_ix):
        if False:
            while True:
                i = 10
        assert self._start[0] <= source_ix <= self._end[0]
        if self._range_trackers[source_ix] is None:
            with self._lock:
                if self._range_trackers[source_ix] is None:
                    source = self._source_bundles[source_ix]
                    if source_ix == self._start[0] and self._start[1] is not None:
                        start = self._start[1]
                    else:
                        start = source.start_position
                    if source_ix == self._end[0] and self._end[1] is not None:
                        stop = self._end[1]
                    else:
                        stop = source.stop_position
                    self._range_trackers[source_ix] = source.source.get_range_tracker(start, stop)
        return self._range_trackers[source_ix]