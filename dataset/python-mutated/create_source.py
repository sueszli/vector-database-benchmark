from apache_beam.io import iobase
from apache_beam.transforms.core import Create

class _CreateSource(iobase.BoundedSource):
    """Internal source that is used by Create()"""

    def __init__(self, serialized_values, coder):
        if False:
            for i in range(10):
                print('nop')
        self._coder = coder
        self._serialized_values = []
        self._total_size = 0
        self._serialized_values = serialized_values
        self._total_size = sum(map(len, self._serialized_values))

    def read(self, range_tracker):
        if False:
            return 10
        start_position = range_tracker.start_position()
        current_position = start_position

        def split_points_unclaimed(stop_position):
            if False:
                for i in range(10):
                    print('nop')
            if current_position >= stop_position:
                return 0
            return stop_position - current_position - 1
        range_tracker.set_split_points_unclaimed_callback(split_points_unclaimed)
        element_iter = iter(self._serialized_values[start_position:])
        for i in range(start_position, range_tracker.stop_position()):
            if not range_tracker.try_claim(i):
                return
            current_position = i
            yield self._coder.decode(next(element_iter))

    def split(self, desired_bundle_size, start_position=None, stop_position=None):
        if False:
            while True:
                i = 10
        if len(self._serialized_values) < 2:
            yield iobase.SourceBundle(weight=0, source=self, start_position=0, stop_position=len(self._serialized_values))
        else:
            if start_position is None:
                start_position = 0
            if stop_position is None:
                stop_position = len(self._serialized_values)
            avg_size_per_value = self._total_size // len(self._serialized_values)
            num_values_per_split = max(int(desired_bundle_size // avg_size_per_value), 1)
            start = start_position
            while start < stop_position:
                end = min(start + num_values_per_split, stop_position)
                remaining = stop_position - end
                if remaining < num_values_per_split // 4:
                    end = stop_position
                sub_source = Create._create_source(self._serialized_values[start:end], self._coder)
                yield iobase.SourceBundle(weight=end - start, source=sub_source, start_position=0, stop_position=end - start)
                start = end

    def get_range_tracker(self, start_position, stop_position):
        if False:
            for i in range(10):
                print('nop')
        if start_position is None:
            start_position = 0
        if stop_position is None:
            stop_position = len(self._serialized_values)
        from apache_beam import io
        return io.OffsetRangeTracker(start_position, stop_position)

    def estimate_size(self):
        if False:
            while True:
                i = 10
        return self._total_size