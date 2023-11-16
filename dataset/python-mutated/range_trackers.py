"""iobase.RangeTracker implementations provided with Apache Beam.
"""
import codecs
import logging
import math
import threading
from typing import Union
from apache_beam.io import iobase
__all__ = ['OffsetRangeTracker', 'LexicographicKeyRangeTracker', 'OrderedPositionRangeTracker', 'UnsplittableRangeTracker']
_LOGGER = logging.getLogger(__name__)

class OffsetRangeTracker(iobase.RangeTracker):
    """A 'RangeTracker' for non-negative positions of type 'long'."""
    OFFSET_INFINITY = float('inf')

    def __init__(self, start, end):
        if False:
            while True:
                i = 10
        super().__init__()
        if start is None:
            raise ValueError("Start offset must not be 'None'")
        if end is None:
            raise ValueError("End offset must not be 'None'")
        assert isinstance(start, int)
        if end != self.OFFSET_INFINITY:
            assert isinstance(end, int)
        assert start <= end
        self._start_offset = start
        self._stop_offset = end
        self._last_record_start = -1
        self._last_attempted_record_start = -1
        self._offset_of_last_split_point = -1
        self._lock = threading.Lock()
        self._split_points_seen = 0
        self._split_points_unclaimed_callback = None

    def start_position(self):
        if False:
            for i in range(10):
                print('nop')
        return self._start_offset

    def stop_position(self):
        if False:
            print('Hello World!')
        return self._stop_offset

    @property
    def last_record_start(self):
        if False:
            return 10
        return self._last_record_start

    @property
    def last_attempted_record_start(self):
        if False:
            for i in range(10):
                print('nop')
        'Return current value of last_attempted_record_start.\n\n    last_attempted_record_start records a valid position that tried to be\n    claimed by calling try_claim(). This value is only updated by `try_claim()`\n    no matter `try_claim()` returns `True` or `False`.\n    '
        return self._last_attempted_record_start

    def _validate_record_start(self, record_start, split_point):
        if False:
            for i in range(10):
                print('nop')
        if not self._lock.locked():
            raise ValueError('This function must only be called under the lock self.lock.')
        if record_start < self._last_record_start:
            raise ValueError('Trying to return a record [starting at %d] which is before the last-returned record [starting at %d]' % (record_start, self._last_record_start))
        if split_point and self._offset_of_last_split_point != -1 and (record_start == self._offset_of_last_split_point):
            raise ValueError('Record at a split point has same offset as the previous split point: %d' % record_start)
        if not split_point and self._last_record_start == -1:
            raise ValueError('The first record [starting at %d] must be at a split point' % record_start)

    def try_claim(self, record_start):
        if False:
            i = 10
            return i + 15
        with self._lock:
            if record_start <= self._last_attempted_record_start:
                raise ValueError('Trying to return a record [starting at %d] which is not greaterthan the last-attempted record [starting at %d]' % (record_start, self._last_attempted_record_start))
            self._validate_record_start(record_start, True)
            self._last_attempted_record_start = record_start
            if record_start >= self.stop_position():
                return False
            self._offset_of_last_split_point = record_start
            self._last_record_start = record_start
            self._split_points_seen += 1
            return True

    def set_current_position(self, record_start):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            self._validate_record_start(record_start, False)
            self._last_record_start = record_start

    def try_split(self, split_offset):
        if False:
            return 10
        assert isinstance(split_offset, int)
        with self._lock:
            if self._stop_offset == OffsetRangeTracker.OFFSET_INFINITY:
                _LOGGER.debug('refusing to split %r at %d: stop position unspecified', self, split_offset)
                return
            if self._last_record_start == -1:
                _LOGGER.debug('Refusing to split %r at %d: unstarted', self, split_offset)
                return
            if split_offset <= self._last_record_start:
                _LOGGER.debug('Refusing to split %r at %d: already past proposed stop offset', self, split_offset)
                return
            if split_offset < self.start_position() or split_offset >= self.stop_position():
                _LOGGER.debug('Refusing to split %r at %d: proposed split position out of range', self, split_offset)
                return
            _LOGGER.debug('Agreeing to split %r at %d', self, split_offset)
            split_fraction = float(split_offset - self._start_offset) / (self._stop_offset - self._start_offset)
            self._stop_offset = split_offset
            return (self._stop_offset, split_fraction)

    def fraction_consumed(self):
        if False:
            print('Hello World!')
        with self._lock:
            return self.position_to_fraction(self._last_record_start, self.start_position(), self.stop_position())

    def position_to_fraction(self, pos, start, stop):
        if False:
            for i in range(10):
                print('nop')
        fraction = 1.0 * (pos - start) / (stop - start) if start != stop else 0.0
        return max(0.0, min(1.0, fraction))

    def position_at_fraction(self, fraction):
        if False:
            i = 10
            return i + 15
        if self.stop_position() == OffsetRangeTracker.OFFSET_INFINITY:
            raise Exception('get_position_for_fraction_consumed is not applicable for an unbounded range')
        return int(math.ceil(self.start_position() + fraction * (self.stop_position() - self.start_position())))

    def split_points(self):
        if False:
            while True:
                i = 10
        with self._lock:
            split_points_consumed = 0 if self._split_points_seen == 0 else self._split_points_seen - 1
            split_points_unclaimed = self._split_points_unclaimed_callback(self.stop_position()) if self._split_points_unclaimed_callback else iobase.RangeTracker.SPLIT_POINTS_UNKNOWN
            split_points_remaining = iobase.RangeTracker.SPLIT_POINTS_UNKNOWN if split_points_unclaimed == iobase.RangeTracker.SPLIT_POINTS_UNKNOWN else split_points_unclaimed + 1
            return (split_points_consumed, split_points_remaining)

    def set_split_points_unclaimed_callback(self, callback):
        if False:
            print('Hello World!')
        self._split_points_unclaimed_callback = callback

class OrderedPositionRangeTracker(iobase.RangeTracker):
    """
  An abstract base class for range trackers whose positions are comparable.

  Subclasses only need to implement the mapping from position ranges
  to and from the closed interval [0, 1].
  """
    UNSTARTED = object()

    def __init__(self, start_position=None, stop_position=None):
        if False:
            return 10
        self._start_position = start_position
        self._stop_position = stop_position
        self._lock = threading.Lock()
        self._last_claim = self.UNSTARTED

    def start_position(self):
        if False:
            return 10
        return self._start_position

    def stop_position(self):
        if False:
            return 10
        with self._lock:
            return self._stop_position

    def try_claim(self, position):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            if self._last_claim is not self.UNSTARTED and position < self._last_claim:
                raise ValueError("Positions must be claimed in order: claim '%s' attempted after claim '%s'" % (position, self._last_claim))
            elif self._start_position is not None and position < self._start_position:
                raise ValueError("Claim '%s' is before start '%s'" % (position, self._start_position))
            if self._stop_position is None or position < self._stop_position:
                self._last_claim = position
                return True
            else:
                return False

    def position_at_fraction(self, fraction):
        if False:
            for i in range(10):
                print('nop')
        return self.fraction_to_position(fraction, self._start_position, self._stop_position)

    def try_split(self, position):
        if False:
            print('Hello World!')
        with self._lock:
            if self._stop_position is not None and position >= self._stop_position or (self._start_position is not None and position <= self._start_position):
                _LOGGER.debug('Refusing to split %r at %d: proposed split position out of range', self, position)
                return
            if self._last_claim is self.UNSTARTED or self._last_claim < position:
                fraction = self.position_to_fraction(position, start=self._start_position, end=self._stop_position)
                self._stop_position = position
                return (position, fraction)

    def fraction_consumed(self):
        if False:
            for i in range(10):
                print('nop')
        if self._last_claim is self.UNSTARTED:
            return 0
        else:
            return self.position_to_fraction(self._last_claim, self._start_position, self._stop_position)

    def fraction_to_position(self, fraction, start, end):
        if False:
            for i in range(10):
                print('nop')
        '\n    Converts a fraction between 0 and 1 to a position between start and end.\n    '
        raise NotImplementedError

    def position_to_fraction(self, position, start, end):
        if False:
            return 10
        'Returns the fraction of keys in the range [start, end) that\n    are less than the given key.\n    '
        raise NotImplementedError

class UnsplittableRangeTracker(iobase.RangeTracker):
    """A RangeTracker that always ignores split requests.

  This can be used to make a given
  :class:`~apache_beam.io.iobase.RangeTracker` object unsplittable by
  ignoring all calls to :meth:`.try_split()`. All other calls will be delegated
  to the given :class:`~apache_beam.io.iobase.RangeTracker`.
  """

    def __init__(self, range_tracker):
        if False:
            i = 10
            return i + 15
        'Initializes UnsplittableRangeTracker.\n\n    Args:\n      range_tracker (~apache_beam.io.iobase.RangeTracker): a\n        :class:`~apache_beam.io.iobase.RangeTracker` to which all method\n        calls except calls to :meth:`.try_split()` will be delegated.\n    '
        assert isinstance(range_tracker, iobase.RangeTracker)
        self._range_tracker = range_tracker

    def start_position(self):
        if False:
            i = 10
            return i + 15
        return self._range_tracker.start_position()

    def stop_position(self):
        if False:
            print('Hello World!')
        return self._range_tracker.stop_position()

    def position_at_fraction(self, fraction):
        if False:
            print('Hello World!')
        return self._range_tracker.position_at_fraction(fraction)

    def try_claim(self, position):
        if False:
            print('Hello World!')
        return self._range_tracker.try_claim(position)

    def try_split(self, position):
        if False:
            for i in range(10):
                print('nop')
        return None

    def set_current_position(self, position):
        if False:
            while True:
                i = 10
        self._range_tracker.set_current_position(position)

    def fraction_consumed(self):
        if False:
            while True:
                i = 10
        return self._range_tracker.fraction_consumed()

    def split_points(self):
        if False:
            while True:
                i = 10
        return (0, 1)

    def set_split_points_unclaimed_callback(self, callback):
        if False:
            return 10
        self._range_tracker.set_split_points_unclaimed_callback(callback)

class LexicographicKeyRangeTracker(OrderedPositionRangeTracker):
    """A range tracker that tracks progress through a lexicographically
  ordered keyspace of strings.
  """

    @classmethod
    def fraction_to_position(cls, fraction: float, start: Union[bytes, str]=None, end: Union[bytes, str]=None) -> Union[bytes, str]:
        if False:
            for i in range(10):
                print('nop')
        'Linearly interpolates a key that is lexicographically\n    fraction of the way between start and end.\n    '
        assert 0 <= fraction <= 1, fraction
        if start is None:
            start = b''
        if fraction == 0:
            return start
        if fraction == 1:
            return end
        if not end:
            common_prefix_len = len(start) - len(start.lstrip(b'\xff'))
        else:
            for (ix, (s, e)) in enumerate(zip(start, end)):
                if s != e:
                    common_prefix_len = ix
                    break
            else:
                common_prefix_len = min(len(start), len(end))
        prec = common_prefix_len + int(-math.log(fraction, 256)) + 7
        istart = cls._bytestring_to_int(start, prec)
        iend = cls._bytestring_to_int(end, prec) if end else 1 << prec * 8
        ikey = istart + int((iend - istart) * fraction)
        if ikey == istart:
            ikey += 1
        elif ikey == iend:
            ikey -= 1
        position: bytes = cls._bytestring_from_int(ikey, prec).rstrip(b'\x00')
        if isinstance(start, bytes):
            return position
        return position.decode(encoding='unicode_escape', errors='replace')

    @classmethod
    def position_to_fraction(cls, key: Union[bytes, str]=None, start: Union[bytes, str]=None, end: Union[bytes, str]=None) -> float:
        if False:
            i = 10
            return i + 15
        'Returns the fraction of keys in the range [start, end) that\n    are less than the given key.\n    '
        if not key:
            return 0
        if start is None:
            start = '' if isinstance(key, str) else b''
        prec = len(start) + 7
        if key.startswith(start):
            trailing_symbol = '\x00' if isinstance(key, str) else b'\x00'
            prec = max(prec, len(key) - len(key[len(start):].strip(trailing_symbol)) + 7)
        istart = cls._bytestring_to_int(start, prec)
        ikey = cls._bytestring_to_int(key, prec)
        iend = cls._bytestring_to_int(end, prec) if end else 1 << prec * 8
        return float(ikey - istart) / (iend - istart)

    @staticmethod
    def _bytestring_to_int(s: Union[bytes, str], prec: int) -> int:
        if False:
            print('Hello World!')
        "Returns int(256**prec * f) where f is the fraction\n    represented by interpreting '.' + s as a base-256\n    floating point number.\n    "
        if not s:
            return 0
        if isinstance(s, str):
            s = s.encode()
        if len(s) < prec:
            s += b'\x00' * (prec - len(s))
        else:
            s = s[:prec]
        h = codecs.encode(s, encoding='hex')
        return int(h, base=16)

    @staticmethod
    def _bytestring_from_int(i: int, prec: int) -> bytes:
        if False:
            print('Hello World!')
        'Inverse of _bytestring_to_int.'
        h = '%x' % i
        return codecs.decode('0' * (2 * prec - len(h)) + h, encoding='hex')