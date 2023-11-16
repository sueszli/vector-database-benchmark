import copy
import itertools
import numpy as np
from urh.util import util
from urh.util.GenericCRC import GenericCRC

class CommonRange(object):

    def __init__(self, start, length, value: np.ndarray=None, score=0, field_type='Generic', message_indices=None, range_type='bit', byte_order='big'):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        :param start:\n        :param length:\n        :param value: Value for this common range as string\n        '
        self.start = start
        self.length = length
        self.__byte_order = byte_order
        self.sync_end = 0
        if isinstance(value, str):
            value = np.array(list(map(lambda x: int(x, 16), value)), dtype=np.uint8)
        self.values = [value] if value is not None else []
        self.score = score
        self.field_type = field_type
        self.range_type = range_type.lower()
        self.message_indices = set() if message_indices is None else set(message_indices)
        '\n        Set of message indices, this range applies to\n        '

    @property
    def end(self):
        if False:
            i = 10
            return i + 15
        return self.start + self.length - 1

    @property
    def bit_start(self):
        if False:
            while True:
                i = 10
        return self.__convert_number(self.start) + self.sync_end

    @property
    def bit_end(self):
        if False:
            print('Hello World!')
        return self.__convert_number(self.start) + self.__convert_number(self.length) - 1 + self.sync_end

    @property
    def length_in_bits(self):
        if False:
            print('Hello World!')
        return self.bit_end - self.bit_start - 1

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.values) == 0:
            return None
        elif len(self.values) == 1:
            return self.values[0]
        else:
            raise ValueError('This range has multiple values!')

    @value.setter
    def value(self, val):
        if False:
            return 10
        if len(self.values) == 0:
            self.values = [val]
        elif len(self.values) == 1:
            self.values[0] = val
        else:
            raise ValueError('This range has multiple values!')

    @property
    def byte_order(self):
        if False:
            while True:
                i = 10
        if self.byte_order_is_unknown:
            return 'big'
        return self.__byte_order

    @byte_order.setter
    def byte_order(self, val: str):
        if False:
            while True:
                i = 10
        self.__byte_order = val

    @property
    def byte_order_is_unknown(self) -> bool:
        if False:
            print('Hello World!')
        return self.__byte_order is None

    def matches(self, start: int, value: np.ndarray):
        if False:
            while True:
                i = 10
        return self.start == start and self.length == len(value) and (self.value.tobytes() == value.tobytes())

    def __convert_number(self, n):
        if False:
            for i in range(10):
                print('nop')
        if self.range_type == 'bit':
            return n
        elif self.range_type == 'hex':
            return n * 4
        elif self.range_type == 'byte':
            return n * 8
        else:
            raise ValueError('Unknown range type {}'.format(self.range_type))

    def __repr__(self):
        if False:
            return 10
        result = '{} {}-{} ({} {})'.format(self.field_type, self.bit_start, self.bit_end, self.length, self.range_type)
        result += ' Values: ' + ' '.join(map(util.convert_numbers_to_hex_string, self.values))
        if self.score is not None:
            result += ' Score: ' + str(self.score)
        result += ' Message indices: {' + ','.join(map(str, sorted(self.message_indices))) + '}'
        return result

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, CommonRange):
            return False
        return self.bit_start == other.bit_start and self.bit_end == other.bit_end and (self.field_type == other.field_type)

    def __hash__(self):
        if False:
            return 10
        return hash((self.start, self.length, self.field_type))

    def __lt__(self, other):
        if False:
            print('Hello World!')
        return self.bit_start < other.bit_start

    def overlaps_with(self, other) -> bool:
        if False:
            return 10
        if not isinstance(other, CommonRange):
            raise ValueError('Need another bit range to compare')
        return any((i in range(self.bit_start, self.bit_end) for i in range(other.bit_start, other.bit_end)))

    def ensure_not_overlaps(self, start: int, end: int):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        :param start:\n        :param end:\n        :rtype: list of CommonRange\n        '
        if end < self.start or start > self.end:
            return [copy.deepcopy(self)]
        if start <= self.start < end < self.end:
            result = copy.deepcopy(self)
            result.length -= end - result.start
            result.start = end
            result.value = result.value[result.start - self.start:result.start - self.start + result.length]
            return [result]
        if self.start < start <= self.end <= end:
            result = copy.deepcopy(self)
            result.length -= self.end + 1 - start
            result.value = result.value[:result.length]
            return [result]
        if self.start < start and self.end > end:
            left = copy.deepcopy(self)
            right = copy.deepcopy(self)
            left.length -= left.end + 1 - start
            left.value = self.value[:left.length]
            right.start = end + 1
            right.length = self.end - end
            right.value = self.value[right.start - self.start:right.start - self.start + right.length]
            return [left, right]
        return []

class ChecksumRange(CommonRange):

    def __init__(self, start, length, crc: GenericCRC, data_range_start, data_range_end, value: np.ndarray=None, score=0, field_type='Generic', message_indices=None, range_type='bit'):
        if False:
            i = 10
            return i + 15
        super().__init__(start, length, value, score, field_type, message_indices, range_type)
        self.data_range_start = data_range_start
        self.data_range_end = data_range_end
        self.crc = crc

    @property
    def data_range_bit_start(self):
        if False:
            for i in range(10):
                print('nop')
        return self.data_range_start + self.sync_end

    @property
    def data_range_bit_end(self):
        if False:
            print('Hello World!')
        return self.data_range_end + self.sync_end

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return super().__eq__(other) and self.data_range_start == other.data_range_start and (self.data_range_end == other.data_range_end) and (self.crc == other.crc)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.start, self.length, self.data_range_start, self.data_range_end, self.crc))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return super().__repr__() + ' \t' + '{}'.format(self.crc.caption) + ' Datarange: {}-{} '.format(self.data_range_start, self.data_range_end)

class EmptyCommonRange(CommonRange):
    """
    Empty Common Bit Range, to indicate, that no common Bit Range was found
    """

    def __init__(self, field_type='Generic'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(0, 0, '')
        self.field_type = field_type

    def __eq__(self, other):
        if False:
            return 10
        return isinstance(other, EmptyCommonRange) and other.field_type == self.field_type

    def __repr__(self):
        if False:
            return 10
        return 'No ' + self.field_type

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(super)

class CommonRangeContainer(object):
    """
    This is the raw equivalent of a Message Type:
    A container of common ranges
    """

    def __init__(self, ranges: list, message_indices: set=None):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(ranges, list)
        self.__ranges = ranges
        self.__ranges.sort()
        if message_indices is None:
            self.update_message_indices()
        else:
            self.message_indices = message_indices

    @property
    def ranges_overlap(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.has_overlapping_ranges(self.__ranges)

    def update_message_indices(self):
        if False:
            return 10
        if len(self) == 0:
            self.message_indices = set()
        else:
            self.message_indices = set(self[0].message_indices)
            for i in range(1, len(self)):
                self.message_indices.intersection_update(self[i].message_indices)

    def add_range(self, rng: CommonRange):
        if False:
            i = 10
            return i + 15
        self.__ranges.append(rng)
        self.__ranges.sort()

    def add_ranges(self, ranges: list):
        if False:
            print('Hello World!')
        self.__ranges.extend(ranges)
        self.__ranges.sort()

    def has_same_ranges(self, ranges: list) -> bool:
        if False:
            i = 10
            return i + 15
        return self.__ranges == ranges

    def has_same_ranges_as_container(self, container):
        if False:
            while True:
                i = 10
        if not isinstance(container, CommonRangeContainer):
            return False
        return self.__ranges == container.__ranges

    @staticmethod
    def has_overlapping_ranges(ranges: list) -> bool:
        if False:
            while True:
                i = 10
        for (rng1, rng2) in itertools.combinations(ranges, 2):
            if rng1.overlaps_with(rng2):
                return True
        return False

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.__ranges)

    def __iter__(self):
        if False:
            print('Hello World!')
        return self.__ranges.__iter__()

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        return self.__ranges[item]

    def __repr__(self):
        if False:
            while True:
                i = 10
        from pprint import pformat
        return pformat(self.__ranges)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, CommonRangeContainer):
            return False
        return self.__ranges == other.__ranges and self.message_indices == other.message_indices