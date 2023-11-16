from __future__ import absolute_import
import ctypes
import hashlib
from st2reactor.container.partitioners import DefaultPartitioner, get_all_enabled_sensors
__all__ = ['HashPartitioner', 'Range']
SUB_RANGE_SEPARATOR = '|'
RANGE_BOUNDARY_SEPARATOR = '..'

class Range(object):
    RANGE_MIN_ENUM = 'min'
    RANGE_MIN_VALUE = 0
    RANGE_MAX_ENUM = 'max'
    RANGE_MAX_VALUE = 2 ** 32

    def __init__(self, range_repr):
        if False:
            i = 10
            return i + 15
        (self.range_start, self.range_end) = self._get_range_boundaries(range_repr)

    def __contains__(self, item):
        if False:
            i = 10
            return i + 15
        return item >= self.range_start and item < self.range_end

    def _get_range_boundaries(self, range_repr):
        if False:
            for i in range(10):
                print('nop')
        range_repr = [value.strip() for value in range_repr.split(RANGE_BOUNDARY_SEPARATOR)]
        if len(range_repr) != 2:
            raise ValueError('Unsupported sub-range format %s.' % range_repr)
        range_start = self._get_valid_range_boundary(range_repr[0])
        range_end = self._get_valid_range_boundary(range_repr[1])
        if range_start > range_end:
            raise ValueError('Misconfigured range [%d..%d]' % (range_start, range_end))
        return (range_start, range_end)

    def _get_valid_range_boundary(self, boundary_value):
        if False:
            return 10
        if boundary_value.lower() == self.RANGE_MIN_ENUM:
            return self.RANGE_MIN_VALUE
        if boundary_value.lower() == self.RANGE_MAX_ENUM:
            return self.RANGE_MAX_VALUE
        boundary_value = int(boundary_value)
        if boundary_value < self.RANGE_MIN_VALUE:
            return self.RANGE_MIN_VALUE
        if boundary_value > self.RANGE_MAX_VALUE:
            return self.RANGE_MAX_VALUE
        return boundary_value

class HashPartitioner(DefaultPartitioner):

    def __init__(self, sensor_node_name, hash_ranges):
        if False:
            while True:
                i = 10
        super(HashPartitioner, self).__init__(sensor_node_name=sensor_node_name)
        self._hash_ranges = self._create_hash_ranges(hash_ranges)

    def is_sensor_owner(self, sensor_db):
        if False:
            for i in range(10):
                print('nop')
        return self._is_in_hash_range(sensor_db.get_reference().ref)

    def get_sensors(self):
        if False:
            while True:
                i = 10
        all_enabled_sensors = get_all_enabled_sensors()
        partition_members = []
        for sensor in all_enabled_sensors:
            sensor_ref = sensor.get_reference()
            if self._is_in_hash_range(sensor_ref.ref):
                partition_members.append(sensor)
        return partition_members

    def _is_in_hash_range(self, sensor_ref):
        if False:
            while True:
                i = 10
        sensor_ref_hash = self._hash_sensor_ref(sensor_ref)
        for hash_range in self._hash_ranges:
            if sensor_ref_hash in hash_range:
                return True
        return False

    def _hash_sensor_ref(self, sensor_ref):
        if False:
            print('Hello World!')
        md5_hash = hashlib.md5(sensor_ref.encode())
        md5_hash_int_repr = int(md5_hash.hexdigest(), 16)
        h = ctypes.c_uint(0)
        for d in reversed(str(md5_hash_int_repr)):
            d = ctypes.c_uint(int(d))
            higherorder = ctypes.c_uint(h.value & 4160749568)
            h = ctypes.c_uint(h.value << 5)
            h = ctypes.c_uint(h.value ^ higherorder.value >> 27)
            h = ctypes.c_uint(h.value ^ d.value)
        return h.value

    def _create_hash_ranges(self, hash_ranges_repr):
        if False:
            i = 10
            return i + 15
        '\n        Extract from a format like - 0..1024|2048..4096|4096..MAX\n        '
        hash_ranges = []
        for range_repr in hash_ranges_repr.split(SUB_RANGE_SEPARATOR):
            hash_range = Range(range_repr.strip())
            hash_ranges.append(hash_range)
        return hash_ranges