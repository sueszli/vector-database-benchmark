from __future__ import absolute_import
import pytest
from kafka.partitioner import DefaultPartitioner, murmur2

def test_default_partitioner():
    if False:
        print('Hello World!')
    partitioner = DefaultPartitioner()
    all_partitions = available = list(range(100))
    p1 = partitioner(b'foo', all_partitions, available)
    p2 = partitioner(b'foo', all_partitions, available)
    assert p1 == p2
    assert p1 in all_partitions
    assert partitioner(None, all_partitions, [123]) == 123
    assert partitioner(None, all_partitions, []) in all_partitions

@pytest.mark.parametrize('bytes_payload,partition_number', [(b'', 681), (b'a', 524), (b'ab', 434), (b'abc', 107), (b'123456789', 566), (b'\x00 ', 742)])
def test_murmur2_java_compatibility(bytes_payload, partition_number):
    if False:
        for i in range(10):
            print('nop')
    partitioner = DefaultPartitioner()
    all_partitions = available = list(range(1000))
    assert partitioner(bytes_payload, all_partitions, available) == partition_number

def test_murmur2_not_ascii():
    if False:
        for i in range(10):
            print('nop')
    murmur2(b'\xa4')
    murmur2(b'\x81' * 1000)