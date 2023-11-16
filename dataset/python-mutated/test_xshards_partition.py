from unittest import TestCase
import pytest
import numpy as np
from bigdl.orca.data import XShards

class TestXshardsPartition(TestCase):

    def test_partition_ndarray(self):
        if False:
            print('Hello World!')
        data = np.random.randn(10, 4)
        xshards = XShards.partition(data)
        data_parts = xshards.rdd.collect()
        reconstructed = np.concatenate(data_parts)
        assert np.allclose(data, reconstructed)

    def test_partition_tuple(self):
        if False:
            return 10
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)
        xshards = XShards.partition((data1, data2))
        data_parts = xshards.rdd.collect()
        data1_parts = [part[0] for part in data_parts]
        data2_parts = [part[1] for part in data_parts]
        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_list(self):
        if False:
            i = 10
            return i + 15
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)
        xshards = XShards.partition([data1, data2])
        data_parts = xshards.rdd.collect()
        data1_parts = [part[0] for part in data_parts]
        data2_parts = [part[1] for part in data_parts]
        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_dict(self):
        if False:
            while True:
                i = 10
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)
        xshards = XShards.partition({'x': data1, 'y': data2})
        data_parts = xshards.rdd.collect()
        data1_parts = [part['x'] for part in data_parts]
        data2_parts = [part['y'] for part in data_parts]
        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_nested(self):
        if False:
            while True:
                i = 10
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)
        xshards = XShards.partition({'x': (data1,), 'y': [data2]})
        data_parts = xshards.rdd.collect()
        data1_parts = [part['x'][0] for part in data_parts]
        data2_parts = [part['y'][0] for part in data_parts]
        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_ndarray_with_num_shards_specification(self):
        if False:
            while True:
                i = 10
        data = np.random.randn(10, 4)
        xshards = XShards.partition(data, num_shards=2)
        data_parts = xshards.rdd.collect()
        reconstructed = np.concatenate(data_parts)
        assert np.allclose(data, reconstructed)
        with pytest.raises(RuntimeError) as errorInfo:
            xshards = XShards.partition(data, num_shards=20)
        assert errorInfo.type == RuntimeError
        assert 'number of shards' in str(errorInfo.value)

    def test_partition_nested_with_num_shards_specification(self):
        if False:
            return 10
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)
        xshards = XShards.partition({'x': (data1,), 'y': [data2]}, num_shards=2)
        data_parts = xshards.rdd.collect()
        data1_parts = [part['x'][0] for part in data_parts]
        data2_parts = [part['y'][0] for part in data_parts]
        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)
        with pytest.raises(RuntimeError) as errorInfo:
            xshards = XShards.partition({'x': (data1,), 'y': [data2]}, num_shards=20)
        assert errorInfo.type == RuntimeError
        assert 'number of shards' in str(errorInfo.value)
if __name__ == '__main__':
    pytest.main([__file__])