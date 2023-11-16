import random
import numpy as np
from ding.utils.compression_helper import get_data_compressor, get_data_decompressor
import pytest

@pytest.mark.unittest
class TestCompression:

    def get_step_data(self):
        if False:
            print('Hello World!')
        return {'input': [random.randint(10, 100) for i in range(100)]}

    def testnaive(self):
        if False:
            for i in range(10):
                print('nop')
        compress_names = ['lz4', 'zlib', 'none']
        for s in compress_names:
            compressor = get_data_compressor(s)
            decompressor = get_data_decompressor(s)
            data = self.get_step_data()
            assert data == decompressor(compressor(data))

    def test_arr_to_st(self):
        if False:
            while True:
                i = 10
        data = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        compress_names = ['jpeg']
        for s in compress_names:
            compressor = get_data_compressor(s)
            decompressor = get_data_decompressor(s)
            assert data.shape == decompressor(compressor(data)).shape