import unittest
import legacy_test.test_collective_api_base as test_base

class TestCollectiveIsendIrecvAPI(test_base.TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_isend_irecv_nccl_dygraph(self):
        if False:
            return 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_isend_irecv_api_dygraph.py', 'sendrecv', 'nccl', static_mode='0', dtype=dtype)
if __name__ == '__main__':
    unittest.main()