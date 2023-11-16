import unittest
import paddle
from paddle.distributed import fleet
paddle.enable_static()

class TestCollectiveSplitAssert(unittest.TestCase):

    def network(self):
        if False:
            return 10
        fleet.init()
        data = paddle.static.data(name='tindata', shape=[10, 1000], dtype='float32')
        emb_out = paddle.distributed.split(data, (7, 8), operation='embedding', num_partitions=2)

    def test_assert(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AssertionError):
            self.network()
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()