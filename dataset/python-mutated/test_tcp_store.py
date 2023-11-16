import os
import unittest
import paddle

class TestTCPStore(unittest.TestCase):

    def test_tcp_store(self):
        if False:
            for i in range(10):
                print('nop')
        dist_port = int(os.getenv('PADDLE_DIST_UT_PORT', 6170))
        print('get dist_port:', dist_port)
        store = paddle.base.core.TCPStore('127.0.0.1', dist_port, True, 1, 1)
        store.add('my', 3)
        ret1 = store.get('my')
        store.add('my', 3)
        ret2 = store.get('my')
        self.assertEqual(ret1[0] + 3, ret2[0])
if __name__ == '__main__':
    unittest.main()