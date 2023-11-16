import unittest
import paddle

class TestDygraphModeOfUnittest(unittest.TestCase):

    def test_dygraph_mode(self):
        if False:
            return 10
        self.assertTrue(paddle.in_dynamic_mode(), 'Default Mode of Unittest should be dygraph mode, but get static graph mode.')
if __name__ == '__main__':
    unittest.main()