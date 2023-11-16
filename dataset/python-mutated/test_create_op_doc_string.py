import unittest
import paddle

class TestDocString(unittest.TestCase):

    def test_layer_doc_string(self):
        if False:
            for i in range(10):
                print('nop')
        print(paddle.nn.functional.dropout.__doc__)
if __name__ == '__main__':
    unittest.main()