import unittest
import paddle.dataset.voc2012
__all__ = []

class TestVOC(unittest.TestCase):

    def check_reader(self, reader):
        if False:
            print('Hello World!')
        sum = 0
        label = 0
        for l in reader():
            self.assertEqual(l[0].size, 3 * l[1].size)
            sum += 1
        return sum

    def test_train(self):
        if False:
            while True:
                i = 10
        count = self.check_reader(paddle.dataset.voc_seg.train())
        self.assertEqual(count, 2913)

    def test_test(self):
        if False:
            for i in range(10):
                print('nop')
        count = self.check_reader(paddle.dataset.voc_seg.test())
        self.assertEqual(count, 1464)

    def test_val(self):
        if False:
            return 10
        count = self.check_reader(paddle.dataset.voc_seg.val())
        self.assertEqual(count, 1449)
if __name__ == '__main__':
    unittest.main()