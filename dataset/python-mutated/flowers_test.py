import unittest
import paddle.dataset.flowers
__all__ = []

class TestFlowers(unittest.TestCase):

    def check_reader(self, reader):
        if False:
            print('Hello World!')
        sum = 0
        label = 0
        size = 224 * 224 * 3
        for l in reader():
            self.assertEqual(l[0].size, size)
            if l[1] > label:
                label = l[1]
            sum += 1
        return (sum, label)

    def test_train(self):
        if False:
            while True:
                i = 10
        (instances, max_label_value) = self.check_reader(paddle.dataset.flowers.train())
        self.assertEqual(instances, 6149)
        self.assertEqual(max_label_value, 102)

    def test_test(self):
        if False:
            for i in range(10):
                print('nop')
        (instances, max_label_value) = self.check_reader(paddle.dataset.flowers.test())
        self.assertEqual(instances, 1020)
        self.assertEqual(max_label_value, 102)

    def test_valid(self):
        if False:
            for i in range(10):
                print('nop')
        (instances, max_label_value) = self.check_reader(paddle.dataset.flowers.valid())
        self.assertEqual(instances, 1020)
        self.assertEqual(max_label_value, 102)
if __name__ == '__main__':
    unittest.main()