import unittest
import paddle.dataset.cifar
__all__ = []

class TestCIFAR(unittest.TestCase):

    def check_reader(self, reader):
        if False:
            return 10
        sum = 0
        label = 0
        for l in reader():
            self.assertEqual(l[0].size, 3072)
            if l[1] > label:
                label = l[1]
            sum += 1
        return (sum, label)

    def test_test10(self):
        if False:
            while True:
                i = 10
        (instances, max_label_value) = self.check_reader(paddle.dataset.cifar.test10())
        self.assertEqual(instances, 10000)
        self.assertEqual(max_label_value, 9)

    def test_train10(self):
        if False:
            return 10
        (instances, max_label_value) = self.check_reader(paddle.dataset.cifar.train10())
        self.assertEqual(instances, 50000)
        self.assertEqual(max_label_value, 9)

    def test_test100(self):
        if False:
            print('Hello World!')
        (instances, max_label_value) = self.check_reader(paddle.dataset.cifar.test100())
        self.assertEqual(instances, 10000)
        self.assertEqual(max_label_value, 99)

    def test_train100(self):
        if False:
            i = 10
            return i + 15
        (instances, max_label_value) = self.check_reader(paddle.dataset.cifar.train100())
        self.assertEqual(instances, 50000)
        self.assertEqual(max_label_value, 99)
if __name__ == '__main__':
    unittest.main()