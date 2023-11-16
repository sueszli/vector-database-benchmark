import unittest
import paddle

class TestCifar10(unittest.TestCase):

    def test_main(self):
        if False:
            return 10
        reader = paddle.dataset.cifar.train10(cycle=False)
        sample_num = 0
        for _ in reader():
            sample_num += 1
        cyclic_reader = paddle.dataset.cifar.train10(cycle=True)
        read_num = 0
        for data in cyclic_reader():
            read_num += 1
            self.assertEqual(len(data), 2)
            if read_num == sample_num * 2:
                break
        self.assertEqual(read_num, sample_num * 2)
if __name__ == '__main__':
    unittest.main()