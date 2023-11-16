import unittest
import paddle.dataset.wmt16
__all__ = []

class TestWMT16(unittest.TestCase):

    def checkout_one_sample(self, sample):
        if False:
            print('Hello World!')
        self.assertEqual(len(sample), 3)
        self.assertEqual(sample[0][0], 0)
        self.assertEqual(sample[0][-1], 1)
        self.assertEqual(sample[1][0], 0)
        self.assertEqual(sample[2][-1], 1)

    def test_train(self):
        if False:
            i = 10
            return i + 15
        for (idx, sample) in enumerate(paddle.dataset.wmt16.train(src_dict_size=100000, trg_dict_size=100000)()):
            if idx >= 10:
                break
            self.checkout_one_sample(sample)

    def test_test(self):
        if False:
            return 10
        for (idx, sample) in enumerate(paddle.dataset.wmt16.test(src_dict_size=1000, trg_dict_size=1000)()):
            if idx >= 10:
                break
            self.checkout_one_sample(sample)

    def test_val(self):
        if False:
            for i in range(10):
                print('nop')
        for (idx, sample) in enumerate(paddle.dataset.wmt16.validation(src_dict_size=1000, trg_dict_size=1000)()):
            if idx >= 10:
                break
            self.checkout_one_sample(sample)

    def test_get_dict(self):
        if False:
            return 10
        dict_size = 1000
        word_dict = paddle.dataset.wmt16.get_dict('en', dict_size, True)
        self.assertEqual(len(word_dict), dict_size)
        self.assertEqual(word_dict[0], '<s>')
        self.assertEqual(word_dict[1], '<e>')
        self.assertEqual(word_dict[2], '<unk>')
if __name__ == '__main__':
    unittest.main()