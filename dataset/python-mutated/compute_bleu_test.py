"""Test functions in compute_blue.py."""
import tempfile
import tensorflow as tf
from official.transformer import compute_bleu

class ComputeBleuTest(tf.test.TestCase):

    def _create_temp_file(self, text):
        if False:
            return 10
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with tf.io.gfile.GFile(temp_file.name, 'w') as w:
            w.write(text)
        return temp_file.name

    def test_bleu_same(self):
        if False:
            while True:
                i = 10
        ref = self._create_temp_file('test 1 two 3\nmore tests!')
        hyp = self._create_temp_file('test 1 two 3\nmore tests!')
        uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
        cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
        self.assertEqual(100, uncased_score)
        self.assertEqual(100, cased_score)

    def test_bleu_same_different_case(self):
        if False:
            print('Hello World!')
        ref = self._create_temp_file('Test 1 two 3\nmore tests!')
        hyp = self._create_temp_file('test 1 two 3\nMore tests!')
        uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
        cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
        self.assertEqual(100, uncased_score)
        self.assertLess(cased_score, 100)

    def test_bleu_different(self):
        if False:
            print('Hello World!')
        ref = self._create_temp_file('Testing\nmore tests!')
        hyp = self._create_temp_file('Dog\nCat')
        uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
        cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
        self.assertLess(uncased_score, 100)
        self.assertLess(cased_score, 100)

    def test_bleu_tokenize(self):
        if False:
            return 10
        s = 'Test0, 1 two, 3'
        tokenized = compute_bleu.bleu_tokenize(s)
        self.assertEqual(['Test0', ',', '1', 'two', ',', '3'], tokenized)
if __name__ == '__main__':
    tf.test.main()