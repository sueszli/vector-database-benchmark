import unittest
from io import StringIO
import mock
from apache_beam.testing.test_pipeline import TestPipeline
try:
    import tensorflow_transform as tft
    from apache_beam.examples.snippets.transforms.elementwise.mltransform import mltransform_scale_to_0_1
    from apache_beam.examples.snippets.transforms.elementwise.mltransform import mltransform_compute_and_apply_vocabulary
    from apache_beam.examples.snippets.transforms.elementwise.mltransform import mltransform_compute_and_apply_vocabulary_with_scalar
except ImportError:
    raise unittest.SkipTest('tensorflow_transform is not installed.')

def check_mltransform_compute_and_apply_vocab():
    if False:
        for i in range(10):
            print('nop')
    expected = '[START mltransform_compute_and_apply_vocab]\nRow(x=array([4, 1, 0]))\nRow(x=array([0, 2, 3]))\n  [END mltransform_compute_and_apply_vocab] '.splitlines()[1:-1]
    return expected

def check_mltransform_scale_to_0_1():
    if False:
        i = 10
        return i + 15
    expected = '[START mltransform_scale_to_0_1]\nRow(x=array([0.       , 0.5714286, 0.2857143], dtype=float32))\nRow(x=array([0.42857143, 0.14285715, 1.        ], dtype=float32))\n  [END mltransform_scale_to_0_1] '.splitlines()[1:-1]
    return expected

def check_mltransform_compute_and_apply_vocabulary_with_scalar():
    if False:
        for i in range(10):
            print('nop')
    expected = '[START mltransform_compute_and_apply_vocabulary_with_scalar]\nRow(x=array([4]))\nRow(x=array([1]))\nRow(x=array([0]))\nRow(x=array([2]))\nRow(x=array([3]))\n  [END mltransform_compute_and_apply_vocabulary_with_scalar] '.splitlines()[1:-1]
    return expected

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('sys.stdout', new_callable=StringIO)
class MLTransformStdOutTest(unittest.TestCase):

    def test_mltransform_compute_and_apply_vocab(self, mock_stdout):
        if False:
            i = 10
            return i + 15
        mltransform_compute_and_apply_vocabulary()
        predicted = mock_stdout.getvalue().splitlines()
        expected = check_mltransform_compute_and_apply_vocab()
        self.assertEqual(predicted, expected)

    def test_mltransform_scale_to_0_1(self, mock_stdout):
        if False:
            while True:
                i = 10
        mltransform_scale_to_0_1()
        predicted = mock_stdout.getvalue().splitlines()
        expected = check_mltransform_scale_to_0_1()
        self.assertEqual(predicted, expected)

    def test_mltransform_compute_and_apply_vocab_scalar(self, mock_stdout):
        if False:
            return 10
        mltransform_compute_and_apply_vocabulary_with_scalar()
        predicted = mock_stdout.getvalue().splitlines()
        expected = check_mltransform_compute_and_apply_vocabulary_with_scalar()
        self.assertEqual(predicted, expected)
if __name__ == '__main__':
    unittest.main()