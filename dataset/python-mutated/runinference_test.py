import unittest
from io import StringIO
import mock
import pytest
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import runinference_sklearn_keyed_model_handler
from . import runinference_sklearn_unkeyed_model_handler
try:
    import torch
    from . import runinference
except ImportError:
    raise unittest.SkipTest('PyTorch dependencies are not installed')
try:
    from apache_beam.io.gcp.gcsfilesystem import GCSFileSystem
except ImportError:
    raise unittest.SkipTest('GCP dependencies are not installed')

def check_torch_keyed_model_handler():
    if False:
        return 10
    expected = "[START torch_keyed_model_handler]\n('first_question', PredictionResult(example=tensor([105.]), inference=tensor([523.6982]), model_id='gs://apache-beam-samples/run_inference/five_times_table_torch.pt'))\n('second_question', PredictionResult(example=tensor([108.]), inference=tensor([538.5867]), model_id='gs://apache-beam-samples/run_inference/five_times_table_torch.pt'))\n('third_question', PredictionResult(example=tensor([1000.]), inference=tensor([4965.4019]), model_id='gs://apache-beam-samples/run_inference/five_times_table_torch.pt'))\n('fourth_question', PredictionResult(example=tensor([1013.]), inference=tensor([5029.9180]), model_id='gs://apache-beam-samples/run_inference/five_times_table_torch.pt'))\n[END torch_keyed_model_handler] ".splitlines()[1:-1]
    return expected

def check_sklearn_keyed_model_handler(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START sklearn_keyed_model_handler]\n('first_question', PredictionResult(example=[105.0], inference=array([525.]), model_id='gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl'))\n('second_question', PredictionResult(example=[108.0], inference=array([540.]), model_id='gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl'))\n('third_question', PredictionResult(example=[1000.0], inference=array([5000.]), model_id='gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl'))\n('fourth_question', PredictionResult(example=[1013.0], inference=array([5065.]), model_id='gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl'))\n[END sklearn_keyed_model_handler] ".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_torch_unkeyed_model_handler():
    if False:
        print('Hello World!')
    expected = "[START torch_unkeyed_model_handler]\nPredictionResult(example=tensor([10.]), inference=tensor([52.2325]), model_id='gs://apache-beam-samples/run_inference/five_times_table_torch.pt')\nPredictionResult(example=tensor([40.]), inference=tensor([201.1165]), model_id='gs://apache-beam-samples/run_inference/five_times_table_torch.pt')\nPredictionResult(example=tensor([60.]), inference=tensor([300.3724]), model_id='gs://apache-beam-samples/run_inference/five_times_table_torch.pt')\nPredictionResult(example=tensor([90.]), inference=tensor([449.2563]), model_id='gs://apache-beam-samples/run_inference/five_times_table_torch.pt')\n[END torch_unkeyed_model_handler] ".splitlines()[1:-1]
    return expected

def check_sklearn_unkeyed_model_handler(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START sklearn_unkeyed_model_handler]\nPredictionResult(example=array([20.], dtype=float32), inference=array([100.], dtype=float32), model_id='gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl')\nPredictionResult(example=array([40.], dtype=float32), inference=array([200.], dtype=float32), model_id='gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl')\nPredictionResult(example=array([60.], dtype=float32), inference=array([300.], dtype=float32), model_id='gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl')\nPredictionResult(example=array([90.], dtype=float32), inference=array([450.], dtype=float32), model_id='gs://apache-beam-samples/run_inference/five_times_table_sklearn.pkl')\n[END sklearn_unkeyed_model_handler]  ".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.runinference_sklearn_unkeyed_model_handler.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.runinference_sklearn_keyed_model_handler.print', str)
class RunInferenceTest(unittest.TestCase):

    def test_sklearn_unkeyed_model_handler(self):
        if False:
            while True:
                i = 10
        runinference_sklearn_unkeyed_model_handler.sklearn_unkeyed_model_handler(check_sklearn_unkeyed_model_handler)

    def test_sklearn_keyed_model_handler(self):
        if False:
            i = 10
            return i + 15
        runinference_sklearn_keyed_model_handler.sklearn_keyed_model_handler(check_sklearn_keyed_model_handler)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('sys.stdout', new_callable=StringIO)
class RunInferenceStdoutTest(unittest.TestCase):

    @pytest.mark.uses_pytorch
    def test_check_torch_keyed_model_handler(self, mock_stdout):
        if False:
            i = 10
            return i + 15
        runinference.torch_keyed_model_handler()
        predicted = mock_stdout.getvalue().splitlines()
        expected = check_torch_keyed_model_handler()
        self.assertEqual(predicted, expected)

    @pytest.mark.uses_pytorch
    def test_check_torch_unkeyed_model_handler(self, mock_stdout):
        if False:
            for i in range(10):
                print('nop')
        runinference.torch_unkeyed_model_handler()
        predicted = mock_stdout.getvalue().splitlines()
        expected = check_torch_unkeyed_model_handler()
        self.assertEqual(predicted, expected)
if __name__ == '__main__':
    unittest.main()