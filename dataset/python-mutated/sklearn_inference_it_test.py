"""End-to-End test for Sklearn Inference"""
import logging
import re
import sys
import unittest
import uuid
import pytest
from apache_beam.examples.inference import sklearn_japanese_housing_regression
from apache_beam.examples.inference import sklearn_mnist_classification
from apache_beam.io.filesystems import FileSystems
from apache_beam.testing.test_pipeline import TestPipeline
try:
    from apache_beam.io.gcp.gcsfilesystem import GCSFileSystem
except ImportError:
    raise unittest.SkipTest('GCP dependencies are not installed')

def process_outputs(filepath):
    if False:
        i = 10
        return i + 15
    with FileSystems().open(filepath) as f:
        lines = f.readlines()
    lines = [l.decode('utf-8').strip('\n') for l in lines]
    return lines

def file_lines_sorted(filepath):
    if False:
        print('Hello World!')
    with FileSystems().open(filepath) as f:
        lines = f.readlines()
    lines = [l.decode('utf-8').strip('\n') for l in lines]
    return sorted(lines)

@pytest.mark.uses_sklearn
@pytest.mark.it_postcommit
class SklearnInference(unittest.TestCase):

    def test_sklearn_mnist_classification(self):
        if False:
            while True:
                i = 10
        test_pipeline = TestPipeline(is_integration_test=True)
        input_file = 'gs://apache-beam-ml/testing/inputs/it_mnist_data.csv'
        output_file_dir = 'gs://temp-storage-for-end-to-end-tests'
        output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])
        model_path = 'gs://apache-beam-ml/models/mnist_model_svm.pickle'
        extra_opts = {'input': input_file, 'output': output_file, 'model_path': model_path}
        sklearn_mnist_classification.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        self.assertEqual(FileSystems().exists(output_file), True)
        expected_output_filepath = 'gs://apache-beam-ml/testing/expected_outputs/test_sklearn_mnist_classification_actuals.txt'
        expected_outputs = process_outputs(expected_output_filepath)
        predicted_outputs = process_outputs(output_file)
        self.assertEqual(len(expected_outputs), len(predicted_outputs))
        predictions_dict = {}
        for i in range(len(predicted_outputs)):
            (true_label, prediction) = predicted_outputs[i].split(',')
            predictions_dict[true_label] = prediction
        for i in range(len(expected_outputs)):
            (true_label, expected_prediction) = expected_outputs[i].split(',')
            self.assertEqual(predictions_dict[true_label], expected_prediction)

    def test_sklearn_mnist_classification_large_model(self):
        if False:
            i = 10
            return i + 15
        test_pipeline = TestPipeline(is_integration_test=True)
        input_file = 'gs://apache-beam-ml/testing/inputs/it_mnist_data.csv'
        output_file_dir = 'gs://temp-storage-for-end-to-end-tests'
        output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])
        model_path = 'gs://apache-beam-ml/models/mnist_model_svm.pickle'
        extra_opts = {'input': input_file, 'output': output_file, 'model_path': model_path, 'large_model': True}
        sklearn_mnist_classification.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        self.assertEqual(FileSystems().exists(output_file), True)
        expected_output_filepath = 'gs://apache-beam-ml/testing/expected_outputs/test_sklearn_mnist_classification_actuals.txt'
        expected_outputs = process_outputs(expected_output_filepath)
        predicted_outputs = process_outputs(output_file)
        self.assertEqual(len(expected_outputs), len(predicted_outputs))
        predictions_dict = {}
        for i in range(len(predicted_outputs)):
            (true_label, prediction) = predicted_outputs[i].split(',')
            predictions_dict[true_label] = prediction
        for i in range(len(expected_outputs)):
            (true_label, expected_prediction) = expected_outputs[i].split(',')
            self.assertEqual(predictions_dict[true_label], expected_prediction)

    @unittest.skipIf(sys.version_info >= (3, 11, 0), 'Beam#27151')
    def test_sklearn_regression(self):
        if False:
            for i in range(10):
                print('nop')
        test_pipeline = TestPipeline(is_integration_test=True)
        input_file = 'gs://apache-beam-ml/testing/inputs/japanese_housing_test_data.csv'
        output_file_dir = 'gs://temp-storage-for-end-to-end-tests'
        output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])
        model_path = 'gs://apache-beam-ml/models/japanese_housing/'
        extra_opts = {'input': input_file, 'output': output_file, 'model_path': model_path}
        sklearn_japanese_housing_regression.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        self.assertEqual(FileSystems().exists(output_file), True)
        expected_output_filepath = 'gs://apache-beam-ml/testing/expected_outputs/japanese_housing_subset.txt'
        expected_outputs = file_lines_sorted(expected_output_filepath)
        actual_outputs = file_lines_sorted(output_file)
        self.assertEqual(len(expected_outputs), len(actual_outputs))
        for (expected, actual) in zip(expected_outputs, actual_outputs):
            (expected_true, expected_predict) = re.findall('\\d+', expected)
            (actual_true, actual_predict) = re.findall('\\d+', actual)
            self.assertEqual(actual_true, expected_true)
            percent_diff = abs(float(expected_predict) - float(actual_predict)) / float(expected_predict) * 100.0
            self.assertLess(percent_diff, 10)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()