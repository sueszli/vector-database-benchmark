import os
import tempfile
import unittest
import numpy as np
from pyspark.sql import SparkSession
from pyspark.testing.connectutils import should_test_connect, connect_requirement_message
have_torch = True
try:
    import torch
except ImportError:
    have_torch = False
if should_test_connect:
    from pyspark.ml.connect.classification import LogisticRegression as LORV2, LogisticRegressionModel as LORV2Model
    import pandas as pd

class ClassificationTestsMixin:

    @staticmethod
    def _check_result(result_dataframe, expected_predictions, expected_probabilities=None):
        if False:
            return 10
        np.testing.assert_array_equal(list(result_dataframe.prediction), expected_predictions)
        if 'probability' in result_dataframe.columns:
            np.testing.assert_allclose(list(result_dataframe.probability), expected_probabilities, rtol=0.1)

    def test_binary_classes_logistic_regression(self):
        if False:
            while True:
                i = 10
        df1 = self.spark.createDataFrame([(1.0, [0.0, 5.0]), (0.0, [1.0, 2.0]), (1.0, [2.0, 1.0]), (0.0, [3.0, 3.0])] * 100, ['label', 'features'])
        eval_df1 = self.spark.createDataFrame([([0.0, 2.0],), ([3.5, 3.0],)], ['features'])
        lorv2 = LORV2(maxIter=200, numTrainWorkers=2, learningRate=0.001)
        assert lorv2.getMaxIter() == 200
        assert lorv2.getNumTrainWorkers() == 2
        assert lorv2.getOrDefault(lorv2.learningRate) == 0.001
        model = lorv2.fit(df1)
        assert model.uid == lorv2.uid
        expected_predictions = [1, 0]
        expected_probabilities = [[0.217875, 0.782125], [0.839615, 0.160385]]
        result = model.transform(eval_df1).toPandas()
        self._check_result(result, expected_predictions, expected_probabilities)
        pandas_eval_df1 = eval_df1.toPandas()
        pandas_eval_df1_copy = pandas_eval_df1.copy()
        local_transform_result = model.transform(pandas_eval_df1)
        pd.testing.assert_frame_equal(pandas_eval_df1, pandas_eval_df1_copy)
        self._check_result(local_transform_result, expected_predictions, expected_probabilities)
        model.set(model.probabilityCol, '')
        result_without_prob = model.transform(eval_df1).toPandas()
        assert 'probability' not in result_without_prob.columns
        self._check_result(result_without_prob, expected_predictions, None)

    def test_multi_classes_logistic_regression(self):
        if False:
            return 10
        df1 = self.spark.createDataFrame([(1.0, [1.0, 5.0]), (2.0, [1.0, -2.0]), (0.0, [-2.0, 1.5])] * 100, ['label', 'features'])
        eval_df1 = self.spark.createDataFrame([([1.5, 5.0],), ([1.0, -2.5],), ([-2.0, 1.0],)], ['features'])
        lorv2 = LORV2(maxIter=200, numTrainWorkers=2, learningRate=0.001)
        model = lorv2.fit(df1)
        expected_predictions = [1, 2, 0]
        expected_probabilities = [[0.005526459, 0.9943553, 0.0001183146], [0.004629959, 0.008141352, 0.9872288], [0.9624363, 0.03080821, 0.006755549]]
        result = model.transform(eval_df1).toPandas()
        self._check_result(result, expected_predictions, expected_probabilities)
        local_transform_result = model.transform(eval_df1.toPandas())
        self._check_result(local_transform_result, expected_predictions, expected_probabilities)

    def test_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_dir:
            estimator = LORV2(maxIter=2, numTrainWorkers=2, learningRate=0.001)
            local_path = os.path.join(tmp_dir, 'estimator')
            estimator.saveToLocal(local_path)
            loaded_estimator = LORV2.loadFromLocal(local_path)
            assert loaded_estimator.uid == estimator.uid
            assert loaded_estimator.getOrDefault(loaded_estimator.maxIter) == 2
            assert loaded_estimator.getOrDefault(loaded_estimator.numTrainWorkers) == 2
            assert loaded_estimator.getOrDefault(loaded_estimator.learningRate) == 0.001
            estimator2 = estimator.copy()
            estimator2.set(estimator2.maxIter, 10)
            estimator2.saveToLocal(local_path, overwrite=True)
            loaded_estimator2 = LORV2.loadFromLocal(local_path)
            assert loaded_estimator2.getOrDefault(loaded_estimator2.maxIter) == 10
            fs_path = os.path.join(tmp_dir, 'fs', 'estimator')
            estimator.save(fs_path)
            loaded_estimator = LORV2.load(fs_path)
            assert loaded_estimator.uid == estimator.uid
            assert loaded_estimator.getOrDefault(loaded_estimator.maxIter) == 2
            assert loaded_estimator.getOrDefault(loaded_estimator.numTrainWorkers) == 2
            assert loaded_estimator.getOrDefault(loaded_estimator.learningRate) == 0.001
            training_dataset = self.spark.createDataFrame([(1.0, [0.0, 5.0]), (0.0, [1.0, 2.0]), (1.0, [2.0, 1.0]), (0.0, [3.0, 3.0])] * 100, ['label', 'features'])
            eval_df1 = self.spark.createDataFrame([([0.0, 2.0],), ([3.5, 3.0],)], ['features'])
            model = estimator.fit(training_dataset)
            model_predictions = model.transform(eval_df1.toPandas())
            assert model.uid == estimator.uid
            local_model_path = os.path.join(tmp_dir, 'model')
            model.saveToLocal(local_model_path)
            lor_torch_model = torch.load(os.path.join(local_model_path, 'LogisticRegressionModel.torch'))
            with torch.inference_mode():
                torch_infer_result = lor_torch_model(torch.tensor(np.stack(list(eval_df1.toPandas().features)), dtype=torch.float32)).numpy()
            np.testing.assert_allclose(np.stack(list(model_predictions.probability)), torch_infer_result, rtol=0.0001)
            loaded_model = LORV2Model.loadFromLocal(local_model_path)
            assert loaded_model.numFeatures == 2
            assert loaded_model.numClasses == 2
            assert loaded_model.getOrDefault(loaded_model.maxIter) == 2
            assert loaded_model.torch_model is not None
            np.testing.assert_allclose(loaded_model.torch_model.weight.detach().numpy(), model.torch_model.weight.detach().numpy())
            np.testing.assert_allclose(loaded_model.torch_model.bias.detach().numpy(), model.torch_model.bias.detach().numpy())
            loaded_model.transform(eval_df1.toPandas())
            fs_model_path = os.path.join(tmp_dir, 'fs', 'model')
            model.save(fs_model_path)
            loaded_model = LORV2Model.load(fs_model_path)
            assert loaded_model.numFeatures == 2
            assert loaded_model.numClasses == 2
            assert loaded_model.getOrDefault(loaded_model.maxIter) == 2
            assert loaded_model.torch_model is not None
            loaded_model.transform(eval_df1.toPandas())

@unittest.skipIf(not should_test_connect or not have_torch, connect_requirement_message or 'No torch found')
class ClassificationTests(ClassificationTestsMixin, unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.spark = SparkSession.builder.master('local[2]').getOrCreate()

    def tearDown(self) -> None:
        if False:
            return 10
        self.spark.stop()
if __name__ == '__main__':
    from pyspark.ml.tests.connect.test_legacy_mode_classification import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)