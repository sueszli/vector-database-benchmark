import tempfile
import unittest
import numpy as np
from pyspark.ml.param import Param, Params
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.testing.connectutils import should_test_connect, connect_requirement_message
have_sklearn = True
try:
    from sklearn.datasets import load_breast_cancer
except ImportError:
    have_sklearn = False
if should_test_connect:
    import pandas as pd
    from pyspark.ml.connect import Model, Estimator
    from pyspark.ml.connect.feature import StandardScaler
    from pyspark.ml.connect.classification import LogisticRegression as LORV2
    from pyspark.ml.connect.pipeline import Pipeline
    from pyspark.ml.connect.tuning import CrossValidator, CrossValidatorModel
    from pyspark.ml.connect.evaluation import BinaryClassificationEvaluator, RegressionEvaluator

class HasInducedError(Params):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(HasInducedError, self).__init__()
        self.inducedError = Param(self, 'inducedError', 'Uniformly-distributed error added to feature')

    def getInducedError(self):
        if False:
            print('Hello World!')
        return self.getOrDefault(self.inducedError)

class InducedErrorModel(Model, HasInducedError):

    def __init__(self):
        if False:
            print('Hello World!')
        super(InducedErrorModel, self).__init__()

    def _transform(self, dataset):
        if False:
            while True:
                i = 10
        return dataset.withColumn('prediction', dataset.feature + rand(0) * self.getInducedError())

class InducedErrorEstimator(Estimator, HasInducedError):

    def __init__(self, inducedError=1.0):
        if False:
            while True:
                i = 10
        super(InducedErrorEstimator, self).__init__()
        self._set(inducedError=inducedError)

    def _fit(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        model = InducedErrorModel()
        self._copyValues(model)
        return model

class CrossValidatorTestsMixin:

    def test_gen_avg_and_std_metrics(self):
        if False:
            return 10
        metrics_all = [[1.0, 3.0, 2.0, 4.0], [3.0, 2.0, 2.0, 4.0], [3.0, 2.5, 2.1, 8.0]]
        (avg_metrics, std_metrics) = CrossValidator._gen_avg_and_std_metrics(metrics_all)
        assert np.allclose(avg_metrics, [2.33333333, 2.5, 2.03333333, 5.33333333])
        assert np.allclose(std_metrics, [0.94280904, 0.40824829, 0.04714045, 1.88561808])
        assert isinstance(avg_metrics, list)
        assert isinstance(std_metrics, list)

    def test_copy(self):
        if False:
            return 10
        dataset = self.spark.createDataFrame([(10, 10.0), (50, 50.0), (100, 100.0), (500, 500.0)] * 10, ['feature', 'label'])
        iee = InducedErrorEstimator()
        evaluator = RegressionEvaluator(metricName='rmse')
        grid = ParamGridBuilder().addGrid(iee.inducedError, [100.0, 0.0, 10000.0]).build()
        cv = CrossValidator(estimator=iee, estimatorParamMaps=grid, evaluator=evaluator, numFolds=2)
        cvCopied = cv.copy()
        for param in [lambda x: x.getEstimator().uid, lambda x: x.getNumFolds(), lambda x: x.getFoldCol(), lambda x: x.getParallelism(), lambda x: x.getSeed()]:
            self.assertEqual(param(cv), param(cvCopied))
        cvModel = cv.fit(dataset)
        cvModelCopied = cvModel.copy()
        for index in range(len(cvModel.avgMetrics)):
            self.assertTrue(abs(cvModel.avgMetrics[index] - cvModelCopied.avgMetrics[index]) < 0.0001)
        self.assertTrue(np.allclose(cvModel.stdMetrics, cvModelCopied.stdMetrics))
        for param in [lambda x: x.getNumFolds(), lambda x: x.getFoldCol(), lambda x: x.getSeed()]:
            self.assertEqual(param(cvModel), param(cvModelCopied))
        cvModel.avgMetrics[0] = 'foo'
        self.assertNotEqual(cvModelCopied.avgMetrics[0], 'foo', 'Changing the original avgMetrics should not affect the copied model')
        cvModel.stdMetrics[0] = 'foo'
        self.assertNotEqual(cvModelCopied.stdMetrics[0], 'foo', 'Changing the original stdMetrics should not affect the copied model')

    def test_fit_minimize_metric(self):
        if False:
            return 10
        dataset = self.spark.createDataFrame([(10, 10.0), (50, 50.0), (100, 100.0), (500, 500.0)] * 10, ['feature', 'label'])
        iee = InducedErrorEstimator()
        evaluator = RegressionEvaluator(metricName='rmse')
        grid = ParamGridBuilder().addGrid(iee.inducedError, [100.0, 0.0, 10000.0]).build()
        cv = CrossValidator(estimator=iee, estimatorParamMaps=grid, evaluator=evaluator)
        cvModel = cv.fit(dataset)
        bestModel = cvModel.bestModel
        bestModelMetric = evaluator.evaluate(bestModel.transform(dataset))
        self.assertEqual(0.0, bestModel.getOrDefault('inducedError'), 'Best model should have zero induced error')
        self.assertEqual(0.0, bestModelMetric, 'Best model has RMSE of 0')

    def test_fit_maximize_metric(self):
        if False:
            return 10
        dataset = self.spark.createDataFrame([(10, 10.0), (50, 50.0), (100, 100.0), (500, 500.0)] * 10, ['feature', 'label'])
        iee = InducedErrorEstimator()
        evaluator = RegressionEvaluator(metricName='r2')
        grid = ParamGridBuilder().addGrid(iee.inducedError, [100.0, 0.0, 10000.0]).build()
        cv = CrossValidator(estimator=iee, estimatorParamMaps=grid, evaluator=evaluator)
        cvModel = cv.fit(dataset)
        bestModel = cvModel.bestModel
        bestModelMetric = evaluator.evaluate(bestModel.transform(dataset))
        self.assertEqual(0.0, bestModel.getOrDefault('inducedError'), 'Best model should have zero induced error')
        self.assertEqual(1.0, bestModelMetric, 'Best model has R-squared of 1')

    @staticmethod
    def _check_result(result_dataframe, expected_predictions, expected_probabilities=None):
        if False:
            while True:
                i = 10
        np.testing.assert_array_equal(list(result_dataframe.prediction), expected_predictions)
        if 'probability' in result_dataframe.columns:
            np.testing.assert_allclose(list(result_dataframe.probability), expected_probabilities, rtol=0.1)

    def test_crossvalidator_on_pipeline(self):
        if False:
            print('Hello World!')
        sk_dataset = load_breast_cancer()
        train_dataset = self.spark.createDataFrame(zip(sk_dataset.data.tolist(), [int(t) for t in sk_dataset.target]), schema='features: array<double>, label: long')
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
        lorv2 = LORV2(numTrainWorkers=2, featuresCol='scaled_features')
        pipeline = Pipeline(stages=[scaler, lorv2])
        grid2 = ParamGridBuilder().addGrid(lorv2.maxIter, [2, 200]).build()
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid2, parallelism=2, evaluator=BinaryClassificationEvaluator())
        cv_model = cv.fit(train_dataset)
        transformed_result = cv_model.transform(train_dataset).select('prediction', 'probability').toPandas()
        expected_transformed_result = cv_model.bestModel.transform(train_dataset).select('prediction', 'probability').toPandas()
        pd.testing.assert_frame_equal(transformed_result, expected_transformed_result)
        assert cv_model.bestModel.stages[1].getMaxIter() == 200
        assert cv_model.avgMetrics[1] > cv_model.avgMetrics[0]

        def _verify_cv_saved_params(instance, loaded_instance):
            if False:
                i = 10
                return i + 15
            assert instance.getEstimator().uid == loaded_instance.getEstimator().uid
            assert instance.getEvaluator().uid == loaded_instance.getEvaluator().uid
            assert instance.getEstimatorParamMaps() == loaded_instance.getEstimatorParamMaps()
        with tempfile.TemporaryDirectory() as tmp_dir:
            cv.saveToLocal(f'{tmp_dir}/cv')
            loaded_cv = CrossValidator.loadFromLocal(f'{tmp_dir}/cv')
            _verify_cv_saved_params(cv, loaded_cv)
            cv_model.saveToLocal(f'{tmp_dir}/cv_model')
            loaded_cv_model = CrossValidatorModel.loadFromLocal(f'{tmp_dir}/cv_model')
            _verify_cv_saved_params(cv_model, loaded_cv_model)
            assert cv_model.uid == loaded_cv_model.uid
            assert cv_model.bestModel.uid == loaded_cv_model.bestModel.uid
            assert cv_model.bestModel.stages[0].uid == loaded_cv_model.bestModel.stages[0].uid
            assert cv_model.bestModel.stages[1].uid == loaded_cv_model.bestModel.stages[1].uid
            assert loaded_cv_model.bestModel.stages[1].getMaxIter() == 200
            np.testing.assert_allclose(cv_model.avgMetrics, loaded_cv_model.avgMetrics)
            np.testing.assert_allclose(cv_model.stdMetrics, loaded_cv_model.stdMetrics)

    def test_crossvalidator_with_fold_col(self):
        if False:
            i = 10
            return i + 15
        sk_dataset = load_breast_cancer()
        train_dataset = self.spark.createDataFrame(zip(sk_dataset.data.tolist(), [int(t) for t in sk_dataset.target], [int(i % 3) for i in range(len(sk_dataset.target))]), schema='features: array<double>, label: long, fold: long')
        lorv2 = LORV2(numTrainWorkers=2)
        grid2 = ParamGridBuilder().addGrid(lorv2.maxIter, [2, 200]).build()
        cv = CrossValidator(estimator=lorv2, estimatorParamMaps=grid2, parallelism=2, evaluator=BinaryClassificationEvaluator(), foldCol='fold', numFolds=3)
        cv.fit(train_dataset)

@unittest.skipIf(not should_test_connect or not have_sklearn, connect_requirement_message or 'No sklearn found')
class CrossValidatorTests(CrossValidatorTestsMixin, unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        self.spark = SparkSession.builder.master('local[2]').getOrCreate()

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        self.spark.stop()
if __name__ == '__main__':
    from pyspark.ml.tests.connect.test_legacy_mode_tuning import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)