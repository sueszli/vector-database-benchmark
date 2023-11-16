import tempfile
import unittest
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.testing.mlutils import DummyLogisticRegression, SparkSessionTestCase
from pyspark.ml.tests.tuning.test_tuning import ValidatorTestUtilsMixin

class CrossValidatorIONestedTests(SparkSessionTestCase, ValidatorTestUtilsMixin):

    def _run_test_save_load_nested_estimator(self, LogisticRegressionCls):
        if False:
            while True:
                i = 10
        temp_path = tempfile.mkdtemp()
        dataset = self.spark.createDataFrame([(Vectors.dense([0.0]), 0.0), (Vectors.dense([0.4]), 1.0), (Vectors.dense([0.5]), 0.0), (Vectors.dense([0.6]), 1.0), (Vectors.dense([1.0]), 1.0)] * 10, ['features', 'label'])
        ova = OneVsRest(classifier=LogisticRegressionCls())
        lr1 = LogisticRegressionCls().setMaxIter(100)
        lr2 = LogisticRegressionCls().setMaxIter(150)
        grid = ParamGridBuilder().addGrid(ova.classifier, [lr1, lr2]).build()
        evaluator = MulticlassClassificationEvaluator()
        cv = CrossValidator(estimator=ova, estimatorParamMaps=grid, evaluator=evaluator)
        cvModel = cv.fit(dataset)
        cvPath = temp_path + '/cv'
        cv.save(cvPath)
        loadedCV = CrossValidator.load(cvPath)
        self.assert_param_maps_equal(loadedCV.getEstimatorParamMaps(), grid)
        self.assertEqual(loadedCV.getEstimator().uid, cv.getEstimator().uid)
        self.assertEqual(loadedCV.getEvaluator().uid, cv.getEvaluator().uid)
        originalParamMap = cv.getEstimatorParamMaps()
        loadedParamMap = loadedCV.getEstimatorParamMaps()
        for (i, param) in enumerate(loadedParamMap):
            for p in param:
                if p.name == 'classifier':
                    self.assertEqual(param[p].uid, originalParamMap[i][p].uid)
                else:
                    self.assertEqual(param[p], originalParamMap[i][p])
        cvModelPath = temp_path + '/cvModel'
        cvModel.save(cvModelPath)
        loadedModel = CrossValidatorModel.load(cvModelPath)
        self.assert_param_maps_equal(loadedModel.getEstimatorParamMaps(), grid)
        self.assertEqual(loadedModel.bestModel.uid, cvModel.bestModel.uid)

    def test_save_load_nested_estimator(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test_save_load_nested_estimator(LogisticRegression)
        self._run_test_save_load_nested_estimator(DummyLogisticRegression)
if __name__ == '__main__':
    from pyspark.ml.tests.tuning.test_cv_io_nested import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)