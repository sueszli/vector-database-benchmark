import tempfile
import unittest
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel
from pyspark.testing.mlutils import DummyLogisticRegression, SparkSessionTestCase
from pyspark.ml.tests.tuning.test_tuning import ValidatorTestUtilsMixin

class TrainValidationSplitIONestedTests(SparkSessionTestCase, ValidatorTestUtilsMixin):

    def _run_test_save_load_pipeline_estimator(self, LogisticRegressionCls):
        if False:
            print('Hello World!')
        temp_path = tempfile.mkdtemp()
        training = self.spark.createDataFrame([(0, 'a b c d e spark', 1.0), (1, 'b d', 0.0), (2, 'spark f g h', 1.0), (3, 'hadoop mapreduce', 0.0), (4, 'b spark who', 1.0), (5, 'g d a y', 0.0), (6, 'spark fly', 1.0), (7, 'was mapreduce', 0.0)], ['id', 'text', 'label'])
        tokenizer = Tokenizer(inputCol='text', outputCol='words')
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='features')
        ova = OneVsRest(classifier=LogisticRegressionCls())
        lr1 = LogisticRegressionCls().setMaxIter(5)
        lr2 = LogisticRegressionCls().setMaxIter(10)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, ova])
        paramGrid = ParamGridBuilder().addGrid(hashingTF.numFeatures, [10, 100]).addGrid(ova.classifier, [lr1, lr2]).build()
        tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=MulticlassClassificationEvaluator())
        tvsPath = temp_path + '/tvs'
        tvs.save(tvsPath)
        loadedTvs = TrainValidationSplit.load(tvsPath)
        self.assert_param_maps_equal(loadedTvs.getEstimatorParamMaps(), paramGrid)
        self.assertEqual(loadedTvs.getEstimator().uid, tvs.getEstimator().uid)
        tvsModel = tvs.fit(training)
        tvsModelPath = temp_path + '/tvsModel'
        tvsModel.save(tvsModelPath)
        loadedModel = TrainValidationSplitModel.load(tvsModelPath)
        self.assertEqual(loadedModel.bestModel.uid, tvsModel.bestModel.uid)
        self.assertEqual(len(loadedModel.bestModel.stages), len(tvsModel.bestModel.stages))
        for (loadedStage, originalStage) in zip(loadedModel.bestModel.stages, tvsModel.bestModel.stages):
            self.assertEqual(loadedStage.uid, originalStage.uid)
        nested_pipeline = Pipeline(stages=[tokenizer, Pipeline(stages=[hashingTF, ova])])
        tvs2 = TrainValidationSplit(estimator=nested_pipeline, estimatorParamMaps=paramGrid, evaluator=MulticlassClassificationEvaluator())
        tvs2Path = temp_path + '/tvs2'
        tvs2.save(tvs2Path)
        loadedTvs2 = TrainValidationSplit.load(tvs2Path)
        self.assert_param_maps_equal(loadedTvs2.getEstimatorParamMaps(), paramGrid)
        self.assertEqual(loadedTvs2.getEstimator().uid, tvs2.getEstimator().uid)
        tvsModel2 = tvs2.fit(training)
        tvsModelPath2 = temp_path + '/tvsModel2'
        tvsModel2.save(tvsModelPath2)
        loadedModel2 = TrainValidationSplitModel.load(tvsModelPath2)
        self.assertEqual(loadedModel2.bestModel.uid, tvsModel2.bestModel.uid)
        loaded_nested_pipeline_model = loadedModel2.bestModel.stages[1]
        original_nested_pipeline_model = tvsModel2.bestModel.stages[1]
        self.assertEqual(loaded_nested_pipeline_model.uid, original_nested_pipeline_model.uid)
        self.assertEqual(len(loaded_nested_pipeline_model.stages), len(original_nested_pipeline_model.stages))
        for (loadedStage, originalStage) in zip(loaded_nested_pipeline_model.stages, original_nested_pipeline_model.stages):
            self.assertEqual(loadedStage.uid, originalStage.uid)

    def test_save_load_pipeline_estimator(self):
        if False:
            i = 10
            return i + 15
        self._run_test_save_load_pipeline_estimator(LogisticRegression)
        self._run_test_save_load_pipeline_estimator(DummyLogisticRegression)
if __name__ == '__main__':
    from pyspark.ml.tests.tuning.test_tvs_io_pipeline import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)