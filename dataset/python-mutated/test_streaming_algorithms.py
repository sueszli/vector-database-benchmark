import os
import unittest
from numpy import array, random, exp, dot, all, mean, abs
from numpy import sum as array_sum
from pyspark import SparkContext
from pyspark.mllib.clustering import StreamingKMeans, StreamingKMeansModel
from pyspark.mllib.classification import StreamingLogisticRegressionWithSGD
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint, StreamingLinearRegressionWithSGD
from pyspark.mllib.util import LinearDataGenerator
from pyspark.streaming import StreamingContext
from pyspark.testing.utils import eventually

class MLLibStreamingTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.sc = SparkContext('local[4]', 'MLlib tests')
        self.ssc = StreamingContext(self.sc, 1.0)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.ssc.stop(False)
        self.sc.stop()

class StreamingKMeansTest(MLLibStreamingTestCase):

    def test_model_params(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the model params are set correctly'
        stkm = StreamingKMeans()
        stkm.setK(5).setDecayFactor(0.0)
        self.assertEqual(stkm._k, 5)
        self.assertEqual(stkm._decayFactor, 0.0)
        self.assertIsNone(stkm.latestModel())
        self.assertRaises(ValueError, stkm.trainOn, [0.0, 1.0])
        stkm.setInitialCenters(centers=[[0.0, 0.0], [1.0, 1.0]], weights=[1.0, 1.0])
        self.assertEqual(stkm.latestModel().centers, [[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(stkm.latestModel().clusterWeights, [1.0, 1.0])

    def test_accuracy_for_single_center(self):
        if False:
            i = 10
            return i + 15
        'Test that parameters obtained are correct for a single center.'
        (centers, batches) = self.streamingKMeansDataGenerator(batches=5, numPoints=5, k=1, d=5, r=0.1, seed=0)
        stkm = StreamingKMeans(1)
        stkm.setInitialCenters([[0.0, 0.0, 0.0, 0.0, 0.0]], [0.0])
        input_stream = self.ssc.queueStream([self.sc.parallelize(batch, 1) for batch in batches])
        stkm.trainOn(input_stream)
        self.ssc.start()

        def condition():
            if False:
                return 10
            self.assertEqual(stkm.latestModel().clusterWeights, [25.0])
            return True
        eventually(catch_assertions=True)(condition)()
        realCenters = array_sum(array(centers), axis=0)
        for i in range(5):
            modelCenters = stkm.latestModel().centers[0][i]
            self.assertAlmostEqual(centers[0][i], modelCenters, 1)
            self.assertAlmostEqual(realCenters[i], modelCenters, 1)

    def streamingKMeansDataGenerator(self, batches, numPoints, k, d, r, seed, centers=None):
        if False:
            i = 10
            return i + 15
        rng = random.RandomState(seed)
        centers = [rng.randn(d) for i in range(k)]
        return (centers, [[Vectors.dense(centers[j % k] + r * rng.randn(d)) for j in range(numPoints)] for i in range(batches)])

    def test_trainOn_model(self):
        if False:
            return 10
        'Test the model on toy data with four clusters.'
        stkm = StreamingKMeans()
        initCenters = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
        stkm.setInitialCenters(centers=initCenters, weights=[1.0, 1.0, 1.0, 1.0])
        offsets = [[0, 0.1], [0, -0.1], [0.1, 0], [-0.1, 0]]
        batches = []
        for offset in offsets:
            batches.append([[offset[0] + center[0], offset[1] + center[1]] for center in initCenters])
        batches = [self.sc.parallelize(batch, 1) for batch in batches]
        input_stream = self.ssc.queueStream(batches)
        stkm.trainOn(input_stream)
        self.ssc.start()

        def condition():
            if False:
                while True:
                    i = 10
            finalModel = stkm.latestModel()
            self.assertTrue(all(finalModel.centers == array(initCenters)))
            self.assertEqual(finalModel.clusterWeights, [5.0, 5.0, 5.0, 5.0])
            return True
        eventually(timeout=90, catch_assertions=True)(condition)()

    def test_predictOn_model(self):
        if False:
            print('Hello World!')
        'Test that the model predicts correctly on toy data.'
        stkm = StreamingKMeans()
        stkm._model = StreamingKMeansModel(clusterCenters=[[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], clusterWeights=[1.0, 1.0, 1.0, 1.0])
        predict_data = [[[1.5, 1.5]], [[-1.5, 1.5]], [[-1.5, -1.5]], [[1.5, -1.5]]]
        predict_data = [self.sc.parallelize(batch, 1) for batch in predict_data]
        predict_stream = self.ssc.queueStream(predict_data)
        predict_val = stkm.predictOn(predict_stream)
        result = []

        def update(rdd):
            if False:
                while True:
                    i = 10
            rdd_collect = rdd.collect()
            if rdd_collect:
                result.append(rdd_collect)
        predict_val.foreachRDD(update)
        self.ssc.start()

        def condition():
            if False:
                while True:
                    i = 10
            self.assertEqual(result, [[0], [1], [2], [3]])
            return True
        eventually(catch_assertions=True)(condition)()

    @unittest.skip('SPARK-10086: Flaky StreamingKMeans test in PySpark')
    def test_trainOn_predictOn(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that prediction happens on the updated model.'
        stkm = StreamingKMeans(decayFactor=0.0, k=2)
        stkm.setInitialCenters([[0.0], [1.0]], [1.0, 1.0])
        batches = [[[-0.5], [0.6], [0.8]], [[0.2], [-0.1], [0.3]]]
        batches = [self.sc.parallelize(batch) for batch in batches]
        input_stream = self.ssc.queueStream(batches)
        predict_results = []

        def collect(rdd):
            if False:
                print('Hello World!')
            rdd_collect = rdd.collect()
            if rdd_collect:
                predict_results.append(rdd_collect)
        stkm.trainOn(input_stream)
        predict_stream = stkm.predictOn(input_stream)
        predict_stream.foreachRDD(collect)
        self.ssc.start()

        def condition():
            if False:
                print('Hello World!')
            self.assertEqual(predict_results, [[0, 1, 1], [1, 0, 1]])
            return True
        eventually(catch_assertions=True)(condition)()

class StreamingLogisticRegressionWithSGDTests(MLLibStreamingTestCase):

    @staticmethod
    def generateLogisticInput(offset, scale, nPoints, seed):
        if False:
            while True:
                i = 10
        '\n        Generate 1 / (1 + exp(-x * scale + offset))\n\n        where,\n        x is randomly distributed and the threshold\n        and labels for each sample in x is obtained from a random uniform\n        distribution.\n        '
        rng = random.RandomState(seed)
        x = rng.randn(nPoints)
        sigmoid = 1.0 / (1 + exp(-(dot(x, scale) + offset)))
        y_p = rng.rand(nPoints)
        cut_off = y_p <= sigmoid
        y_p[cut_off] = 1.0
        y_p[~cut_off] = 0.0
        return [LabeledPoint(y_p[i], Vectors.dense([x[i]])) for i in range(nPoints)]

    def test_parameter_accuracy(self):
        if False:
            return 10
        '\n        Test that the final value of weights is close to the desired value.\n        '
        input_batches = [self.sc.parallelize(self.generateLogisticInput(0, 1.5, 100, 42 + i)) for i in range(20)]
        input_stream = self.ssc.queueStream(input_batches)
        slr = StreamingLogisticRegressionWithSGD(stepSize=0.2, numIterations=25)
        slr.setInitialWeights([0.0])
        slr.trainOn(input_stream)
        self.ssc.start()

        def condition():
            if False:
                while True:
                    i = 10
            rel = (1.5 - slr.latestModel().weights.array[0]) / 1.5
            self.assertAlmostEqual(rel, 0.1, 1)
            return True
        eventually(timeout=120.0, catch_assertions=True)(condition)()

    def test_convergence(self):
        if False:
            return 10
        '\n        Test that weights converge to the required value on toy data.\n        '
        input_batches = [self.sc.parallelize(self.generateLogisticInput(0, 1.5, 100, 42 + i)) for i in range(20)]
        input_stream = self.ssc.queueStream(input_batches)
        models = []
        slr = StreamingLogisticRegressionWithSGD(stepSize=0.2, numIterations=25)
        slr.setInitialWeights([0.0])
        slr.trainOn(input_stream)
        input_stream.foreachRDD(lambda x: models.append(slr.latestModel().weights[0]))
        self.ssc.start()

        def condition():
            if False:
                print('Hello World!')
            self.assertEqual(len(models), len(input_batches))
            return True
        eventually(timeout=120, catch_assertions=True)(condition)()
        t_models = array(models)
        diff = t_models[1:] - t_models[:-1]
        self.assertTrue(all(diff >= -0.1))
        self.assertTrue(array_sum(diff > 0) > 1)

    @staticmethod
    def calculate_accuracy_error(true, predicted):
        if False:
            while True:
                i = 10
        return sum(abs(array(true) - array(predicted))) / len(true)

    def test_predictions(self):
        if False:
            print('Hello World!')
        'Test predicted values on a toy model.'
        input_batches = []
        for i in range(20):
            batch = self.sc.parallelize(self.generateLogisticInput(0, 1.5, 100, 42 + i))
            input_batches.append(batch.map(lambda x: (x.label, x.features)))
        input_stream = self.ssc.queueStream(input_batches)
        slr = StreamingLogisticRegressionWithSGD(stepSize=0.2, numIterations=25)
        slr.setInitialWeights([1.5])
        predict_stream = slr.predictOnValues(input_stream)
        true_predicted = []
        predict_stream.foreachRDD(lambda x: true_predicted.append(x.collect()))
        self.ssc.start()

        def condition():
            if False:
                i = 10
                return i + 15
            self.assertEqual(len(true_predicted), len(input_batches))
            return True
        eventually(catch_assertions=True)(condition)()
        for batch in true_predicted:
            (true, predicted) = zip(*batch)
            self.assertTrue(self.calculate_accuracy_error(true, predicted) < 0.4)

    @unittest.skipIf('COVERAGE_PROCESS_START' in os.environ, 'Flaky with coverage enabled, skipping for now.')
    def test_training_and_prediction(self):
        if False:
            while True:
                i = 10
        'Test that the model improves on toy data with no. of batches'
        input_batches = [self.sc.parallelize(self.generateLogisticInput(0, 1.5, 100, 42 + i)) for i in range(40)]
        predict_batches = [b.map(lambda lp: (lp.label, lp.features)) for b in input_batches]
        slr = StreamingLogisticRegressionWithSGD(stepSize=0.01, numIterations=25)
        slr.setInitialWeights([-0.1])
        errors = []

        def collect_errors(rdd):
            if False:
                i = 10
                return i + 15
            (true, predicted) = zip(*rdd.collect())
            errors.append(self.calculate_accuracy_error(true, predicted))
        input_stream = self.ssc.queueStream(input_batches)
        predict_stream = self.ssc.queueStream(predict_batches)
        slr.trainOn(input_stream)
        ps = slr.predictOnValues(predict_stream)
        ps.foreachRDD(lambda x: collect_errors(x))
        self.ssc.start()

        def condition():
            if False:
                for i in range(10):
                    print('nop')
            if len(errors) == len(predict_batches):
                self.assertGreater(errors[1] - errors[-1], 0.3)
            if len(errors) >= 3 and errors[1] - errors[-1] > 0.3:
                return True
            return 'Latest errors: ' + ', '.join(map(lambda x: str(x), errors))
        eventually(timeout=180.0)(condition)()

class StreamingLinearRegressionWithTests(MLLibStreamingTestCase):

    def assertArrayAlmostEqual(self, array1, array2, dec):
        if False:
            for i in range(10):
                print('nop')
        for (i, j) in (array1, array2):
            self.assertAlmostEqual(i, j, dec)

    def test_parameter_accuracy(self):
        if False:
            while True:
                i = 10
        'Test that coefs are predicted accurately by fitting on toy data.'
        slr = StreamingLinearRegressionWithSGD(stepSize=0.2, numIterations=25)
        slr.setInitialWeights([0.0, 0.0])
        xMean = [0.0, 0.0]
        xVariance = [1.0 / 3.0, 1.0 / 3.0]
        batches = []
        for i in range(10):
            batch = LinearDataGenerator.generateLinearInput(0.0, [10.0, 10.0], xMean, xVariance, 100, 42 + i, 0.1)
            batches.append(self.sc.parallelize(batch))
        input_stream = self.ssc.queueStream(batches)
        slr.trainOn(input_stream)
        self.ssc.start()

        def condition():
            if False:
                return 10
            self.assertArrayAlmostEqual(slr.latestModel().weights.array, [10.0, 10.0], 1)
            self.assertAlmostEqual(slr.latestModel().intercept, 0.0, 1)
            return True
        eventually(catch_assertions=True)(condition)()

    def test_parameter_convergence(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the model parameters improve with streaming data.'
        slr = StreamingLinearRegressionWithSGD(stepSize=0.2, numIterations=25)
        slr.setInitialWeights([0.0])
        batches = []
        for i in range(10):
            batch = LinearDataGenerator.generateLinearInput(0.0, [10.0], [0.0], [1.0 / 3.0], 100, 42 + i, 0.1)
            batches.append(self.sc.parallelize(batch))
        model_weights = []
        input_stream = self.ssc.queueStream(batches)
        input_stream.foreachRDD(lambda x: model_weights.append(slr.latestModel().weights[0]))
        slr.trainOn(input_stream)
        self.ssc.start()

        def condition():
            if False:
                print('Hello World!')
            self.assertEqual(len(model_weights), len(batches))
            return True
        eventually(timeout=90, catch_assertions=True)(condition)()
        w = array(model_weights)
        diff = w[1:] - w[:-1]
        self.assertTrue(all(diff >= -0.1))

    def test_prediction(self):
        if False:
            while True:
                i = 10
        'Test prediction on a model with weights already set.'
        slr = StreamingLinearRegressionWithSGD(stepSize=0.2, numIterations=25)
        slr.setInitialWeights([10.0, 10.0])
        batches = []
        for i in range(10):
            batch = LinearDataGenerator.generateLinearInput(0.0, [10.0, 10.0], [0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0], 100, 42 + i, 0.1)
            batches.append(self.sc.parallelize(batch).map(lambda lp: (lp.label, lp.features)))
        input_stream = self.ssc.queueStream(batches)
        output_stream = slr.predictOnValues(input_stream)
        samples = []
        output_stream.foreachRDD(lambda x: samples.append(x.collect()))
        self.ssc.start()

        def condition():
            if False:
                i = 10
                return i + 15
            self.assertEqual(len(samples), len(batches))
            return True
        eventually(catch_assertions=True)(condition)()
        for batch in samples:
            (true, predicted) = zip(*batch)
            self.assertTrue(mean(abs(array(true) - array(predicted))) < 0.1)

    @unittest.skipIf('COVERAGE_PROCESS_START' in os.environ, 'Flaky with coverage enabled, skipping for now.')
    def test_train_prediction(self):
        if False:
            print('Hello World!')
        'Test that error on test data improves as model is trained.'
        slr = StreamingLinearRegressionWithSGD(stepSize=0.2, numIterations=25)
        slr.setInitialWeights([0.0])
        batches = []
        for i in range(15):
            batch = LinearDataGenerator.generateLinearInput(0.0, [10.0], [0.0], [1.0 / 3.0], 100, 42 + i, 0.1)
            batches.append(self.sc.parallelize(batch))
        predict_batches = [b.map(lambda lp: (lp.label, lp.features)) for b in batches]
        errors = []

        def func(rdd):
            if False:
                for i in range(10):
                    print('nop')
            (true, predicted) = zip(*rdd.collect())
            errors.append(mean(abs(true) - abs(predicted)))
        input_stream = self.ssc.queueStream(batches)
        output_stream = self.ssc.queueStream(predict_batches)
        slr.trainOn(input_stream)
        output_stream = slr.predictOnValues(output_stream)
        output_stream.foreachRDD(func)
        self.ssc.start()

        def condition():
            if False:
                print('Hello World!')
            if len(errors) == len(predict_batches):
                self.assertGreater(errors[1] - errors[-1], 2)
            if len(errors) >= 3 and errors[1] - errors[-1] > 2:
                return True
            return 'Latest errors: ' + ', '.join(map(lambda x: str(x), errors))
        eventually(timeout=180.0)(condition)()
if __name__ == '__main__':
    from pyspark.mllib.tests.test_streaming_algorithms import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)