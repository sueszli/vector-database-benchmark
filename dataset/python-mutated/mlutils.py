import numpy as np
from pyspark import keyword_only
from pyspark.ml import Estimator, Model, Transformer, UnaryTransformer
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasMaxIter, HasRegParam
from pyspark.ml.classification import Classifier, ClassificationModel
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.wrapper import _java2py
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType
from pyspark.testing.utils import ReusedPySparkTestCase as PySparkTestCase

def check_params(test_self, py_stage, check_params_exist=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Checks common requirements for :py:class:`PySpark.ml.Params.params`:\n\n      - set of params exist in Java and Python and are ordered by names\n      - param parent has the same UID as the object's UID\n      - default param value from Java matches value in Python\n      - optionally check if all params from Java also exist in Python\n    "
    py_stage_str = '%s %s' % (type(py_stage), py_stage)
    if not hasattr(py_stage, '_to_java'):
        return
    java_stage = py_stage._to_java()
    if java_stage is None:
        return
    test_self.assertEqual(py_stage.uid, java_stage.uid(), msg=py_stage_str)
    if check_params_exist:
        param_names = [p.name for p in py_stage.params]
        java_params = list(java_stage.params())
        java_param_names = [jp.name() for jp in java_params]
        test_self.assertEqual(param_names, sorted(java_param_names), 'Param list in Python does not match Java for %s:\nJava = %s\nPython = %s' % (py_stage_str, java_param_names, param_names))
    for p in py_stage.params:
        test_self.assertEqual(p.parent, py_stage.uid)
        java_param = java_stage.getParam(p.name)
        py_has_default = py_stage.hasDefault(p)
        java_has_default = java_stage.hasDefault(java_param)
        test_self.assertEqual(py_has_default, java_has_default, 'Default value mismatch of param %s for Params %s' % (p.name, str(py_stage)))
        if py_has_default:
            if p.name == 'seed':
                continue
            java_default = _java2py(test_self.sc, java_stage.clear(java_param).getOrDefault(java_param))
            py_stage.clear(p)
            py_default = py_stage.getOrDefault(p)
            if isinstance(java_default, float) and np.isnan(java_default):
                java_default = 'NaN'
                py_default = 'NaN' if np.isnan(py_default) else 'not NaN'
            test_self.assertEqual(java_default, py_default, 'Java default %s != python default %s of param %s for Params %s' % (str(java_default), str(py_default), p.name, str(py_stage)))

class SparkSessionTestCase(PySparkTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        PySparkTestCase.setUpClass()
        cls.spark = SparkSession(cls.sc)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        PySparkTestCase.tearDownClass()
        cls.spark.stop()

class MockDataset(DataFrame):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.index = 0

class HasFake(Params):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(HasFake, self).__init__()
        self.fake = Param(self, 'fake', 'fake param')

    def getFake(self):
        if False:
            while True:
                i = 10
        return self.getOrDefault(self.fake)

class MockTransformer(Transformer, HasFake):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(MockTransformer, self).__init__()
        self.dataset_index = None

    def _transform(self, dataset):
        if False:
            print('Hello World!')
        self.dataset_index = dataset.index
        dataset.index += 1
        return dataset

class MockUnaryTransformer(UnaryTransformer, DefaultParamsReadable, DefaultParamsWritable):
    shift = Param(Params._dummy(), 'shift', 'The amount by which to shift ' + 'data in a DataFrame', typeConverter=TypeConverters.toFloat)

    def __init__(self, shiftVal=1):
        if False:
            while True:
                i = 10
        super(MockUnaryTransformer, self).__init__()
        self._setDefault(shift=1)
        self._set(shift=shiftVal)

    def getShift(self):
        if False:
            print('Hello World!')
        return self.getOrDefault(self.shift)

    def setShift(self, shift):
        if False:
            for i in range(10):
                print('nop')
        self._set(shift=shift)

    def createTransformFunc(self):
        if False:
            print('Hello World!')
        shiftVal = self.getShift()
        return lambda x: x + shiftVal

    def outputDataType(self):
        if False:
            print('Hello World!')
        return DoubleType()

    def validateInputType(self, inputType):
        if False:
            i = 10
            return i + 15
        if inputType != DoubleType():
            raise TypeError('Bad input type: {}. '.format(inputType) + 'Requires Double.')

class MockEstimator(Estimator, HasFake):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(MockEstimator, self).__init__()
        self.dataset_index = None

    def _fit(self, dataset):
        if False:
            return 10
        self.dataset_index = dataset.index
        model = MockModel()
        self._copyValues(model)
        return model

class MockModel(MockTransformer, Model, HasFake):
    pass

class _DummyLogisticRegressionParams(HasMaxIter, HasRegParam):

    def setMaxIter(self, value):
        if False:
            for i in range(10):
                print('nop')
        return self._set(maxIter=value)

    def setRegParam(self, value):
        if False:
            while True:
                i = 10
        return self._set(regParam=value)

class DummyLogisticRegression(Classifier, _DummyLogisticRegressionParams, DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, regParam=0.0, rawPredictionCol='rawPrediction'):
        if False:
            return 10
        super(DummyLogisticRegression, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, regParam=0.0, rawPredictionCol='rawPrediction'):
        if False:
            i = 10
            return i + 15
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def _fit(self, dataset):
        if False:
            while True:
                i = 10
        return self._copyValues(DummyLogisticRegressionModel())

class DummyLogisticRegressionModel(ClassificationModel, _DummyLogisticRegressionParams, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self):
        if False:
            print('Hello World!')
        super(DummyLogisticRegressionModel, self).__init__()

    def _transform(self, dataset):
        if False:
            print('Hello World!')
        from pyspark.sql.functions import array, lit
        from pyspark.ml.functions import array_to_vector
        rawPredCol = self.getRawPredictionCol()
        if rawPredCol:
            dataset = dataset.withColumn(rawPredCol, array_to_vector(array(lit(-100.0), lit(100.0))))
        predCol = self.getPredictionCol()
        if predCol:
            dataset = dataset.withColumn(predCol, lit(1.0))
        return dataset

    @property
    def numClasses(self):
        if False:
            return 10
        return 2

    @property
    def intercept(self):
        if False:
            i = 10
            return i + 15
        return 0.0

    @property
    def coefficients(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def predictRaw(self, value):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def numFeatures(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def predict(self, value):
        if False:
            print('Hello World!')
        raise NotImplementedError()

class DummyEvaluator(Evaluator, DefaultParamsReadable, DefaultParamsWritable):

    def _evaluate(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        return 1.0