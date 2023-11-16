from pyspark.ml.pipeline import Estimator, Model
from pyspark.ml.param.shared import *
from bigdl.dllib.utils.common import *
import sys
if sys.version >= '3':
    long = int
    unicode = str

class HasBatchSize(Params):
    """
    Mixin for param batchSize: batch size.
    """
    batchSize = Param(Params._dummy(), 'batchSize', 'batchSize')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(HasBatchSize, self).__init__()
        self.batchSize = Param(self, 'batchSize', 'batchSize')
        self._setDefault(batchSize=1)

    def setBatchSize(self, val):
        if False:
            return 10
        '\n        Sets the value of :py:attr:`batchSize`.\n        '
        self._paramMap[self.batchSize] = val
        pythonBigDL_method_name = 'setBatchSize' + self.__class__.__name__
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        return self

    def getBatchSize(self):
        if False:
            return 10
        '\n        Gets the value of batchSize or its default value.\n        '
        return self.getOrDefault(self.batchSize)

class HasMaxEpoch(Params):
    maxEpoch = Param(Params._dummy(), 'maxEpoch', 'number of max Epoch')

    def __init__(self):
        if False:
            while True:
                i = 10
        super(HasMaxEpoch, self).__init__()
        self.maxEpoch = Param(self, 'maxEpoch', 'maxEpoch')
        self._setDefault(maxEpoch=100)

    def setMaxEpoch(self, val):
        if False:
            print('Hello World!')
        self._paramMap[self.maxEpoch] = val
        pythonBigDL_method_name = 'setMaxEpoch' + self.__class__.__name__
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        return self

    def getMaxEpoch(self):
        if False:
            while True:
                i = 10
        '\n        Gets the value of maxEpoch or its default value.\n        '
        return self.getOrDefault(self.maxEpoch)

class HasFeatureSize(Params):
    featureSize = Param(Params._dummy(), 'featureSize', 'size of the feature')

    def __init__(self):
        if False:
            while True:
                i = 10
        super(HasFeatureSize, self).__init__()
        self.featureSize = Param(self, 'featureSize', 'featureSize')
        self._setDefault(featureSize=None)

    def setFeatureSize(self, val):
        if False:
            return 10
        self._paramMap[self.featureSize] = val
        pythonBigDL_mehtod_name = 'setFeatureSize' + self.__class__.__name__
        callBigDlFunc(self.bigdl_type, pythonBigDL_mehtod_name, self.value, val)
        return self

    def getFeatureSize(self):
        if False:
            print('Hello World!')
        return self.getOrDefault(self.featureSize)

class HasLearningRate(Params):
    learningRate = Param(Params._dummy(), 'learningRate', 'learning rate')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(HasLearningRate, self).__init__()
        self.learningRate = Param(self, 'learningRate', 'learning rate')
        self._setDefault(learningRate=100)

    def setLearningRate(self, val):
        if False:
            print('Hello World!')
        self._paramMap[self.learningRate] = val
        pythonBigDL_method_name = 'setLearningRate' + self.__class__.__name__
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        return self

    def getLearningRate(self):
        if False:
            print('Hello World!')
        '\n        Gets the value of maxEpoch or its default value.\n        '
        return self.getOrDefault(self.learningRate)

class DLEstimator(Estimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasBatchSize, HasMaxEpoch, HasLearningRate, JavaValue):
    """
    .. note:: Deprecated in 0.5.0. `DLEstimator` has been migrated to package
     `bigdl.dlframes`. This will be removed in BigDL 0.6.

    """

    def __init__(self, model, criterion, feature_size, label_size, jvalue=None, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        super(DLEstimator, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(bigdl_type, self.jvm_class_constructor(), model, criterion, feature_size, label_size)
        self.bigdl_type = bigdl_type
        self.featureSize = feature_size

    def _fit(self, dataset):
        if False:
            i = 10
            return i + 15
        jmodel = callBigDlFunc(self.bigdl_type, 'fitEstimator', self.value, dataset)
        model = DLModel.of(jmodel, self.featureSize, self.bigdl_type)
        return model

class DLModel(Model, HasFeaturesCol, HasPredictionCol, HasBatchSize, HasFeatureSize, JavaValue):
    """
    .. note:: Deprecated in 0.5.0. `DLModel` has been migrated to package
     `bigdl.dlframes`. This will be removed in BigDL 0.6.

    """

    def __init__(self, model, featureSize, jvalue=None, bigdl_type='float'):
        if False:
            while True:
                i = 10
        super(DLModel, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(bigdl_type, self.jvm_class_constructor(), model, featureSize)
        self.bigdl_type = bigdl_type
        self.setFeatureSize(featureSize)

    def _transform(self, dataset):
        if False:
            while True:
                i = 10
        return callBigDlFunc(self.bigdl_type, 'dlModelTransform', self.value, dataset)

    @classmethod
    def of(self, jvalue, feature_size=None, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        model = DLModel(model=None, featureSize=feature_size, jvalue=jvalue, bigdl_type=bigdl_type)
        return model

class DLClassifier(DLEstimator):
    """
    .. note:: Deprecated in 0.5.0. `DLClassifier` has been migrated to package
     `bigdl.dlframes`. This will be removed in BigDL 0.6.

    """

    def __init__(self, model, criterion, feature_size, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        super(DLClassifier, self).__init__(model, criterion, feature_size, [1], None, bigdl_type)

    def _fit(self, dataset):
        if False:
            while True:
                i = 10
        jmodel = callBigDlFunc(self.bigdl_type, 'fitClassifier', self.value, dataset)
        model = DLClassifierModel.of(jmodel, self.featureSize, self.bigdl_type)
        return model

class DLClassifierModel(DLModel):
    """
    .. note:: Deprecated in 0.5.0. `DLClassifierModel` has been migrated to package
     `bigdl.dlframes`. This will be removed in BigDL 0.6.

    """

    def __init__(self, model, featureSize, jvalue=None, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        super(DLClassifierModel, self).__init__(model, featureSize, jvalue, bigdl_type)

    def _transform(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        return callBigDlFunc(self.bigdl_type, 'dlClassifierModelTransform', self.value, dataset)

    @classmethod
    def of(self, jvalue, feature_size=None, bigdl_type='float'):
        if False:
            return 10
        model = DLClassifierModel(model=None, featureSize=feature_size, jvalue=jvalue, bigdl_type=bigdl_type)
        return model