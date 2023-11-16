from bigdl.dllib.feature.common import *
from bigdl.dllib.utils.log4Error import *
if sys.version >= '3':
    long = int
    unicode = str

class XGBClassifier:

    def __init__(self, params=None):
        if False:
            for i in range(10):
                print('nop')
        super(XGBClassifier, self).__init__()
        bigdl_type = 'float'
        self.value = callZooFunc('float', 'getXGBClassifier', params)

    def setNthread(self, value: int):
        if False:
            i = 10
            return i + 15
        return callZooFunc('float', 'setXGBClassifierNthread', self.value, value)

    def setNumRound(self, value: int):
        if False:
            for i in range(10):
                print('nop')
        return callZooFunc('float', 'setXGBClassifierNumRound', self.value, value)

    def setNumWorkers(self, value: int):
        if False:
            for i in range(10):
                print('nop')
        return callZooFunc('float', 'setXGBClassifierNumWorkers', self.value, value)

    def fit(self, df):
        if False:
            return 10
        model = callZooFunc('float', 'fitXGBClassifier', self.value, df)
        xgb_model = XGBClassifierModel(model)
        features_col = callZooFunc('float', 'getXGBClassifierFeaturesCol', self.value)
        xgb_model.feature_names = [f'f{i}' for i in range(len(df.first()[features_col]))]
        return xgb_model

    def setMissing(self, value: int):
        if False:
            i = 10
            return i + 15
        return callZooFunc('float', 'setXGBClassifierMissing', self.value, value)

    def setMaxDepth(self, value: int):
        if False:
            return 10
        return callZooFunc('float', 'setXGBClassifierMaxDepth', self.value, value)

    def setEta(self, value: float):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setXGBClassifierEta', self.value, value)

    def setGamma(self, value: int):
        if False:
            while True:
                i = 10
        return callZooFunc('float', 'setXGBClassifierGamma', self.value, value)

    def setTreeMethod(self, value: str):
        if False:
            for i in range(10):
                print('nop')
        return callZooFunc('float', 'setXGBClassifierTreeMethod', self.value, value)

    def setObjective(self, value: str):
        if False:
            while True:
                i = 10
        return callZooFunc('float', 'setXGBClassifierObjective', self.value, value)

    def setNumClass(self, value: str):
        if False:
            return 10
        return callZooFunc('float', 'setXGBClassifierNumClass', self.value, value)

    def setFeaturesCol(self, value: str):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setXGBClassifierFeaturesCol', self.value, value)

class XGBClassifierModel:
    """
    XGBClassifierModel is a trained XGBoost classification model. The prediction column
    will have the prediction results.
    """

    def __init__(self, jvalue):
        if False:
            print('Hello World!')
        super(XGBClassifierModel, self).__init__()
        invalidInputError(jvalue is not None, 'XGBClassifierModel jvalue cannot be None')
        self.value = jvalue
        self.feature_names = []

    def setFeaturesCol(self, features):
        if False:
            return 10
        return callZooFunc('float', 'setFeaturesXGBClassifierModel', self.value, features)

    def setPredictionCol(self, prediction):
        if False:
            return 10
        return callZooFunc('float', 'setPredictionXGBClassifierModel', self.value, prediction)

    def setInferBatchSize(self, batch_size):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setInferBatchSizeXGBClassifierModel', self.value, batch_size)

    def transform(self, dataset):
        if False:
            print('Hello World!')
        df = callZooFunc('float', 'transformXGBClassifierModel', self.value, dataset)
        return df

    def getFScore(self, fmap=''):
        if False:
            print('Hello World!')
        scores = callZooFunc('float', 'getFeatureScoreXGBClassifierModel', self.value, fmap)
        return scores

    def getScore(self, fmap='', importance_type='weight'):
        if False:
            i = 10
            return i + 15
        score = callZooFunc('float', 'getScoreXGBClassifierModel', self.value, fmap, importance_type)
        return score

    @property
    def feature_importances(self):
        if False:
            i = 10
            return i + 15
        score = callZooFunc('float', 'getFeatureImportanceXGBClassifierModel', self.value)
        all_features = [score.get(f, 0.0) for f in self.feature_names]
        all_features_arr = np.array(all_features, dtype=np.float32)
        return all_features_arr

    def saveModel(self, path):
        if False:
            for i in range(10):
                print('nop')
        callZooFunc('float', 'saveXGBClassifierModel', self.value, path)

    @staticmethod
    def loadModel(path, numClasses):
        if False:
            return 10
        '\n        load a pretrained XGBoostClassificationModel\n        :param path: pretrained model path\n        :param numClasses: number of classes for classification\n        '
        jvalue = callZooFunc('float', 'loadXGBClassifierModel', path, numClasses)
        return XGBClassifierModel(jvalue=jvalue)

class XGBRegressor:

    def __init__(self, params=None):
        if False:
            i = 10
            return i + 15
        super(XGBRegressor, self).__init__()
        bigdl_type = 'float'
        self.value = callZooFunc('float', 'getXGBRegressor', params)

    def setNthread(self, value: int):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setXGBRegressorNthread', self.value, value)

    def setNumRound(self, value: int):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setXGBRegressorNumRound', self.value, value)

    def setNumWorkers(self, value: int):
        if False:
            for i in range(10):
                print('nop')
        return callZooFunc('float', 'setXGBRegressorNumWorkers', self.value, value)

    def fit(self, df):
        if False:
            while True:
                i = 10
        model = callZooFunc('float', 'fitXGBRegressor', self.value, df)
        return XGBRegressorModel(model)

class XGBRegressorModel:

    def __init__(self, jvalue):
        if False:
            return 10
        super(XGBRegressorModel, self).__init__()
        invalidInputError(jvalue is not None, 'XGBRegressorModel jvalue cannot be None')
        self.value = jvalue

    def setFeaturesCol(self, features):
        if False:
            return 10
        return callZooFunc('float', 'setFeaturesXGBRegressorModel', self.value, features)

    def setPredictionCol(self, prediction):
        if False:
            i = 10
            return i + 15
        return callZooFunc('float', 'setPredictionXGBRegressorModel', self.value, prediction)

    def setInferBatchSize(self, value: int):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setInferBatchSizeXGBRegressorModel', self.value, value)

    def transform(self, dataset):
        if False:
            i = 10
            return i + 15
        df = callZooFunc('float', 'transformXGBRegressorModel', self.value, dataset)
        return df

    def save(self, path):
        if False:
            i = 10
            return i + 15
        print('start saving in python side')
        callZooFunc('float', 'saveXGBRegressorModel', self.value, path)

    @staticmethod
    def load(path):
        if False:
            i = 10
            return i + 15
        jvalue = callZooFunc('float', 'loadXGBRegressorModel', path)
        return XGBRegressorModel(jvalue=jvalue)

class LightGBMClassifier:

    def __init__(self, params=None):
        if False:
            while True:
                i = 10
        super(LightGBMClassifier, self).__init__()
        bigdl_type = 'float'
        self.value = callZooFunc('float', 'getLightGBMClassifier', params)

    def setFeaturesCol(self, value: str):
        if False:
            while True:
                i = 10
        return callZooFunc('float', 'setLGBMFeaturesCol', self.value, value)

    def setLabelCol(self, value: str):
        if False:
            while True:
                i = 10
        return callZooFunc('float', 'setLGBMLabelCol', self.value, value)

    def setBoostType(self, value: int):
        if False:
            return 10
        return callZooFunc('float', 'setLGBMBoostType', self.value, value)

    def fit(self, df):
        if False:
            while True:
                i = 10
        model = callZooFunc('float', 'fitLGBM', self.value, df)
        model = LightGBMClassifierModel(model)
        return model

    def setMaxDepth(self, value: int):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setLGBMMaxDepth', self.value, value)

    def setObjective(self, value: str):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setLGBMObjective', self.value, value)

    def setLearningRate(self, value: str):
        if False:
            for i in range(10):
                print('nop')
        return callZooFunc('float', 'setLGBMLearningRate', self.value, value)

    def setNumIterations(self, value: int):
        if False:
            i = 10
            return i + 15
        return callZooFunc('float', 'setLGBMNumIterations', self.value, value)

class LightGBMClassifierModel:
    """
    LightGBMClassifierModel is a trained LightGBMClassification model. The prediction column
    will have the prediction results.
    """

    def __init__(self, jvalue):
        if False:
            print('Hello World!')
        super(LightGBMClassifierModel, self).__init__()
        invalidInputError(jvalue is not None, 'LightGBMClassifierModel jvalue cannot be None')
        self.value = jvalue

    def setFeaturesCol(self, features):
        if False:
            return 10
        return callZooFunc('float', 'setFeaturesLGBMModel', self.value, features)

    def setPredictionCol(self, prediction):
        if False:
            for i in range(10):
                print('nop')
        return callZooFunc('float', 'setPredictionLGBMModel', self.value, prediction)

    def transform(self, dataset):
        if False:
            print('Hello World!')
        df = callZooFunc('float', 'transformLGBMModel', self.value, dataset)
        return df

    def saveModel(self, path):
        if False:
            while True:
                i = 10
        callZooFunc('float', 'saveLGBMModel', self.value, path)

    @staticmethod
    def loadModel(path):
        if False:
            return 10
        '\n        load a pretrained LightGBMClassificationModel\n        :param path: pretrained model path\n        '
        jvalue = callZooFunc('float', 'loadLGBMClassifierModel', path)
        return LightGBMClassifierModel(jvalue=jvalue)

class LightGBMRegressor:

    def __init__(self, params=None):
        if False:
            print('Hello World!')
        super(LightGBMRegressor, self).__init__()
        bigdl_type = 'float'
        self.value = callZooFunc('float', 'getLightGBMRegressor', params)

    def setFeaturesCol(self, value: str):
        if False:
            i = 10
            return i + 15
        return callZooFunc('float', 'setLGBMFeaturesCol', self.value, value)

    def setLabelCol(self, value: str):
        if False:
            for i in range(10):
                print('nop')
        return callZooFunc('float', 'setLGBMLabelCol', self.value, value)

    def setBoostType(self, value: int):
        if False:
            for i in range(10):
                print('nop')
        return callZooFunc('float', 'setLGBMBoostType', self.value, value)

    def fit(self, df):
        if False:
            while True:
                i = 10
        model = callZooFunc('float', 'fitLGBM', self.value, df)
        model = LightGBMRegressorModel(model)
        return model

    def setMaxDepth(self, value: int):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setLGBMMaxDepth', self.value, value)

    def setObjective(self, value: str):
        if False:
            return 10
        return callZooFunc('float', 'setLGBMObjective', self.value, value)

    def setLearningRate(self, value: str):
        if False:
            while True:
                i = 10
        return callZooFunc('float', 'setLGBMLearningRate', self.value, value)

    def setNumIterations(self, value: int):
        if False:
            for i in range(10):
                print('nop')
        return callZooFunc('float', 'setLGBMNumIterations', self.value, value)

class LightGBMRegressorModel:
    """
    LightGBMRegressorModel is a trained LightGBMRegression model. The prediction column
    will have the prediction results.
    """

    def __init__(self, jvalue):
        if False:
            print('Hello World!')
        super(LightGBMRegressorModel, self).__init__()
        invalidInputError(jvalue is not None, 'LightGBMRegressorModel jvalue cannot be None')
        self.value = jvalue

    def setFeaturesCol(self, features):
        if False:
            print('Hello World!')
        return callZooFunc('float', 'setFeaturesLGBMModel', self.value, features)

    def setPredictionCol(self, prediction):
        if False:
            while True:
                i = 10
        return callZooFunc('float', 'setPredictionLGBMModel', self.value, prediction)

    def transform(self, dataset):
        if False:
            while True:
                i = 10
        df = callZooFunc('float', 'transformLGBMModel', self.value, dataset)
        return df

    def saveModel(self, path):
        if False:
            print('Hello World!')
        callZooFunc('float', 'saveLGBMModel', self.value, path)

    @staticmethod
    def loadModel(path):
        if False:
            print('Hello World!')
        '\n        load a pretrained LightGBMRegressorModel\n        :param path: pretrained model path\n        '
        jvalue = callZooFunc('float', 'loadLGBMRegressorModel', path)
        return LightGBMRegressorModel(jvalue=jvalue)