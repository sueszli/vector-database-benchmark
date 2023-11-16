"""
Isotonic Regression Example.
"""
from pyspark import SparkContext
import math
from pyspark.mllib.regression import IsotonicRegression, IsotonicRegressionModel
from pyspark.mllib.util import MLUtils
if __name__ == '__main__':
    sc = SparkContext(appName='PythonIsotonicRegressionExample')

    def parsePoint(labeledData):
        if False:
            return 10
        return (labeledData.label, labeledData.features[0], 1.0)
    data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_isotonic_regression_libsvm_data.txt')
    parsedData = data.map(parsePoint)
    (training, test) = parsedData.randomSplit([0.6, 0.4], 11)
    model = IsotonicRegression.train(training)
    predictionAndLabel = test.map(lambda p: (model.predict(p[1]), p[0]))
    meanSquaredError = predictionAndLabel.map(lambda pl: math.pow(pl[0] - pl[1], 2)).mean()
    print('Mean Squared Error = ' + str(meanSquaredError))
    model.save(sc, 'target/tmp/myIsotonicRegressionModel')
    sameModel = IsotonicRegressionModel.load(sc, 'target/tmp/myIsotonicRegressionModel')