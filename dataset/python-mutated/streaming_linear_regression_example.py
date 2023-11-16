"""
Streaming Linear Regression Example.
"""
import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: streaming_linear_regression_example.py <trainingDir> <testDir>', file=sys.stderr)
        sys.exit(-1)
    sc = SparkContext(appName='PythonLogisticRegressionWithLBFGSExample')
    ssc = StreamingContext(sc, 1)

    def parse(lp):
        if False:
            for i in range(10):
                print('nop')
        label = float(lp[lp.find('(') + 1:lp.find(',')])
        vec = Vectors.dense(lp[lp.find('[') + 1:lp.find(']')].split(','))
        return LabeledPoint(label, vec)
    trainingData = ssc.textFileStream(sys.argv[1]).map(parse).cache()
    testData = ssc.textFileStream(sys.argv[2]).map(parse)
    numFeatures = 3
    model = StreamingLinearRegressionWithSGD()
    model.setInitialWeights([0.0, 0.0, 0.0])
    model.trainOn(trainingData)
    print(model.predictOnValues(testData.map(lambda lp: (lp.label, lp.features))))
    ssc.start()
    ssc.awaitTermination()