from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import DenseVector
from pyspark import SparkContext
if __name__ == '__main__':
    sc = SparkContext(appName='Regression Metrics Example')

    def parsePoint(line):
        if False:
            print('Hello World!')
        values = line.split()
        return LabeledPoint(float(values[0]), DenseVector([float(x.split(':')[1]) for x in values[1:]]))
    data = sc.textFile('data/mllib/sample_linear_regression_data.txt')
    parsedData = data.map(parsePoint)
    model = LinearRegressionWithSGD.train(parsedData)
    valuesAndPreds = parsedData.map(lambda p: (float(model.predict(p.features)), p.label))
    metrics = RegressionMetrics(valuesAndPreds)
    print('MSE = %s' % metrics.meanSquaredError)
    print('RMSE = %s' % metrics.rootMeanSquaredError)
    print('R-squared = %s' % metrics.r2)
    print('MAE = %s' % metrics.meanAbsoluteError)
    print('Explained variance = %s' % metrics.explainedVariance)