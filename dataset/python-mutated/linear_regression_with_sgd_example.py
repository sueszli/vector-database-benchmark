"""
Linear Regression With SGD Example.
"""
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
if __name__ == '__main__':
    sc = SparkContext(appName='PythonLinearRegressionWithSGDExample')

    def parsePoint(line):
        if False:
            while True:
                i = 10
        values = [float(x) for x in line.replace(',', ' ').split(' ')]
        return LabeledPoint(values[0], values[1:])
    data = sc.textFile('data/mllib/ridge-data/lpsa.data')
    parsedData = data.map(parsePoint)
    model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=1e-08)
    valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    MSE = valuesAndPreds.map(lambda vp: (vp[0] - vp[1]) ** 2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    print('Mean Squared Error = ' + str(MSE))
    model.save(sc, 'target/tmp/pythonLinearRegressionWithSGDModel')
    sameModel = LinearRegressionModel.load(sc, 'target/tmp/pythonLinearRegressionWithSGDModel')