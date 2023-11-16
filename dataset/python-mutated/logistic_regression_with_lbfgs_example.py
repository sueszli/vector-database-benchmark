"""
Logistic Regression With LBFGS Example.
"""
from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
if __name__ == '__main__':
    sc = SparkContext(appName='PythonLogisticRegressionWithLBFGSExample')

    def parsePoint(line):
        if False:
            i = 10
            return i + 15
        values = [float(x) for x in line.split(' ')]
        return LabeledPoint(values[0], values[1:])
    data = sc.textFile('data/mllib/sample_svm_data.txt')
    parsedData = data.map(parsePoint)
    model = LogisticRegressionWithLBFGS.train(parsedData)
    labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
    print('Training Error = ' + str(trainErr))
    model.save(sc, 'target/tmp/pythonLogisticRegressionWithLBFGSModel')
    sameModel = LogisticRegressionModel.load(sc, 'target/tmp/pythonLogisticRegressionWithLBFGSModel')