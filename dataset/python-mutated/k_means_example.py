from numpy import array
from math import sqrt
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
if __name__ == '__main__':
    sc = SparkContext(appName='KMeansExample')
    data = sc.textFile('data/mllib/kmeans_data.txt')
    parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
    clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode='random')

    def error(point):
        if False:
            print('Hello World!')
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x ** 2 for x in point - center]))
    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print('Within Set Sum of Squared Error = ' + str(WSSSE))
    clusters.save(sc, 'target/org/apache/spark/PythonKMeansExample/KMeansModel')
    sameModel = KMeansModel.load(sc, 'target/org/apache/spark/PythonKMeansExample/KMeansModel')
    sc.stop()