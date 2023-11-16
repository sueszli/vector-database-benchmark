from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans
if __name__ == '__main__':
    sc = SparkContext(appName='StreamingKMeansExample')
    ssc = StreamingContext(sc, 1)

    def parse(lp):
        if False:
            print('Hello World!')
        label = float(lp[lp.find('(') + 1:lp.find(')')])
        vec = Vectors.dense(lp[lp.find('[') + 1:lp.find(']')].split(','))
        return LabeledPoint(label, vec)
    trainingData = sc.textFile('data/mllib/kmeans_data.txt').map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
    testingData = sc.textFile('data/mllib/streaming_kmeans_data_test.txt').map(parse)
    trainingQueue = [trainingData]
    testingQueue = [testingData]
    trainingStream = ssc.queueStream(trainingQueue)
    testingStream = ssc.queueStream(testingQueue)
    model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(3, 1.0, 0)
    model.trainOn(trainingStream)
    result = model.predictOnValues(testingStream.map(lambda lp: (lp.label, lp.features)))
    result.pprint()
    ssc.start()
    ssc.stop(stopSparkContext=True, stopGraceFully=True)
    print('Final centers: ' + str(model.latestModel().centers))