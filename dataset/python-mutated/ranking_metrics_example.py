from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark import SparkContext
if __name__ == '__main__':
    sc = SparkContext(appName='Ranking Metrics Example')
    lines = sc.textFile('data/mllib/sample_movielens_data.txt')

    def parseLine(line):
        if False:
            return 10
        fields = line.split('::')
        return Rating(int(fields[0]), int(fields[1]), float(fields[2]) - 2.5)
    ratings = lines.map(lambda r: parseLine(r))
    model = ALS.train(ratings, 10, 10, 0.01)
    testData = ratings.map(lambda p: (p.user, p.product))
    predictions = model.predictAll(testData).map(lambda r: ((r.user, r.product), r.rating))
    ratingsTuple = ratings.map(lambda r: ((r.user, r.product), r.rating))
    scoreAndLabels = predictions.join(ratingsTuple).map(lambda tup: tup[1])
    metrics = RegressionMetrics(scoreAndLabels)
    print('RMSE = %s' % metrics.rootMeanSquaredError)
    print('R-squared = %s' % metrics.r2)