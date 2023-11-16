import sys
import time
from pyspark.sql import SparkSession
if __name__ == '__main__':
    '\n        Usage: decommissioning\n    '
    print('Starting decom test')
    spark = SparkSession.builder.appName('DecomTest').getOrCreate()
    sc = spark._sc
    acc = sc.accumulator(0)

    def addToAcc(x):
        if False:
            while True:
                i = 10
        acc.add(1)
        return x
    initialRdd = sc.parallelize(range(100), 5)
    accRdd = initialRdd.map(addToAcc)
    rdd = accRdd.map(lambda x: (x, x)).groupByKey()
    for i in range(1, 2):
        shuffleRdd = sc.parallelize(range(1, 10), 5).map(lambda x: (x, x)).groupByKey()
        shuffleRdd.collect()
    rdd.collect()
    print('1st accumulator value is: ' + str(acc.value))
    print('Waiting to give nodes time to finish migration, decom exec 1.')
    print('...')
    time.sleep(30)
    rdd.count()
    rdd.collect()
    print('Final accumulator value is: ' + str(acc.value))
    print('Finished waiting, stopping Spark.')
    spark.stop()
    print('Done, exiting Python')
    sys.exit(0)