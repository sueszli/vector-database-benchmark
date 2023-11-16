import resource
import sys
from pyspark.sql import SparkSession
if __name__ == '__main__':
    '\n        Usage: worker_memory_check [Memory_in_Mi]\n    '
    spark = SparkSession.builder.appName('PyMemoryTest').getOrCreate()
    sc = spark.sparkContext
    if len(sys.argv) < 2:
        print('Usage: worker_memory_check [Memory_in_Mi]', file=sys.stderr)
        sys.exit(-1)

    def f(x):
        if False:
            while True:
                i = 10
        rLimit = resource.getrlimit(resource.RLIMIT_AS)
        print('RLimit is ' + str(rLimit))
        return rLimit
    resourceValue = sc.parallelize([1]).map(f).collect()[0][0]
    print('Resource Value is ' + str(resourceValue))
    truthCheck = resourceValue == int(sys.argv[1])
    print('PySpark Worker Memory Check is: ' + str(truthCheck))
    spark.stop()