import operator
import sys
from pyspark.sql import SparkSession

def main(argv):
    if False:
        i = 10
        return i + 15
    input_paths = argv[1].split(',')
    output_path = argv[2]
    spark = SparkSession.builder.getOrCreate()
    streams = spark.read.option('sep', '\t').csv(input_paths[0])
    for stream_path in input_paths[1:]:
        streams.union(spark.read.option('sep', '\t').csv(stream_path))
    counts = streams.map(lambda row: (row[1], 1)).reduceByKey(operator.add)
    counts.write.option('sep', '\t').csv(output_path)
if __name__ == '__main__':
    sys.exit(main(sys.argv))