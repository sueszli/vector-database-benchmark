import argparse

def spark_streaming_to_pubsublite(project_number: int, location: str, topic_id: str) -> None:
    if False:
        return 10
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import array, create_map, col, lit, when
    from pyspark.sql.types import BinaryType, StringType
    import uuid
    spark = SparkSession.builder.appName('write-app').getOrCreate()
    sdf = spark.readStream.format('rate').option('rowsPerSecond', 1).load()
    sdf = sdf.withColumn('key', lit('example').cast(BinaryType())).withColumn('data', col('value').cast(StringType()).cast(BinaryType())).withColumnRenamed('timestamp', 'event_timestamp').withColumn('attributes', create_map(lit('key1'), array(when(col('value') % 2 == 0, b'even').otherwise(b'odd')))).drop('value')
    sdf.printSchema()
    query = sdf.writeStream.format('pubsublite').option('pubsublite.topic', f'projects/{project_number}/locations/{location}/topics/{topic_id}').option('checkpointLocation', '/tmp/app' + uuid.uuid4().hex).outputMode('append').trigger(processingTime='1 second').start()
    query.awaitTermination(60)
    query.stop()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_number', help='Google Cloud Project Number')
    parser.add_argument('--location', help='Your Cloud location, e.g. us-central1-a')
    parser.add_argument('--topic_id', help='Your Pub/Sub Lite topic ID')
    args = parser.parse_args()
    spark_streaming_to_pubsublite(args.project_number, args.location, args.topic_id)