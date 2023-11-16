import argparse

def spark_streaming_from_pubsublite(project_number: int, location: str, subscription_id: str) -> None:
    if False:
        return 10
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StringType
    spark = SparkSession.builder.appName('read-app').master('yarn').getOrCreate()
    sdf = spark.readStream.format('pubsublite').option('pubsublite.subscription', f'projects/{project_number}/locations/{location}/subscriptions/{subscription_id}').load()
    sdf = sdf.withColumn('data', sdf.data.cast(StringType()))
    query = sdf.writeStream.format('console').outputMode('append').trigger(processingTime='1 second').start()
    query.awaitTermination(120)
    query.stop()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_number', help='Google Cloud Project Number')
    parser.add_argument('--location', help='Your Cloud location, e.g. us-central1-a')
    parser.add_argument('--subscription_id', help='Your Pub/Sub Lite subscription ID')
    args = parser.parse_args()
    spark_streaming_from_pubsublite(args.project_number, args.location, args.subscription_id)