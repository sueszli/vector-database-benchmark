"""Pyspark Hudi example."""
import sys
from pyspark.sql import SparkSession

def create_hudi_table(spark, table_name, table_uri):
    if False:
        for i in range(10):
            print('nop')
    'Creates Hudi table.'
    create_table_sql = f"\n        CREATE TABLE IF NOT EXISTS {table_name} (\n            uuid string,\n            begin_lat double,\n            begin_lon double,\n            end_lat double,\n            end_lon double,\n            driver string,\n            rider string,\n            fare double,\n            partitionpath string,\n            ts long\n        ) USING hudi\n        LOCATION '{table_uri}'\n        TBLPROPERTIES (\n            type = 'cow',\n            primaryKey = 'uuid',\n            preCombineField = 'ts'\n        )\n        PARTITIONED BY (partitionpath)\n    "
    spark.sql(create_table_sql)

def delete_hudi_table(spark, table_name):
    if False:
        for i in range(10):
            print('nop')
    'Deletes Hudi table.'
    spark.sql(f'DROP TABLE IF EXISTS {table_name}')

def generate_test_dataframe(spark, n_rows):
    if False:
        print('Hello World!')
    "Generates test dataframe with Hudi's built-in data generator."
    spark_context = spark.sparkContext
    utils = spark_context._jvm.org.apache.hudi.QuickstartUtils
    data_generator = utils.DataGenerator()
    inserts = utils.convertToStringList(data_generator.generateInserts(n_rows))
    return spark.read.json(spark_context.parallelize(inserts, 2))

def write_hudi_table(name, uri, dataframe):
    if False:
        while True:
            i = 10
    'Writes Hudi table.'
    options = {'hoodie.table.name': name, 'hoodie.datasource.write.recordkey.field': 'uuid', 'hoodie.datasource.write.partitionpath.field': 'partitionpath', 'hoodie.datasource.write.table.name': name, 'hoodie.datasource.write.operation': 'upsert', 'hoodie.datasource.write.precombine.field': 'ts', 'hoodie.upsert.shuffle.parallelism': 2, 'hoodie.insert.shuffle.parallelism': 2}
    dataframe.write.format('hudi').options(**options).mode('append').save(uri)

def query_commit_history(spark, name, uri):
    if False:
        return 10
    'Query commit history.'
    tmp_table = f'{name}_commit_history'
    spark.read.format('hudi').load(uri).createOrReplaceTempView(tmp_table)
    query = f'\n        SELECT DISTINCT(_hoodie_commit_time)\n        FROM {tmp_table}\n        ORDER BY _hoodie_commit_time\n        DESC\n    '
    return spark.sql(query)

def read_hudi_table(spark, table_name, table_uri, commit_ts=''):
    if False:
        for i in range(10):
            print('nop')
    'Reads Hudi table at the given commit timestamp.'
    if commit_ts:
        options = {'as.of.instant': commit_ts}
    else:
        options = {}
    tmp_table = f'{table_name}_snapshot'
    spark.read.format('hudi').options(**options).load(table_uri).createOrReplaceTempView(tmp_table)
    query = f'\n        SELECT _hoodie_commit_time, begin_lat, begin_lon,\n                driver, end_lat, end_lon, fare, partitionpath,\n                rider, ts, uuid\n        FROM {tmp_table}\n    '
    return spark.sql(query)

def main():
    if False:
        print('Hello World!')
    'Test create write and read Hudi table.'
    if len(sys.argv) != 3:
        raise Exception('Expected arguments: <table_name> <table_uri>')
    table_name = sys.argv[1]
    table_uri = sys.argv[2]
    app_name = f'pyspark-hudi-test_{table_name}'
    print(f'Creating Spark session {app_name} ...')
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print(f'Creating Hudi table {table_name} at {table_uri} ...')
    create_hudi_table(spark, table_name, table_uri)
    print('Generating test data batch 1...')
    n_rows1 = 10
    input_df1 = generate_test_dataframe(spark, n_rows1)
    input_df1.show(truncate=False)
    print('Writing Hudi table, batch 1 ...')
    write_hudi_table(table_name, table_uri, input_df1)
    print('Generating test data batch 2...')
    n_rows2 = 10
    input_df2 = generate_test_dataframe(spark, n_rows2)
    input_df2.show(truncate=False)
    print('Writing Hudi table, batch 2 ...')
    write_hudi_table(table_name, table_uri, input_df2)
    print('Querying commit history ...')
    commits_df = query_commit_history(spark, table_name, table_uri)
    commits_df.show(truncate=False)
    previous_commit = commits_df.collect()[1]._hoodie_commit_time
    print('Reading the Hudi table snapshot at the latest commit ...')
    output_df1 = read_hudi_table(spark, table_name, table_uri)
    output_df1.show(truncate=False)
    print(f'Reading the Hudi table snapshot at {previous_commit} ...')
    output_df2 = read_hudi_table(spark, table_name, table_uri, previous_commit)
    output_df2.show(truncate=False)
    print('Deleting Hudi table ...')
    delete_hudi_table(spark, table_name)
    print('Stopping Spark session ...')
    spark.stop()
    print('All done')
main()