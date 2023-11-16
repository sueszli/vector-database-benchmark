import dagster_aws.emr.configs_spark as aws_configs_spark
import dagster_spark.configs_spark as spark_configs_spark

def test_spark_configs_same_as_in_dagster_spark():
    if False:
        for i in range(10):
            print('nop')
    aws_contents = open(aws_configs_spark.__file__, encoding='utf8').read()
    spark_contents = open(spark_configs_spark.__file__, encoding='utf8').read()
    assert aws_contents == spark_contents