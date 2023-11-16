import pytest
from bigdl.orca.data import spark_df_to_ray_dataset
from bigdl.orca import OrcaContext
from pyspark.sql import SparkSession

def test_spark_df_to_ray_dataset(orca_context_fixture):
    if False:
        return 10
    sc = OrcaContext.get_spark_context()
    spark = SparkSession(sc)
    spark_df = spark.createDataFrame([(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')], ['one', 'two'])
    rows = [(r.one, r.two) for r in spark_df.take(4)]
    ds = spark_df_to_ray_dataset(spark_df)
    values = [(r['one'], r['two']) for r in ds.take(8)]
    assert values == rows
if __name__ == '__main__':
    pytest.main([__file__])