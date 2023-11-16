import sys
import pandas as pd
from py4j.protocol import Py4JJavaError
from pyspark.sql import SparkSession
import pyspark.sql.functions as functions
if __name__ == '__main__':
    BUCKET_NAME = sys.argv[1]
    READ_TABLE = sys.argv[2]
    DF_WRITE_TABLE = sys.argv[3]
    PRCP_MEAN_WRITE_TABLE = sys.argv[4]
    SNOW_MEAN_WRITE_TABLE = sys.argv[5]
    PHX_PRCP_WRITE_TABLE = sys.argv[6]
    PHX_SNOW_WRITE_TABLE = sys.argv[7]
    spark = SparkSession.builder.appName('data_processing').getOrCreate()
    try:
        df = spark.read.format('bigquery').load(READ_TABLE)
    except Py4JJavaError:
        raise Exception(f'Error reading {READ_TABLE}')
    western_states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
    df = df.where(df.STATE.isin(western_states))
    df.show(n=10)
    print('After state filtering, # of rows remaining is:', df.count())
    df = df.where(df.ELEMENT.isin(['PRCP', 'SNOW']))
    df.show(n=10)
    print('After element filtering, # of rows remaining is:', df.count())
    df = df.withColumn('VALUE', df.VALUE / 10)
    df.show()
    df = df.withColumn('DATE', functions.year(df.DATE)).withColumnRenamed('DATE', 'YEAR')
    prcp_mean_df = df.where(df.ELEMENT == 'PRCP').groupBy('YEAR').agg(functions.avg('VALUE').alias('ANNUAL_PRCP_MEAN')).sort('YEAR')
    print('PRCP mean table')
    prcp_mean_df.show(n=50)
    snow_mean_df = df.where(df.ELEMENT == 'SNOW').groupBy('YEAR').agg(functions.avg('VALUE').alias('ANNUAL_SNOW_MEAN')).sort('YEAR')
    print('SNOW mean table')
    snow_mean_df.show(n=50)
    states_near_phx = ['AZ', 'CA', 'CO', 'NM', 'NV', 'UT']
    annual_df = df.where(df.STATE.isin(states_near_phx))

    @functions.pandas_udf('YEAR integer, VALUE double', functions.PandasUDFType.GROUPED_MAP)
    def phx_dw_compute(year: tuple, df: pd.DataFrame) -> pd.DataFrame:
        if False:
            print('Hello World!')
        PHX_LATITUDE = 33.4484
        PHX_LONGITUDE = -112.074
        inverse_distance_factors = 1.0 / ((PHX_LATITUDE - df.LATITUDE) ** 2 + (PHX_LONGITUDE - df.LONGITUDE) ** 2)
        weights = inverse_distance_factors / inverse_distance_factors.sum()
        return pd.DataFrame({'YEAR': year, 'VALUE': (weights * df.ANNUAL_AMOUNT).sum()})
    phx_annual_prcp_df = annual_df.where(annual_df.ELEMENT == 'PRCP').groupBy('ID', 'LATITUDE', 'LONGITUDE', 'YEAR').agg(functions.sum('VALUE').alias('ANNUAL_AMOUNT')).groupBy('YEAR').apply(phx_dw_compute)
    phx_annual_snow_df = annual_df.where(annual_df.ELEMENT == 'SNOW').groupBy('ID', 'LATITUDE', 'LONGITUDE', 'YEAR').agg(functions.sum('VALUE').alias('ANNUAL_AMOUNT')).groupBy('YEAR').apply(phx_dw_compute)
    phx_annual_prcp_df.show()
    phx_annual_snow_df.show()
    if '--dry-run' in sys.argv:
        print('Data will not be uploaded to BigQuery')
    else:
        temp_path = BUCKET_NAME
        df.write.format('bigquery').option('writeMethod', 'direct').mode('overwrite').save(DF_WRITE_TABLE)
        prcp_mean_df.write.format('bigquery').option('writeMethod', 'direct').mode('overwrite').save(PRCP_MEAN_WRITE_TABLE)
        snow_mean_df.write.format('bigquery').option('temporaryGcsBucket', temp_path).mode('overwrite').save(SNOW_MEAN_WRITE_TABLE)
        phx_annual_prcp_df.write.format('bigquery').option('writeMethod', 'direct').mode('overwrite').save(PHX_PRCP_WRITE_TABLE)
        phx_annual_snow_df.write.format('bigquery').option('writeMethod', 'direct').mode('overwrite').save(PHX_SNOW_WRITE_TABLE)
        print('Data written to BigQuery')