from __future__ import annotations
import pendulum
import requests
from airflow.decorators import dag, task
from airflow.io.store.path import ObjectStoragePath
API = 'https://opendata.fmi.fi/timeseries'
aq_fields = {'fmisid': 'int32', 'time': 'datetime64[ns]', 'AQINDEX_PT1H_avg': 'float64', 'PM10_PT1H_avg': 'float64', 'PM25_PT1H_avg': 'float64', 'O3_PT1H_avg': 'float64', 'CO_PT1H_avg': 'float64', 'SO2_PT1H_avg': 'float64', 'NO2_PT1H_avg': 'float64', 'TRSC_PT1H_avg': 'float64'}
base = ObjectStoragePath('s3://airflow-tutorial-data/', conn_id='aws_default')

@dag(schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example'])
def tutorial_objectstorage():
    if False:
        for i in range(10):
            print('nop')
    '\n    ### Object Storage Tutorial Documentation\n    This is a tutorial DAG to showcase the usage of the Object Storage API.\n    Documentation that goes along with the Airflow Object Storage tutorial is\n    located\n    [here](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/objectstorage.html)\n    '
    import duckdb
    import pandas as pd

    @task
    def get_air_quality_data(**kwargs) -> ObjectStoragePath:
        if False:
            i = 10
            return i + 15
        "\n        #### Get Air Quality Data\n        This task gets air quality data from the Finnish Meteorological Institute's\n        open data API. The data is saved as parquet.\n        "
        execution_date = kwargs['logical_date']
        start_time = kwargs['data_interval_start']
        params = {'format': 'json', 'precision': 'double', 'groupareas': '0', 'producer': 'airquality_urban', 'area': 'Uusimaa', 'param': ','.join(aq_fields.keys()), 'starttime': start_time.isoformat(timespec='seconds'), 'endtime': execution_date.isoformat(timespec='seconds'), 'tz': 'UTC'}
        response = requests.get(API, params=params)
        response.raise_for_status()
        base.mkdir(exists_ok=True)
        formatted_date = execution_date.format('YYYYMMDD')
        path = base / f'air_quality_{formatted_date}.parquet'
        df = pd.DataFrame(response.json()).astype(aq_fields)
        with path.open('wb') as file:
            df.to_parquet(file)
        return path

    @task
    def analyze(path: ObjectStoragePath, **kwargs):
        if False:
            while True:
                i = 10
        '\n        #### Analyze\n        This task analyzes the air quality data, prints the results\n        '
        conn = duckdb.connect(database=':memory:')
        conn.register_filesystem(path.fs)
        conn.execute(f"CREATE OR REPLACE TABLE airquality_urban AS SELECT * FROM read_parquet('{path}')")
        df2 = conn.execute('SELECT * FROM airquality_urban').fetchdf()
        print(df2.head())
    obj_path = get_air_quality_data()
    analyze(obj_path)
tutorial_objectstorage()