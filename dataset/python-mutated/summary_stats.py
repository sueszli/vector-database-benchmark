from datetime import datetime
from dagster_pandas import create_dagster_pandas_dataframe_type
from pandas import DataFrame, read_csv
from dagster import Out, file_relative_path, job, op

def compute_trip_dataframe_summary_statistics(dataframe):
    if False:
        while True:
            i = 10
    return {'min_start_time': min(dataframe['start_time']).strftime('%Y-%m-%d'), 'max_end_time': max(dataframe['end_time']).strftime('%Y-%m-%d'), 'num_unique_bikes': str(dataframe['bike_id'].nunique()), 'n_rows': len(dataframe), 'columns': str(dataframe.columns)}
SummaryStatsTripDataFrame = create_dagster_pandas_dataframe_type(name='SummaryStatsTripDataFrame', metadata_fn=compute_trip_dataframe_summary_statistics)

@op(out=Out(SummaryStatsTripDataFrame))
def load_summary_stats_trip_dataframe() -> DataFrame:
    if False:
        for i in range(10):
            print('nop')
    return read_csv(file_relative_path(__file__, './ebike_trips.csv'), parse_dates=['start_time', 'end_time'], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'), dtype={'color': 'category'})

@job
def summary_stats_trip():
    if False:
        for i in range(10):
            print('nop')
    load_summary_stats_trip_dataframe()