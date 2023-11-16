from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from definitions import benchmark_feature_service, benchmark_feature_views, driver, driver_hourly_stats_view, entity, transformed_conv_rate
from feast import FeatureStore
print('Running materialize.py')
start = datetime.now() - timedelta(days=10)
df = pd.DataFrame()
df['driver_id'] = np.arange(1000, 1010)
df['created'] = datetime.now()
df['conv_rate'] = np.arange(0, 1, 0.1)
df['acc_rate'] = np.arange(0.5, 1, 0.05)
df['avg_daily_trips'] = np.arange(0, 1000, 100)
df['event_timestamp'] = start + pd.Series(np.arange(0, 10)).map(lambda days: timedelta(days=days))
df.to_parquet('driver_stats.parquet')

def generate_data(num_rows: int, num_features: int, destination: str) -> pd.DataFrame:
    if False:
        return 10
    features = [f'feature_{i}' for i in range(num_features)]
    columns = ['entity', 'event_timestamp'] + features
    df = pd.DataFrame(0, index=np.arange(num_rows), columns=columns)
    df['event_timestamp'] = datetime.utcnow()
    for column in features:
        df[column] = np.random.randint(1, num_rows, num_rows)
    df['entity'] = 'key-' + pd.Series(np.arange(1, num_rows + 1)).astype(pd.StringDtype())
    df.to_parquet(destination)
generate_data(10 ** 3, 250, 'benchmark_data.parquet')
fs = FeatureStore('.')
fs.apply([driver_hourly_stats_view, transformed_conv_rate, driver, entity, benchmark_feature_service, *benchmark_feature_views])
now = datetime.now()
fs.materialize(start, now)
print('Materialization finished')