import pandas as pd
from dagster import AssetKey, SourceAsset, asset
from pandas import DataFrame
sfo_q2_weather_sample = SourceAsset(key=AssetKey('sfo_q2_weather_sample'), description='Weather samples, taken every five minutes at SFO', metadata={'format': 'csv'})

@asset
def daily_temperature_highs(sfo_q2_weather_sample: DataFrame) -> DataFrame:
    if False:
        while True:
            i = 10
    'Computes the temperature high for each day.'
    sfo_q2_weather_sample['valid_date'] = pd.to_datetime(sfo_q2_weather_sample['valid'])
    return sfo_q2_weather_sample.groupby('valid_date').max().rename(columns={'tmpf': 'max_tmpf'})

@asset
def hottest_dates(daily_temperature_highs: DataFrame) -> DataFrame:
    if False:
        for i in range(10):
            print('nop')
    'Computes the 10 hottest dates.'
    return daily_temperature_highs.nlargest(10, 'max_tmpf')