import time
from dagster import AssetKey, IOManager, IOManagerDefinition, SourceAsset, asset, with_resources
sfo_q2_weather_sample = SourceAsset(key=AssetKey('sfo_q2_weather_sample'))

class DataFrame:
    pass

class DummyIOManager(IOManager):

    def handle_output(self, context, obj: DataFrame):
        if False:
            return 10
        assert context
        assert obj

    def load_input(self, context):
        if False:
            print('Hello World!')
        assert context
        return DataFrame()

@asset
def daily_temperature_highs(sfo_q2_weather_sample: DataFrame) -> DataFrame:
    if False:
        for i in range(10):
            print('nop')
    'Computes the temperature high for each day.'
    assert sfo_q2_weather_sample
    time.sleep(3)
    return DataFrame()

@asset
def hottest_dates(daily_temperature_highs: DataFrame) -> DataFrame:
    if False:
        while True:
            i = 10
    "Computes the 10 hottest dates.\n\n    In a more advanced demo, this might perform a complex SQL query to aggregate the data. For now,\n    just imagine that this implements something like:\n\n    ```sql\n    SELECT temp, date_part('day', date) FROM daily_temperature_highs ORDER BY date DESC;\n    ```\n\n    This could make use of [DATE_PART](https://www.postgresql.org/docs/8.1/functions-datetime.html),\n    and we can even link to that because this supports Markdown.\n\n    This concludes the demo of a long asset description.\n    "
    assert daily_temperature_highs
    time.sleep(3)
    return DataFrame()
software_defined_assets = with_resources([daily_temperature_highs, hottest_dates, sfo_q2_weather_sample], resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(DummyIOManager())})