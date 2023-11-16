from dagster import AssetExecutionContext, AssetKey, BackfillPolicy, DailyPartitionsDefinition, asset

@asset(partitions_def=DailyPartitionsDefinition(start_date='2020-01-01'), backfill_policy=BackfillPolicy.single_run(), deps=[AssetKey('raw_events')])
def events(context: AssetExecutionContext):
    if False:
        return 10
    (start_datetime, end_datetime) = context.partition_time_window
    input_data = read_data_in_datetime_range(start_datetime, end_datetime)
    output_data = compute_events_from_raw_events(input_data)
    overwrite_data_in_datetime_range(start_datetime, end_datetime, output_data)

def compute_events_from_raw_events(*args):
    if False:
        while True:
            i = 10
    ...

def read_data_in_datetime_range(*args):
    if False:
        for i in range(10):
            print('nop')
    ...

def overwrite_data_in_datetime_range(*args):
    if False:
        return 10
    ...