from dagster import job, op, OpExecutionContext

@op(config_schema={'date': str})
def process_data_for_date(context: OpExecutionContext):
    if False:
        print('Hello World!')
    date = context.op_config['date']
    context.log.info(f'processing data for {date}')
from dagster import daily_partitioned_config
from datetime import datetime

@daily_partitioned_config(start_date=datetime(2020, 1, 1))
def my_partitioned_config(start: datetime, _end: datetime):
    if False:
        print('Hello World!')
    return {'ops': {'process_data_for_date': {'config': {'date': start.strftime('%Y-%m-%d')}}}}

@job(config=my_partitioned_config)
def do_stuff_partitioned():
    if False:
        print('Hello World!')
    process_data_for_date()