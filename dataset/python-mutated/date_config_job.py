from dagster import Config, OpExecutionContext, job, op

class ProcessDateConfig(Config):
    date: str

@op
def process_data_for_date(context: OpExecutionContext, config: ProcessDateConfig):
    if False:
        return 10
    date = config.date
    context.log.info(f'processing data for {date}')

@job
def do_stuff():
    if False:
        i = 10
        return i + 15
    process_data_for_date()