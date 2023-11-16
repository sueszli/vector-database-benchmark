from dagster import Config, OpExecutionContext, job, op, static_partitioned_config
CONTINENTS = ['Africa', 'Antarctica', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']

@static_partitioned_config(partition_keys=CONTINENTS)
def continent_config(partition_key: str):
    if False:
        print('Hello World!')
    return {'ops': {'continent_op': {'config': {'continent_name': partition_key}}}}

class ContinentOpConfig(Config):
    continent_name: str

@op
def continent_op(context: OpExecutionContext, config: ContinentOpConfig):
    if False:
        while True:
            i = 10
    context.log.info(config.continent_name)

@job(config=continent_config)
def continent_job():
    if False:
        for i in range(10):
            print('nop')
    continent_op()