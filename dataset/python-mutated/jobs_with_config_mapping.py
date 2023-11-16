from dagster import Config, OpExecutionContext, RunConfig, config_mapping, job, op

class DoSomethingConfig(Config):
    config_param: str

@op
def do_something(context: OpExecutionContext, config: DoSomethingConfig) -> None:
    if False:
        return 10
    context.log.info('config_param: ' + config.config_param)

class SimplifiedConfig(Config):
    simplified_param: str

@config_mapping
def simplified_config(val: SimplifiedConfig) -> RunConfig:
    if False:
        while True:
            i = 10
    return RunConfig(ops={'do_something': DoSomethingConfig(config_param=val.simplified_param)})

@job(config=simplified_config)
def do_it_all_with_simplified_config():
    if False:
        print('Hello World!')
    do_something()
if __name__ == '__main__':
    do_it_all_with_simplified_config.execute_in_process(run_config={'simplified_param': 'stuff'})