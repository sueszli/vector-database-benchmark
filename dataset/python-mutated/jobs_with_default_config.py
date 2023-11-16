from dagster import Config, OpExecutionContext, RunConfig, job, op

class DoSomethingConfig(Config):
    config_param: str

@op
def do_something(context: OpExecutionContext, config: DoSomethingConfig):
    if False:
        while True:
            i = 10
    context.log.info('config_param: ' + config.config_param)
default_config = RunConfig(ops={'do_something': DoSomethingConfig(config_param='stuff')})

@job(config=default_config)
def do_it_all_with_default_config():
    if False:
        for i in range(10):
            print('nop')
    do_something()
if __name__ == '__main__':
    do_it_all_with_default_config.execute_in_process()