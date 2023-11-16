from dagster import Field, OpExecutionContext, configured, op

@op(config_schema={'iterations': int, 'word': Field(str, is_required=False, default_value='hello')})
def example(context: OpExecutionContext):
    if False:
        print('Hello World!')
    for _ in range(context.op_config['iterations']):
        context.log.info(context.op_config['word'])
configured_example = configured(example, name='configured_example')({'iterations': 6, 'word': 'wheaties'})

@configured(example, int)
def another_configured_example(config):
    if False:
        return 10
    return {'iterations': config, 'word': 'wheaties'}