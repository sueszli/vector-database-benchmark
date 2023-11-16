from dagster import Field, In, Int, List, OpExecutionContext, configured, job, op

@op(config_schema={'is_sample': Field(bool, is_required=False, default_value=False)}, ins={'xs': In(List[Int])})
def get_dataset(context: OpExecutionContext, xs):
    if False:
        return 10
    if context.op_config['is_sample']:
        return xs[:5]
    else:
        return xs
sample_dataset = configured(get_dataset, name='sample_dataset')({'is_sample': True})
full_dataset = configured(get_dataset, name='full_dataset')({'is_sample': False})

@job
def datasets():
    if False:
        return 10
    sample_dataset()
    full_dataset()