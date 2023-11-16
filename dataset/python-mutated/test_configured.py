from dagster import job, op, resource

def test_configured_ops_and_resources():
    if False:
        print('Hello World!')

    @op(config_schema={'greeting': str}, required_resource_keys={'animal', 'plant'})
    def emit_greet_creature(context):
        if False:
            i = 10
            return i + 15
        greeting = context.op_config['greeting']
        return f'{greeting}, {context.resources.animal}, {context.resources.plant}'
    emit_greet_salutation = emit_greet_creature.configured({'greeting': 'salutation'}, 'emit_greet_salutation')
    emit_greet_howdy = emit_greet_creature.configured({'greeting': 'howdy'}, 'emit_greet_howdy')

    @resource(config_schema={'creature': str})
    def emit_creature(context):
        if False:
            i = 10
            return i + 15
        return context.resource_config['creature']

    @job(resource_defs={'animal': emit_creature.configured({'creature': 'dog'}), 'plant': emit_creature.configured({'creature': 'tree'})})
    def myjob():
        if False:
            while True:
                i = 10
        return (emit_greet_salutation(), emit_greet_howdy())
    result = myjob.execute_in_process()
    assert result.success