from dagster import Field, Noneable, op
from dagster._utils.test import wrap_op_in_graph_and_execute

def test_default_implies_not_required_field_correct():
    if False:
        i = 10
        return i + 15

    @op(config_schema={'default_to_one': Field(int, default_value=1)})
    def return_default_to_one(context):
        if False:
            while True:
                i = 10
        return context.op_config['default_to_one']
    default_to_one_field = return_default_to_one.config_schema.as_field().config_type.fields['default_to_one']
    assert default_to_one_field.is_required is False

def test_default_implies_not_required_wrap_op_in_graph_and_execute():
    if False:
        print('Hello World!')

    @op(config_schema={'default_to_one': Field(int, default_value=1)})
    def return_default_to_one(context):
        if False:
            return 10
        return context.op_config['default_to_one']
    wrap_op_in_graph_and_execute(return_default_to_one)

def test_scalar_field_defaults():
    if False:
        i = 10
        return i + 15
    assert Field(int).is_required is True
    assert Field(Noneable(int)).is_required is False
    assert Field(Noneable(int)).default_value is None

def test_noneable_shaped_field_defaults():
    if False:
        i = 10
        return i + 15
    schema = {'an_int': int}
    assert Field(schema).is_required is True
    assert Field(Noneable(schema)).is_required is False
    assert Field(Noneable(schema)).default_value is None

def test_noneable_string_in_op():
    if False:
        print('Hello World!')
    executed = {}

    @op(config_schema=Noneable(int))
    def default_noneable_int(context):
        if False:
            print('Hello World!')
        assert context.op_config is None
        executed['yes'] = True
    wrap_op_in_graph_and_execute(default_noneable_int)
    assert executed['yes']