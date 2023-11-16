from dagster import AssetMaterialization, Output, op
from dagster._annotations import experimental
from dagster._utils.test import wrap_op_in_graph_and_execute

def test_generator_return_op():
    if False:
        print('Hello World!')

    def _gen():
        if False:
            print('Hello World!')
        yield Output('done')

    @op
    def gen_ret_op(_):
        if False:
            print('Hello World!')
        return _gen()
    result = wrap_op_in_graph_and_execute(gen_ret_op)
    assert result.output_value() == 'done'

def test_generator_yield_op():
    if False:
        for i in range(10):
            print('nop')

    def _gen():
        if False:
            return 10
        yield Output('done')

    @op
    def gen_yield_op(_):
        if False:
            while True:
                i = 10
        for event in _gen():
            yield event
    result = wrap_op_in_graph_and_execute(gen_yield_op)
    assert result.output_value() == 'done'

def test_generator_yield_from_op():
    if False:
        print('Hello World!')

    def _gen():
        if False:
            return 10
        yield Output('done')

    @op
    def gen_yield_op(_):
        if False:
            while True:
                i = 10
        yield from _gen()
    result = wrap_op_in_graph_and_execute(gen_yield_op)
    assert result.output_value() == 'done'

def test_nested_generator_op():
    if False:
        for i in range(10):
            print('nop')

    def _gen1():
        if False:
            while True:
                i = 10
        yield AssetMaterialization('test')

    def _gen2():
        if False:
            while True:
                i = 10
        yield Output('done')

    def _gen():
        if False:
            for i in range(10):
                print('nop')
        yield from _gen1()
        yield from _gen2()

    @op
    def gen_return_op(_):
        if False:
            while True:
                i = 10
        return _gen()
    result = wrap_op_in_graph_and_execute(gen_return_op)
    assert result.output_value() == 'done'

def test_experimental_generator_op():
    if False:
        i = 10
        return i + 15

    @op
    @experimental
    def gen_op():
        if False:
            while True:
                i = 10
        yield Output('done')
    result = wrap_op_in_graph_and_execute(gen_op)
    assert result.output_value() == 'done'