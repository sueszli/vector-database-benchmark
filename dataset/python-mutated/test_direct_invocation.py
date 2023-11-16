import asyncio
import pytest
from dagster import ConfigurableResource, asset, op
from dagster._core.errors import DagsterInvalidInvocationError, DagsterInvariantViolationError
from dagster._core.execution.context.invocation import build_op_context

def test_direct_op_invocation() -> None:
    if False:
        i = 10
        return i + 15

    class MyResource(ConfigurableResource):
        a_str: str

    @op
    def my_op(context, my_resource: MyResource) -> str:
        if False:
            i = 10
            return i + 15
        assert my_resource.a_str == 'foo'
        return my_resource.a_str
    assert my_op(build_op_context(resources={'my_resource': MyResource(a_str='foo')})) == 'foo'
    with pytest.raises(DagsterInvalidInvocationError, match='Cannot provide resources in both context and kwargs'):
        assert my_op(context=build_op_context(resources={'my_resource': MyResource(a_str='foo')}), my_resource=MyResource(a_str='foo')) == 'foo'
    assert my_op(context=build_op_context(), my_resource=MyResource(a_str='foo')) == 'foo'
    assert my_op(build_op_context(), MyResource(a_str='foo')) == 'foo'

    @op
    def my_op_no_context(my_resource: MyResource) -> str:
        if False:
            i = 10
            return i + 15
        assert my_resource.a_str == 'foo'
        return my_resource.a_str
    assert my_op_no_context(build_op_context(resources={'my_resource': MyResource(a_str='foo')})) == 'foo'
    assert my_op_no_context(my_resource=MyResource(a_str='foo')) == 'foo'

def test_direct_op_invocation_multiple_resources() -> None:
    if False:
        return 10

    class MyResource(ConfigurableResource):
        a_str: str

    @op
    def my_op(context, my_resource: MyResource, my_other_resource: MyResource) -> str:
        if False:
            print('Hello World!')
        assert my_resource.a_str == 'foo'
        assert my_other_resource.a_str == 'bar'
        return my_resource.a_str
    assert my_op(build_op_context(resources={'my_resource': MyResource(a_str='foo'), 'my_other_resource': MyResource(a_str='bar')})) == 'foo'
    assert my_op(context=build_op_context(), my_resource=MyResource(a_str='foo'), my_other_resource=MyResource(a_str='bar')) == 'foo'

    @op
    def my_op_no_context(my_resource: MyResource, my_other_resource: MyResource) -> str:
        if False:
            while True:
                i = 10
        assert my_resource.a_str == 'foo'
        assert my_other_resource.a_str == 'bar'
        return my_resource.a_str
    assert my_op_no_context(build_op_context(resources={'my_resource': MyResource(a_str='foo'), 'my_other_resource': MyResource(a_str='bar')})) == 'foo'
    assert my_op_no_context(my_resource=MyResource(a_str='foo'), my_other_resource=MyResource(a_str='bar')) == 'foo'

def test_direct_op_invocation_with_inputs() -> None:
    if False:
        i = 10
        return i + 15

    class MyResource(ConfigurableResource):
        z: int

    @op
    def my_wacky_addition_op(context, x: int, y: int, my_resource: MyResource) -> int:
        if False:
            for i in range(10):
                print('nop')
        return x + y + my_resource.z
    assert my_wacky_addition_op(build_op_context(resources={'my_resource': MyResource(z=2)}), 4, 5) == 11
    assert my_wacky_addition_op(build_op_context(resources={'my_resource': MyResource(z=3)}), y=1, x=2) == 6
    assert my_wacky_addition_op(build_op_context(), 10, 20, my_resource=MyResource(z=30)) == 60
    assert my_wacky_addition_op(build_op_context(), y=1, x=2, my_resource=MyResource(z=3)) == 6

    @op
    def my_wacky_addition_op_no_context(x: int, y: int, my_resource: MyResource) -> int:
        if False:
            while True:
                i = 10
        return x + y + my_resource.z
    assert my_wacky_addition_op_no_context(build_op_context(resources={'my_resource': MyResource(z=2)}), 4, 5) == 11
    assert my_wacky_addition_op_no_context(build_op_context(resources={'my_resource': MyResource(z=3)}), y=1, x=2) == 6
    assert my_wacky_addition_op_no_context(10, 20, my_resource=MyResource(z=30)) == 60
    assert my_wacky_addition_op_no_context(y=1, x=2, my_resource=MyResource(z=3)) == 6

    @op
    def my_wacky_addition_op_resource_first(my_resource: MyResource, x: int, y: int) -> int:
        if False:
            i = 10
            return i + 15
        return x + y + my_resource.z
    assert my_wacky_addition_op_resource_first(build_op_context(resources={'my_resource': MyResource(z=2)}), x=4, y=5) == 11
    assert my_wacky_addition_op_resource_first(MyResource(z=2), 45, 53) == 100

def test_direct_asset_invocation() -> None:
    if False:
        return 10

    class MyResource(ConfigurableResource):
        a_str: str

    @asset
    def my_asset(context, my_resource: MyResource) -> str:
        if False:
            i = 10
            return i + 15
        assert my_resource.a_str == 'foo'
        return my_resource.a_str
    assert my_asset(build_op_context(resources={'my_resource': MyResource(a_str='foo')})) == 'foo'
    with pytest.raises(DagsterInvalidInvocationError, match='Cannot provide resources in both context and kwargs'):
        assert my_asset(context=build_op_context(resources={'my_resource': MyResource(a_str='foo')}), my_resource=MyResource(a_str='foo')) == 'foo'
    assert my_asset(context=build_op_context(), my_resource=MyResource(a_str='foo')) == 'foo'
    assert my_asset(build_op_context(), MyResource(a_str='foo')) == 'foo'

    @asset
    def my_asset_no_context(my_resource: MyResource) -> str:
        if False:
            while True:
                i = 10
        assert my_resource.a_str == 'foo'
        return my_resource.a_str
    assert my_asset_no_context(build_op_context(resources={'my_resource': MyResource(a_str='foo')})) == 'foo'
    assert my_asset_no_context(my_resource=MyResource(a_str='foo')) == 'foo'

def test_direct_asset_invocation_with_inputs() -> None:
    if False:
        while True:
            i = 10

    class MyResource(ConfigurableResource):
        z: int

    @asset
    def my_wacky_addition_asset(context, x: int, y: int, my_resource: MyResource) -> int:
        if False:
            print('Hello World!')
        return x + y + my_resource.z
    assert my_wacky_addition_asset(build_op_context(resources={'my_resource': MyResource(z=2)}), 4, 5) == 11
    assert my_wacky_addition_asset(build_op_context(resources={'my_resource': MyResource(z=3)}), y=1, x=2) == 6
    assert my_wacky_addition_asset(build_op_context(), 10, 20, my_resource=MyResource(z=30)) == 60
    assert my_wacky_addition_asset(build_op_context(), y=1, x=2, my_resource=MyResource(z=3)) == 6

    @asset
    def my_wacky_addition_asset_no_context(x: int, y: int, my_resource: MyResource) -> int:
        if False:
            for i in range(10):
                print('nop')
        return x + y + my_resource.z
    assert my_wacky_addition_asset_no_context(build_op_context(resources={'my_resource': MyResource(z=2)}), 4, 5) == 11
    assert my_wacky_addition_asset_no_context(build_op_context(resources={'my_resource': MyResource(z=3)}), y=1, x=2) == 6
    assert my_wacky_addition_asset_no_context(10, 20, my_resource=MyResource(z=30)) == 60
    assert my_wacky_addition_asset_no_context(y=1, x=2, my_resource=MyResource(z=3)) == 6

def test_direct_op_invocation_plain_arg_with_resource_definition_no_inputs_no_context() -> None:
    if False:
        i = 10
        return i + 15

    class NumResource(ConfigurableResource):
        num: int
    executed = {}

    @op
    def an_op(my_resource: NumResource) -> None:
        if False:
            print('Hello World!')
        assert my_resource.num == 1
        executed['yes'] = True
    an_op(NumResource(num=1))
    assert executed['yes']

def test_direct_op_invocation_kwarg_with_resource_definition_no_inputs_no_context() -> None:
    if False:
        while True:
            i = 10

    class NumResource(ConfigurableResource):
        num: int
    executed = {}

    @op
    def an_op(my_resource: NumResource) -> None:
        if False:
            while True:
                i = 10
        assert my_resource.num == 1
        executed['yes'] = True
    an_op(my_resource=NumResource(num=1))
    assert executed['yes']

def test_direct_asset_invocation_plain_arg_with_resource_definition_no_inputs_no_context() -> None:
    if False:
        for i in range(10):
            print('nop')

    class NumResource(ConfigurableResource):
        num: int
    executed = {}

    @asset
    def an_asset(my_resource: NumResource) -> None:
        if False:
            i = 10
            return i + 15
        assert my_resource.num == 1
        executed['yes'] = True
    an_asset(NumResource(num=1))
    assert executed['yes']

def test_direct_asset_invocation_kwarg_with_resource_definition_no_inputs_no_context() -> None:
    if False:
        while True:
            i = 10

    class NumResource(ConfigurableResource):
        num: int
    executed = {}

    @asset
    def an_asset(my_resource: NumResource) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert my_resource.num == 1
        executed['yes'] = True
    an_asset(my_resource=NumResource(num=1))
    assert executed['yes']

def test_direct_asset_invocation_many_resource_args() -> None:
    if False:
        return 10

    class NumResource(ConfigurableResource):
        num: int
    executed = {}

    @asset
    def an_asset(my_resource: NumResource, my_other_resource: NumResource) -> None:
        if False:
            while True:
                i = 10
        assert my_resource.num == 1
        assert my_other_resource.num == 2
        executed['yes'] = True
    an_asset(NumResource(num=1), NumResource(num=2))
    assert executed['yes']
    executed.clear()
    an_asset(my_resource=NumResource(num=1), my_other_resource=NumResource(num=2))
    assert executed['yes']
    executed.clear()
    an_asset(my_other_resource=NumResource(num=2), my_resource=NumResource(num=1))
    assert executed['yes']
    executed.clear()
    an_asset(NumResource(num=1), my_other_resource=NumResource(num=2))
    assert executed['yes']

def test_direct_asset_invocation_many_resource_args_context() -> None:
    if False:
        return 10

    class NumResource(ConfigurableResource):
        num: int
    executed = {}

    @asset
    def an_asset(context, my_resource: NumResource, my_other_resource: NumResource) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert context.resources.my_resource.num == 1
        assert context.resources.my_other_resource.num == 2
        assert my_resource.num == 1
        assert my_other_resource.num == 2
        executed['yes'] = True
    an_asset(build_op_context(), NumResource(num=1), NumResource(num=2))
    assert executed['yes']
    executed.clear()
    an_asset(build_op_context(), my_resource=NumResource(num=1), my_other_resource=NumResource(num=2))
    assert executed['yes']
    executed.clear()
    an_asset(my_other_resource=NumResource(num=2), my_resource=NumResource(num=1), context=build_op_context())
    assert executed['yes']
    executed.clear()

def test_direct_invocation_output_metadata():
    if False:
        return 10

    @asset
    def my_asset(context):
        if False:
            return 10
        context.add_output_metadata({'foo': 'bar'})

    @asset
    def my_other_asset(context):
        if False:
            return 10
        context.add_output_metadata({'baz': 'qux'})
    ctx = build_op_context()
    my_asset(ctx)
    assert ctx.get_output_metadata('result') == {'foo': 'bar'}
    with pytest.raises(DagsterInvariantViolationError, match="attempted to log metadata for output 'result' more than once"):
        my_other_asset(ctx)

def test_async_assets_with_shared_context():
    if False:
        print('Hello World!')

    @asset
    async def async_asset_one(context):
        assert context.asset_key.to_user_string() == 'async_asset_one'
        await asyncio.sleep(0.01)
        return 'one'

    @asset
    async def async_asset_two(context):
        assert context.asset_key.to_user_string() == 'async_asset_two'
        await asyncio.sleep(0.01)
        return 'two'
    ctx = build_op_context()

    async def main():
        return await asyncio.gather(async_asset_one(ctx), async_asset_two(ctx))
    result = asyncio.run(main())
    assert result[0] == 'one'
    assert result[1] == 'two'