import sys
from typing import List, Optional
import dagster
import pydantic
import pytest
from dagster import AssetOut, EnvVar, _check as check, asset, job, materialize, multi_asset, op, validate_run_config
from dagster._config.config_type import ConfigTypeKind, Noneable
from dagster._config.field_utils import convert_potential_field
from dagster._config.pythonic_config import Config, infer_schema_from_config_class
from dagster._config.source import BoolSource, IntSource, StringSource
from dagster._config.type_printer import print_config_type_to_string
from dagster._core.definitions.assets_job import build_assets_job
from dagster._core.definitions.definitions_class import Definitions
from dagster._core.definitions.op_definition import OpDefinition
from dagster._core.definitions.run_config import RunConfig
from dagster._core.definitions.unresolved_asset_job_definition import define_asset_job
from dagster._core.errors import DagsterInvalidConfigError, DagsterInvalidInvocationError, DagsterInvalidPythonicConfigDefinitionError
from dagster._core.execution.context.invocation import build_op_context
from dagster._core.test_utils import environ
from dagster._utils.cached_method import cached_method
from pydantic import BaseModel, Field as PyField

def test_disallow_config_schema_conflict():
    if False:
        while True:
            i = 10

    class ANewConfigOpConfig(Config):
        a_string: str
    with pytest.raises(check.ParameterCheckError):

        @op(config_schema=str)
        def a_double_config(config: ANewConfigOpConfig):
            if False:
                i = 10
                return i + 15
            pass

def test_infer_config_schema():
    if False:
        i = 10
        return i + 15
    old_schema = {'a_string': StringSource, 'an_int': IntSource}

    class ConfigClassTest(Config):
        a_string: str
        an_int: int
    assert type_string_from_config_schema(old_schema) == type_string_from_pydantic(ConfigClassTest)
    from_old_schema_field = convert_potential_field(old_schema)
    config_class_config_field = infer_schema_from_config_class(ConfigClassTest)
    assert type_string_from_config_schema(from_old_schema_field.config_type) == type_string_from_config_schema(config_class_config_field)

def type_string_from_config_schema(config_schema):
    if False:
        print('Hello World!')
    return print_config_type_to_string(convert_potential_field(config_schema).config_type)

def type_string_from_pydantic(cls):
    if False:
        i = 10
        return i + 15
    return print_config_type_to_string(infer_schema_from_config_class(cls).config_type)

def test_decorated_op_function():
    if False:
        i = 10
        return i + 15

    class ANewConfigOpConfig(Config):
        a_string: str

    @op
    def a_struct_config_op(config: ANewConfigOpConfig):
        if False:
            while True:
                i = 10
        pass

    @op(config_schema={'a_string': str})
    def an_old_config_op():
        if False:
            while True:
                i = 10
        pass
    from dagster._core.definitions.decorators.op_decorator import DecoratedOpFunction
    assert not DecoratedOpFunction(an_old_config_op).has_config_arg()
    assert DecoratedOpFunction(a_struct_config_op).has_config_arg()
    config_param = DecoratedOpFunction(a_struct_config_op).get_config_arg()
    assert config_param.name == 'config'

def test_struct_config():
    if False:
        for i in range(10):
            print('nop')

    class ANewConfigOpConfig(Config):
        a_string: str
        an_int: int
    executed = {}

    @op
    def a_struct_config_op(config: ANewConfigOpConfig):
        if False:
            while True:
                i = 10
        executed['yes'] = True
        assert config.a_string == 'foo'
        assert config.an_int == 2
    from dagster._core.definitions.decorators.op_decorator import DecoratedOpFunction
    assert DecoratedOpFunction(a_struct_config_op).has_config_arg()
    assert a_struct_config_op.config_schema.config_type.kind == ConfigTypeKind.STRICT_SHAPE
    assert list(a_struct_config_op.config_schema.config_type.fields.keys()) == ['a_string', 'an_int']

    @job
    def a_job():
        if False:
            for i in range(10):
                print('nop')
        a_struct_config_op()
    assert a_job
    from dagster._core.errors import DagsterInvalidConfigError
    with pytest.raises(DagsterInvalidConfigError):
        a_job.execute_in_process({'ops': {'a_struct_config_op': {'config': {'a_string_mispelled': 'foo', 'an_int': 2}}}})
    a_job.execute_in_process({'ops': {'a_struct_config_op': {'config': {'a_string': 'foo', 'an_int': 2}}}})
    assert executed['yes']

def test_with_assets():
    if False:
        return 10

    class AnAssetConfig(Config):
        a_string: str
        an_int: int
    executed = {}

    @asset
    def my_asset(config: AnAssetConfig):
        if False:
            i = 10
            return i + 15
        assert config.a_string == 'foo'
        assert config.an_int == 2
        executed['yes'] = True
    assert build_assets_job('blah', [my_asset], config={'ops': {'my_asset': {'config': {'a_string': 'foo', 'an_int': 2}}}}).execute_in_process().success
    assert executed['yes']

def test_multi_asset():
    if False:
        for i in range(10):
            print('nop')

    class AMultiAssetConfig(Config):
        a_string: str
        an_int: int
    executed = {}

    @multi_asset(outs={'a': AssetOut(key='asset_a'), 'b': AssetOut(key='asset_b')})
    def two_assets(config: AMultiAssetConfig):
        if False:
            print('Hello World!')
        assert config.a_string == 'foo'
        assert config.an_int == 2
        executed['yes'] = True
        return (1, 2)
    assert build_assets_job('blah', [two_assets], config={'ops': {'two_assets': {'config': {'a_string': 'foo', 'an_int': 2}}}}).execute_in_process().success
    assert executed['yes']

def test_primitive_struct_config():
    if False:
        return 10
    executed = {}

    @op
    def a_str_op(config: str):
        if False:
            print('Hello World!')
        executed['yes'] = True
        assert config == 'foo'
    from dagster._core.definitions.decorators.op_decorator import DecoratedOpFunction
    assert DecoratedOpFunction(a_str_op).has_config_arg()

    @job
    def a_job():
        if False:
            return 10
        a_str_op()
    assert a_job
    from dagster._core.errors import DagsterInvalidConfigError
    with pytest.raises(DagsterInvalidConfigError):
        a_job.execute_in_process({'ops': {'a_str_op': {'config': 1}}})
    a_job.execute_in_process({'ops': {'a_str_op': {'config': 'foo'}}})
    assert executed['yes']

    @op
    def a_bool_op(config: bool):
        if False:
            print('Hello World!')
        assert not config

    @op
    def a_int_op(config: int):
        if False:
            return 10
        assert config == 1

    @op
    def a_dict_op(config: dict):
        if False:
            i = 10
            return i + 15
        assert config == {'foo': 1}

    @op
    def a_list_op(config: list):
        if False:
            for i in range(10):
                print('nop')
        assert config == [1, 2, 3]

    @job
    def a_larger_job():
        if False:
            for i in range(10):
                print('nop')
        a_str_op()
        a_bool_op()
        a_int_op()
        a_dict_op()
        a_list_op()
    a_larger_job.execute_in_process({'ops': {'a_str_op': {'config': 'foo'}, 'a_bool_op': {'config': False}, 'a_int_op': {'config': 1}, 'a_dict_op': {'config': {'foo': 1}}, 'a_list_op': {'config': [1, 2, 3]}}})

def test_invalid_struct_config():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(DagsterInvalidPythonicConfigDefinitionError):

        class BaseModelExtendingConfig(BaseModel):
            a_string: str
            an_int: int

        @op
        def a_basemodel_config_op(config: BaseModelExtendingConfig):
            if False:
                while True:
                    i = 10
            pass

def test_nested_struct_config():
    if False:
        while True:
            i = 10

    class ANestedConfig(Config):
        a_string: str
        an_int: int

    class ANewConfigOpConfig(Config):
        a_nested_value: ANestedConfig
        a_bool: bool
    executed = {}

    @op
    def a_struct_config_op(config: ANewConfigOpConfig):
        if False:
            for i in range(10):
                print('nop')
        executed['yes'] = True
        assert config.a_nested_value.a_string == 'foo'
        assert config.a_nested_value.an_int == 2
        assert config.a_bool is True
    from dagster._core.definitions.decorators.op_decorator import DecoratedOpFunction
    assert DecoratedOpFunction(a_struct_config_op).has_config_arg()
    assert a_struct_config_op.config_schema.config_type.kind == ConfigTypeKind.STRICT_SHAPE
    assert list(a_struct_config_op.config_schema.config_type.fields.keys()) == ['a_nested_value', 'a_bool']

    @job
    def a_job():
        if False:
            i = 10
            return i + 15
        a_struct_config_op()
    assert a_job
    a_job.execute_in_process({'ops': {'a_struct_config_op': {'config': {'a_bool': True, 'a_nested_value': {'a_string': 'foo', 'an_int': 2}}}}})
    assert executed['yes']

def test_direct_op_invocation() -> None:
    if False:
        return 10

    class MyBasicOpConfig(Config):
        foo: str

    @op
    def basic_op(context, config: MyBasicOpConfig):
        if False:
            for i in range(10):
                print('nop')
        assert config.foo == 'bar'
    basic_op(build_op_context(op_config={'foo': 'bar'}))
    with pytest.raises(AssertionError):
        basic_op(build_op_context(op_config={'foo': 'qux'}))
    with pytest.raises(DagsterInvalidConfigError):
        basic_op(build_op_context(op_config={'baz': 'qux'}))

    @op
    def basic_op_no_context(config: MyBasicOpConfig):
        if False:
            for i in range(10):
                print('nop')
        assert config.foo == 'bar'
    basic_op_no_context(build_op_context(op_config={'foo': 'bar'}))
    with pytest.raises(AssertionError):
        basic_op_no_context(build_op_context(op_config={'foo': 'qux'}))
    with pytest.raises(DagsterInvalidConfigError):
        basic_op_no_context(build_op_context(op_config={'baz': 'qux'}))

def test_direct_op_invocation_complex_config() -> None:
    if False:
        i = 10
        return i + 15

    class MyBasicOpConfig(Config):
        foo: str
        bar: bool
        baz: int
        qux: List[str]

    @op
    def basic_op(context, config: MyBasicOpConfig):
        if False:
            while True:
                i = 10
        assert config.foo == 'bar'
    basic_op(build_op_context(op_config={'foo': 'bar', 'bar': True, 'baz': 1, 'qux': ['a', 'b']}))
    with pytest.raises(AssertionError):
        basic_op(build_op_context(op_config={'foo': 'qux', 'bar': True, 'baz': 1, 'qux': ['a', 'b']}))
    with pytest.raises(DagsterInvalidConfigError):
        basic_op(build_op_context(op_config={'foo': 'bar', 'bar': 'true', 'baz': 1, 'qux': ['a', 'b']}))

    @op
    def basic_op_no_context(config: MyBasicOpConfig):
        if False:
            return 10
        assert config.foo == 'bar'
    basic_op_no_context(build_op_context(op_config={'foo': 'bar', 'bar': True, 'baz': 1, 'qux': ['a', 'b']}))
    with pytest.raises(AssertionError):
        basic_op_no_context(build_op_context(op_config={'foo': 'qux', 'bar': True, 'baz': 1, 'qux': ['a', 'b']}))
    with pytest.raises(DagsterInvalidConfigError):
        basic_op_no_context(build_op_context(op_config={'foo': 'bar', 'bar': 'true', 'baz': 1, 'qux': ['a', 'b']}))

def test_validate_run_config():
    if False:
        return 10

    class MyBasicOpConfig(Config):
        foo: str

    @op()
    def requires_config(config: MyBasicOpConfig):
        if False:
            for i in range(10):
                print('nop')
        pass

    @job
    def job_requires_config():
        if False:
            while True:
                i = 10
        requires_config()
    result = validate_run_config(job_requires_config, {'ops': {'requires_config': {'config': {'foo': 'bar'}}}})
    assert result == {'ops': {'requires_config': {'config': {'foo': 'bar'}, 'inputs': {}, 'outputs': None}}, 'execution': {'multi_or_in_process_executor': {'multiprocess': {'max_concurrent': None, 'retries': {'enabled': {}}}}}, 'resources': {'io_manager': {'config': None}}, 'loggers': {}}
    result_with_runconfig = validate_run_config(job_requires_config, RunConfig(ops={'requires_config': {'config': {'foo': 'bar'}}}))
    assert result_with_runconfig == result
    result_with_structured_in = validate_run_config(job_requires_config, RunConfig(ops={'requires_config': MyBasicOpConfig(foo='bar')}))
    assert result_with_structured_in == result
    result_with_dict_config = validate_run_config(job_requires_config, {'ops': {'requires_config': {'config': {'foo': 'bar'}}}})
    assert result_with_dict_config == {'ops': {'requires_config': {'config': {'foo': 'bar'}, 'inputs': {}, 'outputs': None}}, 'execution': {'multi_or_in_process_executor': {'multiprocess': {'max_concurrent': None, 'retries': {'enabled': {}}}}}, 'resources': {'io_manager': {'config': None}}, 'loggers': {}}
    with pytest.raises(DagsterInvalidConfigError):
        validate_run_config(job_requires_config)

@pytest.mark.skipif(sys.version_info < (3, 8), reason='requires python3.8')
def test_cached_property():
    if False:
        print('Hello World!')
    from functools import cached_property
    counts = {'plus': 0, 'mult': 0}

    class SomeConfig(Config):
        x: int
        y: int

        @cached_property
        def plus(self):
            if False:
                for i in range(10):
                    print('nop')
            counts['plus'] += 1
            return self.x + self.y

        @cached_property
        def mult(self):
            if False:
                return 10
            counts['mult'] += 1
            return self.x * self.y
    config = SomeConfig(x=3, y=5)
    assert counts['plus'] == 0
    assert counts['mult'] == 0
    assert config.plus == 8
    assert counts['plus'] == 1
    assert counts['mult'] == 0
    assert config.plus == 8
    assert counts['plus'] == 1
    assert counts['mult'] == 0
    assert config.mult == 15
    assert counts['plus'] == 1
    assert counts['mult'] == 1

def test_cached_method():
    if False:
        print('Hello World!')
    counts = {'plus': 0, 'mult': 0}

    class SomeConfig(Config):
        x: int
        y: int

        @cached_method
        def plus(self):
            if False:
                while True:
                    i = 10
            counts['plus'] += 1
            return self.x + self.y

        @cached_method
        def mult(self):
            if False:
                return 10
            counts['mult'] += 1
            return self.x * self.y
    config = SomeConfig(x=3, y=5)
    assert counts['plus'] == 0
    assert counts['mult'] == 0
    assert config.plus() == 8
    assert counts['plus'] == 1
    assert counts['mult'] == 0
    assert config.plus() == 8
    assert counts['plus'] == 1
    assert counts['mult'] == 0
    assert config.mult() == 15
    assert counts['plus'] == 1
    assert counts['mult'] == 1

def test_string_source_default():
    if False:
        i = 10
        return i + 15

    class RawStringConfigSchema(Config):
        a_str: str
    assert print_config_type_to_string({'a_str': StringSource}) == print_config_type_to_string(infer_schema_from_config_class(RawStringConfigSchema).config_type)

def test_string_source_default_directly_on_op():
    if False:
        for i in range(10):
            print('nop')

    @op
    def op_with_raw_str_config(config: str):
        if False:
            i = 10
            return i + 15
        raise Exception('not called')
    assert isinstance(op_with_raw_str_config, OpDefinition)
    assert op_with_raw_str_config.config_field
    assert op_with_raw_str_config.config_field.config_type is StringSource

def test_bool_source_default():
    if False:
        print('Hello World!')

    class RawBoolConfigSchema(Config):
        a_bool: bool
    assert print_config_type_to_string({'a_bool': BoolSource}) == print_config_type_to_string(infer_schema_from_config_class(RawBoolConfigSchema).config_type)

def test_int_source_default():
    if False:
        return 10

    class RawIntConfigSchema(Config):
        an_int: int
    assert print_config_type_to_string({'an_int': IntSource}) == print_config_type_to_string(infer_schema_from_config_class(RawIntConfigSchema).config_type)

def test_optional_string_source_default() -> None:
    if False:
        while True:
            i = 10

    class RawStringConfigSchema(Config):
        a_str: Optional[str]
    assert print_config_type_to_string({'a_str': dagster.Field(Noneable(StringSource))}) == print_config_type_to_string(infer_schema_from_config_class(RawStringConfigSchema).config_type)
    assert RawStringConfigSchema(a_str=None).a_str is None

def test_optional_string_source_with_default_none() -> None:
    if False:
        print('Hello World!')

    class RawStringConfigSchema(Config):
        a_str: Optional[str] = None
    assert print_config_type_to_string({'a_str': dagster.Field(Noneable(StringSource))}) == print_config_type_to_string(infer_schema_from_config_class(RawStringConfigSchema).config_type)
    assert RawStringConfigSchema().a_str is None
    assert RawStringConfigSchema(a_str=None).a_str is None

def test_optional_bool_source_default() -> None:
    if False:
        return 10

    class RawBoolConfigSchema(Config):
        a_bool: Optional[bool]
    assert print_config_type_to_string({'a_bool': dagster.Field(Noneable(BoolSource))}) == print_config_type_to_string(infer_schema_from_config_class(RawBoolConfigSchema).config_type)

def test_optional_int_source_default() -> None:
    if False:
        for i in range(10):
            print('nop')

    class OptionalInt(Config):
        an_int: Optional[int]
    assert print_config_type_to_string({'an_int': dagster.Field(Noneable(IntSource))}) == print_config_type_to_string(infer_schema_from_config_class(OptionalInt).config_type)

def test_schema_aliased_field():
    if False:
        i = 10
        return i + 15

    class ConfigWithSchema(Config):
        schema_: str = pydantic.Field(alias='schema')
    obj = ConfigWithSchema(schema='foo')
    assert obj.schema_ == 'foo'
    assert obj.dict() == {'schema_': 'foo'}
    assert obj.dict(by_alias=True) == {'schema': 'foo'}
    assert print_config_type_to_string({'schema': dagster.Field(StringSource)}) == print_config_type_to_string(infer_schema_from_config_class(ConfigWithSchema).config_type)
    executed = {}

    @op
    def an_op(context, config: ConfigWithSchema):
        if False:
            return 10
        assert config.schema_ == 'bar'
        assert context.op_config == {'schema': 'bar'}
        executed['yes'] = True

    @job
    def a_job():
        if False:
            print('Hello World!')
        an_op()
    assert a_job.execute_in_process({'ops': {'an_op': {'config': {'schema': 'bar'}}}}).success
    assert executed['yes']

def test_env_var():
    if False:
        return 10
    with environ({'ENV_VARIABLE_FOR_TEST_INT': '2', 'ENV_VARIABLE_FOR_TEST': 'foo'}):

        class AnAssetConfig(Config):
            a_string: str
            an_int: int
        executed = {}

        @asset
        def my_asset(config: AnAssetConfig):
            if False:
                for i in range(10):
                    print('nop')
            assert config.a_string == 'foo'
            assert config.an_int == 2
            executed['yes'] = True
        assert build_assets_job('blah', [my_asset], config={'ops': {'my_asset': {'config': {'a_string': {'env': 'ENV_VARIABLE_FOR_TEST'}, 'an_int': {'env': 'ENV_VARIABLE_FOR_TEST_INT'}}}}}).execute_in_process().success
        assert executed['yes']

def test_structured_run_config_ops():
    if False:
        print('Hello World!')

    class ANewConfigOpConfig(Config):
        a_string: str
        an_int: int
    executed = {}

    @op
    def a_struct_config_op(config: ANewConfigOpConfig):
        if False:
            while True:
                i = 10
        executed['yes'] = True
        assert config.a_string == 'foo'
        assert config.an_int == 2

    @job
    def a_job():
        if False:
            print('Hello World!')
        a_struct_config_op()
    a_job.execute_in_process(RunConfig(ops={'a_struct_config_op': ANewConfigOpConfig(a_string='foo', an_int=2)}))
    assert executed['yes']

def test_structured_run_config_optional() -> None:
    if False:
        print('Hello World!')

    class ANewConfigOpConfig(Config):
        a_string: Optional[str]
        an_int: Optional[int] = None
        a_float: float = PyField(None)
    executed = {}

    @op
    def a_struct_config_op(config: ANewConfigOpConfig):
        if False:
            for i in range(10):
                print('nop')
        executed['yes'] = True
        assert config.a_string is None
        assert config.an_int is None
        assert config.a_float is None

    @job
    def a_job():
        if False:
            while True:
                i = 10
        a_struct_config_op()
    a_job.execute_in_process(RunConfig(ops={'a_struct_config_op': ANewConfigOpConfig(a_string=None)}))
    assert executed['yes']

def test_structured_run_config_multi_asset():
    if False:
        while True:
            i = 10

    class AMultiAssetConfig(Config):
        a_string: str
        an_int: int
    executed = {}

    @multi_asset(outs={'a': AssetOut(key='asset_a'), 'b': AssetOut(key='asset_b')})
    def two_assets(config: AMultiAssetConfig):
        if False:
            for i in range(10):
                print('nop')
        assert config.a_string == 'foo'
        assert config.an_int == 2
        executed['yes'] = True
        return (1, 2)
    assert build_assets_job('blah', [two_assets], config=RunConfig(ops={'two_assets': AMultiAssetConfig(a_string='foo', an_int=2)})).execute_in_process().success

def test_structured_run_config_assets():
    if False:
        for i in range(10):
            print('nop')

    class AnAssetConfig(Config):
        a_string: str
        an_int: int
    executed = {}

    @asset
    def my_asset(config: AnAssetConfig):
        if False:
            return 10
        assert config.a_string == 'foo'
        assert config.an_int == 2
        executed['yes'] = True
    assert build_assets_job('blah', [my_asset], config=RunConfig(ops={'my_asset': AnAssetConfig(a_string='foo', an_int=2)})).execute_in_process().success
    assert executed['yes']
    del executed['yes']
    my_asset_job = define_asset_job('my_asset_job', selection='my_asset', config=RunConfig(ops={'my_asset': AnAssetConfig(a_string='foo', an_int=2)}))
    defs = Definitions(assets=[my_asset], jobs=[my_asset_job])
    defs.get_job_def('my_asset_job').execute_in_process()
    assert executed['yes']
    del executed['yes']
    asset_result = materialize([my_asset], run_config=RunConfig(ops={'my_asset': AnAssetConfig(a_string='foo', an_int=2)}))
    assert asset_result.success
    assert executed['yes']

def test_structured_run_config_assets_optional() -> None:
    if False:
        i = 10
        return i + 15

    class AnAssetConfig(Config):
        a_string: str = PyField(None)
        an_int: Optional[int] = None
    executed = {}

    @asset
    def my_asset(config: AnAssetConfig):
        if False:
            return 10
        assert config.a_string is None
        assert config.an_int is None
        executed['yes'] = True
    asset_result = materialize([my_asset], run_config=RunConfig(ops={'my_asset': AnAssetConfig()}))
    assert asset_result.success
    assert executed['yes']

def test_direct_op_invocation_plain_arg_with_config() -> None:
    if False:
        i = 10
        return i + 15

    class MyConfig(Config):
        num: int
    executed = {}

    @op
    def an_op(config: MyConfig) -> None:
        if False:
            return 10
        assert config.num == 1
        executed['yes'] = True
    an_op(MyConfig(num=1))
    assert executed['yes']

def test_direct_op_invocation_kwarg_with_config() -> None:
    if False:
        print('Hello World!')

    class MyConfig(Config):
        num: int
    executed = {}

    @op
    def an_op(config: MyConfig) -> None:
        if False:
            print('Hello World!')
        assert config.num == 1
        executed['yes'] = True
    an_op(config=MyConfig(num=1))
    assert executed['yes']

def test_direct_op_invocation_arg_complex() -> None:
    if False:
        print('Hello World!')

    class MyConfig(Config):
        num: int

    class MyOuterConfig(Config):
        inner: MyConfig
        string: str
    executed = {}

    @op
    def an_op(config: MyOuterConfig) -> None:
        if False:
            print('Hello World!')
        assert config.inner.num == 1
        assert config.string == 'foo'
        executed['yes'] = True
    an_op(MyOuterConfig(inner=MyConfig(num=1), string='foo'))
    assert executed['yes']

def test_direct_op_invocation_kwarg_complex() -> None:
    if False:
        print('Hello World!')

    class MyConfig(Config):
        num: int

    class MyOuterConfig(Config):
        inner: MyConfig
        string: str
    executed = {}

    @op
    def an_op(config: MyOuterConfig) -> None:
        if False:
            return 10
        assert config.inner.num == 1
        assert config.string == 'foo'
        executed['yes'] = True
    an_op(config=MyOuterConfig(inner=MyConfig(num=1), string='foo'))
    assert executed['yes']

def test_direct_op_invocation_kwarg_very_complex() -> None:
    if False:
        i = 10
        return i + 15

    class MyConfig(Config):
        num: int

    class MyOuterConfig(Config):
        inner: MyConfig
        string: str

    class MyOutermostConfig(Config):
        inner: MyOuterConfig
        boolean: bool
    executed = {}

    @op
    def an_op(config: MyOutermostConfig) -> None:
        if False:
            return 10
        assert config.inner.inner.num == 2
        assert config.inner.string == 'foo'
        assert config.boolean is False
        executed['yes'] = True
    with environ({'ENV_VARIABLE_FOR_TEST_INT': '2'}):
        an_op(config=MyOutermostConfig(inner=MyOuterConfig(inner=MyConfig(num=EnvVar.int('ENV_VARIABLE_FOR_TEST_INT')), string='foo'), boolean=False))
    assert executed['yes']

def test_direct_asset_invocation_plain_arg_with_config() -> None:
    if False:
        i = 10
        return i + 15

    class MyConfig(Config):
        num: int
    executed = {}

    @asset
    def an_asset(config: MyConfig) -> None:
        if False:
            while True:
                i = 10
        assert config.num == 1
        executed['yes'] = True
    an_asset(MyConfig(num=1))
    assert executed['yes']

def test_direct_asset_invocation_kwarg_with_config() -> None:
    if False:
        print('Hello World!')

    class MyConfig(Config):
        num: int
    executed = {}

    @asset
    def an_asset(config: MyConfig) -> None:
        if False:
            while True:
                i = 10
        assert config.num == 1
        executed['yes'] = True
    an_asset(config=MyConfig(num=1))
    assert executed['yes']

def test_direct_op_invocation_kwarg_with_config_and_context() -> None:
    if False:
        return 10

    class MyConfig(Config):
        num: int
    executed = {}

    @op
    def an_op(context, config: MyConfig) -> None:
        if False:
            while True:
                i = 10
        assert config.num == 1
        executed['yes'] = True
    an_op(context=build_op_context(), config=MyConfig(num=1))
    assert executed['yes']

def test_direct_op_invocation_kwarg_with_config_and_context_err() -> None:
    if False:
        return 10

    class MyConfig(Config):
        num: int
    executed = {}

    @op
    def an_op(context, config: MyConfig) -> None:
        if False:
            while True:
                i = 10
        assert config.num == 1
        executed['yes'] = True
    with pytest.raises(DagsterInvalidInvocationError, match='Cannot provide config in both context and kwargs'):
        an_op(context=build_op_context(config={'num': 2}), config=MyConfig(num=1))

def test_truthy_and_falsey_defaults() -> None:
    if False:
        return 10

    class ConfigClassToConvertTrue(Config):
        bool_with_default_true_value: bool = PyField(default=True)
    fields = ConfigClassToConvertTrue.to_fields_dict()
    true_default_field = fields['bool_with_default_true_value']
    assert true_default_field.is_required is False
    assert true_default_field.default_provided is True
    assert true_default_field.default_value is True

    class ConfigClassToConvertFalse(Config):
        bool_with_default_false_value: bool = PyField(default=False)
    fields = ConfigClassToConvertFalse.to_fields_dict()
    false_default_field = fields['bool_with_default_false_value']
    assert false_default_field.is_required is False
    assert false_default_field.default_provided is True
    assert false_default_field.default_value is False

def execution_run_config() -> None:
    if False:
        return 10
    from dagster import RunConfig, job, op

    @op
    def foo_op():
        if False:
            i = 10
            return i + 15
        pass

    @job
    def foo_job():
        if False:
            for i in range(10):
                print('nop')
        foo_op()
    result = foo_job.execute_in_process(run_config=RunConfig(execution={'config': {'multiprocess': {'config': {'max_concurrent': 0}}}}))
    assert result.success