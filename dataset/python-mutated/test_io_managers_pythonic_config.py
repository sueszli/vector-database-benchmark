import os
import tempfile
from typing import Any, Type
from dagster import Config, ConfigurableIOManagerFactory, DataVersion, Definitions, FilesystemIOManager, In, IOManagerDefinition, RunConfig, StringSource, asset, io_manager, job, observable_source_asset, op
from dagster._config.pythonic_config import ConfigurableIOManager, ConfigurableResource
from dagster._config.type_printer import print_config_type_to_string
from dagster._core.storage.io_manager import IOManager

def type_string_from_config_schema(config_schema):
    if False:
        i = 10
        return i + 15
    return print_config_type_to_string(config_schema.config_type)

def test_load_input_handle_output():
    if False:
        i = 10
        return i + 15

    class MyIOManager(ConfigurableIOManager):

        def handle_output(self, context, obj):
            if False:
                return 10
            pass

        def load_input(self, context):
            if False:
                i = 10
                return i + 15
            assert False, 'should not be called'

    class MyInputManager(MyIOManager):

        def load_input(self, context):
            if False:
                i = 10
                return i + 15
            if context.upstream_output is None:
                assert False, 'upstream output should not be None'
            else:
                return 6
    did_run = {}

    @op
    def first_op():
        if False:
            for i in range(10):
                print('nop')
        did_run['first_op'] = True
        return 1

    @op(ins={'an_input': In(input_manager_key='my_input_manager')})
    def second_op(an_input):
        if False:
            return 10
        assert an_input == 6
        did_run['second_op'] = True

    @job(resource_defs={'io_manager': MyIOManager(), 'my_input_manager': MyInputManager()})
    def check_input_managers():
        if False:
            return 10
        out = first_op()
        second_op(out)
    check_input_managers.execute_in_process()
    assert did_run['first_op']
    assert did_run['second_op']

def test_runtime_config():
    if False:
        while True:
            i = 10
    out_txt = []

    class MyIOManager(ConfigurableIOManager):
        prefix: str

        def handle_output(self, context, obj):
            if False:
                print('Hello World!')
            out_txt.append(f'{self.prefix}{obj}')

        def load_input(self, context):
            if False:
                print('Hello World!')
            assert False, 'should not be called'

    @asset
    def hello_world_asset():
        if False:
            i = 10
            return i + 15
        return 'hello, world!'
    defs = Definitions(assets=[hello_world_asset], resources={'io_manager': MyIOManager.configure_at_launch()})
    assert defs.get_implicit_global_asset_job_def().execute_in_process({'resources': {'io_manager': {'config': {'prefix': ''}}}}).success
    assert out_txt == ['hello, world!']
    out_txt.clear()
    assert defs.get_implicit_global_asset_job_def().execute_in_process({'resources': {'io_manager': {'config': {'prefix': 'greeting: '}}}}).success
    assert out_txt == ['greeting: hello, world!']

def test_nested_resources():
    if False:
        i = 10
        return i + 15
    out_txt = []

    class IOConfigResource(ConfigurableResource):
        prefix: str

    class MyIOManager(ConfigurableIOManager):
        config: IOConfigResource

        def handle_output(self, context, obj):
            if False:
                return 10
            out_txt.append(f'{self.config.prefix}{obj}')

        def load_input(self, context):
            if False:
                print('Hello World!')
            assert False, 'should not be called'

    @asset
    def hello_world_asset():
        if False:
            return 10
        return 'hello, world!'
    defs = Definitions(assets=[hello_world_asset], resources={'io_manager': MyIOManager(config=IOConfigResource(prefix='greeting: '))})
    assert defs.get_implicit_global_asset_job_def().execute_in_process().success
    assert out_txt == ['greeting: hello, world!']

def test_nested_resources_runtime_config():
    if False:
        while True:
            i = 10
    out_txt = []

    class IOConfigResource(ConfigurableResource):
        prefix: str

    class MyIOManager(ConfigurableIOManager):
        config: IOConfigResource

        def handle_output(self, context, obj):
            if False:
                i = 10
                return i + 15
            out_txt.append(f'{self.config.prefix}{obj}')

        def load_input(self, context):
            if False:
                print('Hello World!')
            assert False, 'should not be called'

    @asset
    def hello_world_asset():
        if False:
            i = 10
            return i + 15
        return 'hello, world!'
    io_config = IOConfigResource.configure_at_launch()
    defs = Definitions(assets=[hello_world_asset], resources={'io_config': io_config, 'io_manager': MyIOManager(config=io_config)})
    assert defs.get_implicit_global_asset_job_def().execute_in_process({'resources': {'io_config': {'config': {'prefix': ''}}}}).success
    assert out_txt == ['hello, world!']
    out_txt.clear()
    assert defs.get_implicit_global_asset_job_def().execute_in_process({'resources': {'io_config': {'config': {'prefix': 'greeting: '}}}}).success
    assert out_txt == ['greeting: hello, world!']

def test_pythonic_fs_io_manager() -> None:
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as tmpdir_path:

        @asset
        def hello_world_asset():
            if False:
                return 10
            return 'hello, world!'
        defs = Definitions(assets=[hello_world_asset], resources={'io_manager': FilesystemIOManager(base_dir=tmpdir_path)})
        assert not os.path.exists(os.path.join(tmpdir_path, 'hello_world_asset'))
        assert defs.get_implicit_global_asset_job_def().execute_in_process().success
        assert os.path.exists(os.path.join(tmpdir_path, 'hello_world_asset'))

def test_pythonic_fs_io_manager_runtime_config() -> None:
    if False:
        print('Hello World!')
    with tempfile.TemporaryDirectory() as tmpdir_path:

        @asset
        def hello_world_asset():
            if False:
                i = 10
                return i + 15
            return 'hello, world!'
        defs = Definitions(assets=[hello_world_asset], resources={'io_manager': FilesystemIOManager.configure_at_launch()})
        assert not os.path.exists(os.path.join(tmpdir_path, 'hello_world_asset'))
        assert defs.get_implicit_global_asset_job_def().execute_in_process(run_config=RunConfig(resources={'io_manager': FilesystemIOManager(base_dir=tmpdir_path)})).success
        assert os.path.exists(os.path.join(tmpdir_path, 'hello_world_asset'))

def test_config_schemas() -> None:
    if False:
        return 10

    @io_manager(config_schema={'base_dir': StringSource}, output_config_schema={'path': StringSource}, input_config_schema={'format': StringSource})
    def an_io_manager():
        if False:
            return 10
        pass

    class OutputConfigSchema(Config):
        path: str

    class InputConfigSchema(Config):
        format: str

    class MyIOManager(ConfigurableIOManager):
        base_dir: str

        @classmethod
        def input_config_schema(cls) -> Type[Config]:
            if False:
                return 10
            return InputConfigSchema

        @classmethod
        def output_config_schema(cls) -> Type[Config]:
            if False:
                while True:
                    i = 10
            return OutputConfigSchema

        def handle_output(self, context, obj):
            if False:
                print('Hello World!')
            pass

        def load_input(self, context):
            if False:
                while True:
                    i = 10
            pass
    configured_io_manager = MyIOManager(base_dir='/a/b/c').get_resource_definition()
    assert isinstance(configured_io_manager, IOManagerDefinition)
    assert type_string_from_config_schema(configured_io_manager.output_config_schema) == type_string_from_config_schema(an_io_manager.output_config_schema)
    assert type_string_from_config_schema(configured_io_manager.input_config_schema) == type_string_from_config_schema(an_io_manager.input_config_schema)

    class MyIOManagerNonPythonicSchemas(ConfigurableIOManager):
        base_dir: str

        @classmethod
        def input_config_schema(cls):
            if False:
                i = 10
                return i + 15
            return {'format': StringSource}

        @classmethod
        def output_config_schema(cls):
            if False:
                print('Hello World!')
            return {'path': StringSource}

        def handle_output(self, context, obj):
            if False:
                return 10
            pass

        def load_input(self, context):
            if False:
                while True:
                    i = 10
            pass
    configured_io_manager_non_pythonic = MyIOManagerNonPythonicSchemas(base_dir='/a/b/c').get_resource_definition()
    assert isinstance(configured_io_manager_non_pythonic, IOManagerDefinition)
    assert type_string_from_config_schema(configured_io_manager_non_pythonic.output_config_schema) == type_string_from_config_schema(an_io_manager.output_config_schema)
    assert type_string_from_config_schema(configured_io_manager_non_pythonic.input_config_schema) == type_string_from_config_schema(an_io_manager.input_config_schema)
import pytest
from dagster import InputContext, Out, OutputContext
from dagster._core.errors import DagsterInvalidConfigError

def test_load_input_handle_output_input_config() -> None:
    if False:
        return 10

    class MyIOManager(ConfigurableIOManager):

        def handle_output(self, context, obj):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def load_input(self, context):
            if False:
                for i in range(10):
                    print('nop')
            assert False, 'should not be called'

    class InputConfigSchema(Config):
        config_value: int

    class MyInputManager(MyIOManager):

        def load_input(self, context):
            if False:
                return 10
            if context.upstream_output is None:
                assert False, 'upstream output should not be None'
            else:
                return context.config['config_value']

        @classmethod
        def input_config_schema(cls) -> Type[Config]:
            if False:
                for i in range(10):
                    print('nop')
            return InputConfigSchema
    did_run = {}

    @op
    def first_op():
        if False:
            i = 10
            return i + 15
        did_run['first_op'] = True
        return 1

    @op(ins={'an_input': In(input_manager_key='my_input_manager')})
    def second_op(an_input):
        if False:
            for i in range(10):
                print('nop')
        assert an_input == 6
        did_run['second_op'] = True

    @job(resource_defs={'io_manager': MyIOManager(), 'my_input_manager': MyInputManager()})
    def check_input_managers():
        if False:
            i = 10
            return i + 15
        out = first_op()
        second_op(out)
    check_input_managers.execute_in_process(run_config={'ops': {'second_op': {'inputs': {'an_input': {'config_value': 6}}}}})
    assert did_run['first_op']
    assert did_run['second_op']
    with pytest.raises(DagsterInvalidConfigError):
        check_input_managers.execute_in_process(run_config={'ops': {'second_op': {'inputs': {'an_input': {'config_value': 'a_string'}}}}})

def test_config_param_load_input_handle_output_config() -> None:
    if False:
        print('Hello World!')
    storage = {}

    class InputConfigSchema(Config):
        prefix_input: str

    class OutputConfigSchema(Config):
        postfix_output: str

    class MyIOManager(ConfigurableIOManager):
        prefix_output: str

        @classmethod
        def input_config_schema(cls) -> Type[Config]:
            if False:
                return 10
            return InputConfigSchema

        @classmethod
        def output_config_schema(cls) -> Type[Config]:
            if False:
                print('Hello World!')
            return OutputConfigSchema

        def load_input(self, context: InputContext):
            if False:
                return 10
            return f"{context.config['prefix_input']}{storage[context.name]}"

        def handle_output(self, context: OutputContext, obj: str):
            if False:
                for i in range(10):
                    print('nop')
            storage[context.name] = f"{self.prefix_output}{obj}{context.config['postfix_output']}"
    did_run = {}

    @op(out={'first_op': Out(io_manager_key='io_manager')})
    def first_op():
        if False:
            print('Hello World!')
        did_run['first_op'] = True
        return 'foo'

    @op(ins={'first_op': In(input_manager_key='io_manager')}, out={'second_op': Out(io_manager_key='io_manager')})
    def second_op(first_op):
        if False:
            return 10
        assert first_op == 'barprefoopost'
        did_run['second_op'] = True
        return first_op

    @job(resource_defs={'io_manager': MyIOManager(prefix_output='pre')})
    def check_input_managers():
        if False:
            for i in range(10):
                print('nop')
        out = first_op()
        second_op(out)
    check_input_managers.execute_in_process(run_config={'ops': {'first_op': {'outputs': {'first_op': {'postfix_output': 'post'}}}, 'second_op': {'inputs': {'first_op': {'prefix_input': 'bar'}}, 'outputs': {'second_op': {'postfix_output': 'post'}}}}})
    assert did_run['first_op']
    assert did_run['second_op']
    assert storage['first_op'] == 'prefoopost'
    assert storage['second_op'] == 'prebarprefoopostpost'

def test_io_manager_def() -> None:
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryDirectory() as tmpdir_path:

        @asset(io_manager_def=FilesystemIOManager(base_dir=tmpdir_path))
        def hello_world_asset():
            if False:
                i = 10
                return i + 15
            return 'hello, world!'
        defs = Definitions(assets=[hello_world_asset])
        assert not os.path.exists(os.path.join(tmpdir_path, 'hello_world_asset'))
        assert defs.get_implicit_global_asset_job_def().execute_in_process().success
        assert os.path.exists(os.path.join(tmpdir_path, 'hello_world_asset'))

def test_observable_source_asset_io_manager_def() -> None:
    if False:
        print('Hello World!')

    class FileStringIOManager(ConfigurableIOManager):
        base_path: str

        def load_input(self, context: 'InputContext') -> object:
            if False:
                for i in range(10):
                    print('nop')
            with open(os.path.join(self.base_path, '/'.join(context.asset_key.path)), mode='r', encoding='utf-8') as ff:
                return str(ff.read())

        def handle_output(self, context: 'OutputContext', obj: Any) -> None:
            if False:
                while True:
                    i = 10
            with open(os.path.join(self.base_path, '/'.join(context.asset_key.path)), mode='w', encoding='utf-8') as ff:
                ff.write(str(obj))
    with tempfile.TemporaryDirectory() as tmpdir_path:
        with open(os.path.join(tmpdir_path, 'my_observable_asset'), 'w') as f:
            f.write('foo')

        @observable_source_asset(io_manager_def=FileStringIOManager(base_path=tmpdir_path))
        def my_observable_asset() -> DataVersion:
            if False:
                print('Hello World!')
            return DataVersion('alpha')

        @asset
        def my_downstream_asset(my_observable_asset: str) -> str:
            if False:
                i = 10
                return i + 15
            return my_observable_asset + 'bar'
        defs = Definitions(assets=[my_observable_asset, my_downstream_asset])
        result = defs.get_implicit_global_asset_job_def().execute_in_process()
        assert result.success
        assert result.output_for_node('my_downstream_asset') == 'foobar'

def test_telemetry_custom_io_manager():
    if False:
        while True:
            i = 10

    class MyIOManager(ConfigurableIOManager):

        def handle_output(self, context, obj):
            if False:
                return 10
            return {}

        def load_input(self, context):
            if False:
                print('Hello World!')
            return 1
    assert not MyIOManager._is_dagster_maintained()

def test_telemetry_dagster_io_manager():
    if False:
        print('Hello World!')

    class MyIOManager(ConfigurableIOManager):

        @classmethod
        def _is_dagster_maintained(cls) -> bool:
            if False:
                i = 10
                return i + 15
            return True

        def handle_output(self, context, obj):
            if False:
                print('Hello World!')
            return {}

        def load_input(self, context):
            if False:
                for i in range(10):
                    print('nop')
            return 1
    assert MyIOManager()._is_dagster_maintained()

def test_telemetry_custom_io_manager_factory():
    if False:
        return 10

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                return 10
            return {}

        def load_input(self, context):
            if False:
                print('Hello World!')
            return 1

    class AnIOManagerFactory(ConfigurableIOManagerFactory):

        def create_io_manager(self, _) -> IOManager:
            if False:
                i = 10
                return i + 15
            return MyIOManager()
    assert not AnIOManagerFactory()._is_dagster_maintained()

def test_telemetry_dagster_io_manager_factory():
    if False:
        while True:
            i = 10

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                return 10
            return {}

        def load_input(self, context):
            if False:
                print('Hello World!')
            return 1

    class AnIOManagerFactory(ConfigurableIOManagerFactory):

        @classmethod
        def _is_dagster_maintained(cls) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return True

        def create_io_manager(self, _) -> IOManager:
            if False:
                return 10
            return MyIOManager()
    assert AnIOManagerFactory()._is_dagster_maintained()