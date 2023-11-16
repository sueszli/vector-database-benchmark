from enum import Enum
from typing import Any, Dict
import pytest
from dagster import Config, RunConfig, config_mapping, graph, job, op
from dagster._check import CheckError

def test_binding_runconfig() -> None:
    if False:
        print('Hello World!')

    class DoSomethingConfig(Config):
        config_param: str

    @op
    def do_something(config: DoSomethingConfig) -> str:
        if False:
            i = 10
            return i + 15
        return config.config_param

    @job(config=RunConfig(ops={'do_something': DoSomethingConfig(config_param='foo')}))
    def do_it_all_with_baked_in_config() -> None:
        if False:
            i = 10
            return i + 15
        do_something()
    result = do_it_all_with_baked_in_config.execute_in_process()
    assert result.success
    assert result.output_for_node('do_something') == 'foo'

def test_config_mapping_return_config_dict() -> None:
    if False:
        while True:
            i = 10

    class DoSomethingConfig(Config):
        config_param: str

    @op
    def do_something(config: DoSomethingConfig) -> str:
        if False:
            return 10
        return config.config_param

    class ConfigMappingConfig(Config):
        simplified_param: str

    @config_mapping
    def simplified_config(config_in: ConfigMappingConfig) -> Dict[str, Any]:
        if False:
            return 10
        return {'ops': {'do_something': {'config': {'config_param': config_in.simplified_param}}}}

    @job(config=simplified_config)
    def do_it_all_with_simplified_config() -> None:
        if False:
            print('Hello World!')
        do_something()
    result = do_it_all_with_simplified_config.execute_in_process(run_config={'simplified_param': 'foo'})
    assert result.success
    assert result.output_for_node('do_something') == 'foo'

def test_config_mapping_return_run_config() -> None:
    if False:
        return 10

    class DoSomethingConfig(Config):
        config_param: str

    @op
    def do_something(config: DoSomethingConfig) -> str:
        if False:
            return 10
        return config.config_param

    class ConfigMappingConfig(Config):
        simplified_param: str

    @config_mapping
    def simplified_config(config_in: ConfigMappingConfig) -> RunConfig:
        if False:
            print('Hello World!')
        return RunConfig(ops={'do_something': DoSomethingConfig(config_param=config_in.simplified_param)})

    @job(config=simplified_config)
    def do_it_all_with_simplified_config() -> None:
        if False:
            i = 10
            return i + 15
        do_something()
    result = do_it_all_with_simplified_config.execute_in_process(run_config={'simplified_param': 'foo'})
    assert result.success
    assert result.output_for_node('do_something') == 'foo'

def test_config_mapping_config_schema_errs() -> None:
    if False:
        i = 10
        return i + 15

    class DoSomethingConfig(Config):
        config_param: str

    @op
    def do_something(config: DoSomethingConfig) -> str:
        if False:
            while True:
                i = 10
        return config.config_param

    class ConfigMappingConfig(Config):
        simplified_param: str
    with pytest.raises(CheckError):

        @config_mapping(config_schema={'simplified_param': str})
        def simplified_config(config_in: ConfigMappingConfig) -> RunConfig:
            if False:
                while True:
                    i = 10
            return RunConfig(ops={'do_something': DoSomethingConfig(config_param=config_in.simplified_param)})

def test_config_mapping_enum() -> None:
    if False:
        return 10

    class MyEnum(Enum):
        FOO = 'foo'
        BAR = 'bar'

    class DoSomethingConfig(Config):
        config_param: MyEnum

    @op
    def do_something(config: DoSomethingConfig) -> MyEnum:
        if False:
            return 10
        return config.config_param

    class ConfigMappingConfig(Config):
        simplified_param: MyEnum

    @config_mapping
    def simplified_config(config_in: ConfigMappingConfig) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return {'ops': {'do_something': {'config': {'config_param': config_in.simplified_param.name}}}}

    @job(config=simplified_config)
    def do_it_all_with_simplified_config() -> None:
        if False:
            print('Hello World!')
        do_something()
    result = do_it_all_with_simplified_config.execute_in_process(run_config={'simplified_param': 'FOO'})
    assert result.success
    assert result.output_for_node('do_something') == MyEnum.FOO

def test_config_mapping_return_run_config_nested() -> None:
    if False:
        print('Hello World!')

    class DoSomethingConfig(Config):
        config_param: str

    @op
    def do_something(config: DoSomethingConfig) -> str:
        if False:
            while True:
                i = 10
        return config.config_param

    class ConfigMappingConfig(Config):
        simplified_param: str

    @config_mapping
    def simplified_config(config_in: ConfigMappingConfig) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return {'do_something': {'config': {'config_param': config_in.simplified_param}}}

    @graph(config=simplified_config)
    def do_it_all_with_simplified_config() -> None:
        if False:
            return 10
        do_something()

    class OuterConfigMappingConfig(Config):
        simplest_param: str

    @config_mapping
    def even_simpler_config(config_in: OuterConfigMappingConfig) -> RunConfig:
        if False:
            for i in range(10):
                print('nop')
        return RunConfig(ops={'do_it_all_with_simplified_config': ConfigMappingConfig(simplified_param=config_in.simplest_param)})

    @job(config=even_simpler_config)
    def do_it_all_with_even_simpler_config() -> None:
        if False:
            for i in range(10):
                print('nop')
        do_it_all_with_simplified_config()
    result = do_it_all_with_even_simpler_config.execute_in_process(run_config={'simplest_param': 'foo'})
    assert result.success
    assert result.output_for_node('do_it_all_with_simplified_config.do_something') == 'foo'