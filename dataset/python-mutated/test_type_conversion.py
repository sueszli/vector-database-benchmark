from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple
import click
import pytest
import typer
from typer.testing import CliRunner
runner = CliRunner()

def test_optional():
    if False:
        print('Hello World!')
    app = typer.Typer()

    @app.command()
    def opt(user: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        if user:
            print(f'User: {user}')
        else:
            print('No user')
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert 'No user' in result.output
    result = runner.invoke(app, ['--user', 'Camila'])
    assert result.exit_code == 0
    assert 'User: Camila' in result.output

def test_no_type():
    if False:
        i = 10
        return i + 15
    app = typer.Typer()

    @app.command()
    def no_type(user):
        if False:
            while True:
                i = 10
        print(f'User: {user}')
    result = runner.invoke(app, ['Camila'])
    assert result.exit_code == 0
    assert 'User: Camila' in result.output

class SomeEnum(Enum):
    ONE = 'one'
    TWO = 'two'
    THREE = 'three'

@pytest.mark.parametrize('type_annotation', [List[Path], List[SomeEnum], List[str]])
def test_list_parameters_convert_to_lists(type_annotation):
    if False:
        return 10
    expected_element_type = type_annotation.__args__[0]
    app = typer.Typer()

    @app.command()
    def list_conversion(container: type_annotation):
        if False:
            print('Hello World!')
        assert isinstance(container, list)
        for element in container:
            assert isinstance(element, expected_element_type)
    result = runner.invoke(app, ['one', 'two', 'three'])
    assert result.exit_code == 0

@pytest.mark.parametrize('type_annotation', [Tuple[str, str], Tuple[str, Path], Tuple[Path, Path], Tuple[str, SomeEnum], Tuple[SomeEnum, SomeEnum]])
def test_tuple_parameter_elements_are_converted_recursively(type_annotation):
    if False:
        while True:
            i = 10
    expected_element_types = type_annotation.__args__
    app = typer.Typer()

    @app.command()
    def tuple_recursive_conversion(container: type_annotation):
        if False:
            return 10
        assert isinstance(container, tuple)
        for (element, expected_type) in zip(container, expected_element_types):
            assert isinstance(element, expected_type)
    result = runner.invoke(app, ['one', 'two'])
    assert result.exit_code == 0

def test_custom_parse():
    if False:
        for i in range(10):
            print('nop')
    app = typer.Typer()

    @app.command()
    def custom_parser(hex_value: int=typer.Argument(None, parser=lambda x: int(x, 0))):
        if False:
            print('Hello World!')
        assert hex_value == 86
    result = runner.invoke(app, ['0x56'])
    assert result.exit_code == 0

def test_custom_click_type():
    if False:
        print('Hello World!')

    class BaseNumberParamType(click.ParamType):
        name = 'base_integer'

        def convert(self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> Any:
            if False:
                print('Hello World!')
            return int(value, 0)
    app = typer.Typer()

    @app.command()
    def custom_click_type(hex_value: int=typer.Argument(None, click_type=BaseNumberParamType())):
        if False:
            return 10
        assert hex_value == 86
    result = runner.invoke(app, ['0x56'])
    assert result.exit_code == 0