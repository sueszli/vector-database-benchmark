import pytest
import typer
from typer.testing import CliRunner
from typer.utils import AnnotatedParamWithDefaultValueError, DefaultFactoryAndDefaultValueError, MixedAnnotatedAndDefaultStyleError, MultipleTyperAnnotationsError, _split_annotation_from_typer_annotations
from typing_extensions import Annotated
runner = CliRunner()

def test_split_annotations_from_typer_annotations_simple():
    if False:
        return 10
    given = Annotated[str, typer.Argument()]
    (base, typer_annotations) = _split_annotation_from_typer_annotations(given)
    assert base is str
    assert len(typer_annotations) == 1

def test_forbid_default_value_in_annotated_argument():
    if False:
        print('Hello World!')
    app = typer.Typer()

    @app.command()
    def cmd(my_param: Annotated[str, typer.Argument('foo')]):
        if False:
            while True:
                i = 10
        ...
    with pytest.raises(AnnotatedParamWithDefaultValueError) as excinfo:
        runner.invoke(app)
    assert vars(excinfo.value) == dict(param_type=typer.models.ArgumentInfo, argument_name='my_param')

def test_allow_options_to_have_names():
    if False:
        i = 10
        return i + 15
    app = typer.Typer()

    @app.command()
    def cmd(my_param: Annotated[str, typer.Option('--some-opt')]):
        if False:
            i = 10
            return i + 15
        print(my_param)
    result = runner.invoke(app, ['--some-opt', 'hello'])
    assert result.exit_code == 0, result.output
    assert 'hello' in result.output

@pytest.mark.parametrize(['param', 'param_info_type'], [(typer.Argument, typer.models.ArgumentInfo), (typer.Option, typer.models.OptionInfo)])
def test_forbid_annotated_param_and_default_param(param, param_info_type):
    if False:
        while True:
            i = 10
    app = typer.Typer()

    @app.command()
    def cmd(my_param: Annotated[str, param()]=param('foo')):
        if False:
            return 10
        ...
    with pytest.raises(MixedAnnotatedAndDefaultStyleError) as excinfo:
        runner.invoke(app)
    assert vars(excinfo.value) == dict(argument_name='my_param', annotated_param_type=param_info_type, default_param_type=param_info_type)

def test_forbid_multiple_typer_params_in_annotated():
    if False:
        for i in range(10):
            print('nop')
    app = typer.Typer()

    @app.command()
    def cmd(my_param: Annotated[str, typer.Argument(), typer.Argument()]):
        if False:
            for i in range(10):
                print('nop')
        ...
    with pytest.raises(MultipleTyperAnnotationsError) as excinfo:
        runner.invoke(app)
    assert vars(excinfo.value) == dict(argument_name='my_param')

def test_allow_multiple_non_typer_params_in_annotated():
    if False:
        i = 10
        return i + 15
    app = typer.Typer()

    @app.command()
    def cmd(my_param: Annotated[str, 'someval', typer.Argument(), 4]='hello'):
        if False:
            i = 10
            return i + 15
        print(my_param)
    result = runner.invoke(app)
    assert result.exit_code == 0, result.output
    assert 'hello' in result.output

@pytest.mark.parametrize(['param', 'param_info_type'], [(typer.Argument, typer.models.ArgumentInfo), (typer.Option, typer.models.OptionInfo)])
def test_forbid_default_factory_and_default_value_in_annotated(param, param_info_type):
    if False:
        return 10

    def make_string():
        if False:
            for i in range(10):
                print('nop')
        return 'foo'
    app = typer.Typer()

    @app.command()
    def cmd(my_param: Annotated[str, param(default_factory=make_string)]='hello'):
        if False:
            i = 10
            return i + 15
        ...
    with pytest.raises(DefaultFactoryAndDefaultValueError) as excinfo:
        runner.invoke(app)
    assert vars(excinfo.value) == dict(argument_name='my_param', param_type=param_info_type)

@pytest.mark.parametrize('param', [typer.Argument, typer.Option])
def test_allow_default_factory_with_default_param(param):
    if False:
        return 10

    def make_string():
        if False:
            i = 10
            return i + 15
        return 'foo'
    app = typer.Typer()

    @app.command()
    def cmd(my_param: str=param(default_factory=make_string)):
        if False:
            return 10
        print(my_param)
    result = runner.invoke(app)
    assert result.exit_code == 0, result.output
    assert 'foo' in result.output

@pytest.mark.parametrize(['param', 'param_info_type'], [(typer.Argument, typer.models.ArgumentInfo), (typer.Option, typer.models.OptionInfo)])
def test_forbid_default_and_default_factory_with_default_param(param, param_info_type):
    if False:
        return 10

    def make_string():
        if False:
            print('Hello World!')
        return 'foo'
    app = typer.Typer()

    @app.command()
    def cmd(my_param: str=param('hi', default_factory=make_string)):
        if False:
            i = 10
            return i + 15
        ...
    with pytest.raises(DefaultFactoryAndDefaultValueError) as excinfo:
        runner.invoke(app)
    assert vars(excinfo.value) == dict(argument_name='my_param', param_type=param_info_type)

@pytest.mark.parametrize(['error', 'message'], [(AnnotatedParamWithDefaultValueError(argument_name='my_argument', param_type=typer.models.ArgumentInfo), "`Argument` default value cannot be set in `Annotated` for 'my_argument'. Set the default value with `=` instead."), (MixedAnnotatedAndDefaultStyleError(argument_name='my_argument', annotated_param_type=typer.models.OptionInfo, default_param_type=typer.models.ArgumentInfo), "Cannot specify `Option` in `Annotated` and `Argument` as a default value together for 'my_argument'"), (MixedAnnotatedAndDefaultStyleError(argument_name='my_argument', annotated_param_type=typer.models.OptionInfo, default_param_type=typer.models.OptionInfo), "Cannot specify `Option` in `Annotated` and default value together for 'my_argument'"), (MixedAnnotatedAndDefaultStyleError(argument_name='my_argument', annotated_param_type=typer.models.ArgumentInfo, default_param_type=typer.models.ArgumentInfo), "Cannot specify `Argument` in `Annotated` and default value together for 'my_argument'"), (MultipleTyperAnnotationsError(argument_name='my_argument'), "Cannot specify multiple `Annotated` Typer arguments for 'my_argument'"), (DefaultFactoryAndDefaultValueError(argument_name='my_argument', param_type=typer.models.OptionInfo), 'Cannot specify `default_factory` and a default value together for `Option`')])
def test_error_rendering(error, message):
    if False:
        return 10
    assert str(error) == message