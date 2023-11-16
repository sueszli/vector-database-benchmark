import subprocess
import sys
import typer
import typer.core
from typer.testing import CliRunner
from docs_src.parameter_types.number import tutorial001_an as mod
runner = CliRunner()
app = typer.Typer()
app.command()(mod.main)

def test_help():
    if False:
        while True:
            i = 10
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert '--age' in result.output
    assert 'INTEGER RANGE' in result.output
    assert '--score' in result.output
    assert 'FLOAT RANGE' in result.output

def test_help_no_rich():
    if False:
        return 10
    rich = typer.core.rich
    typer.core.rich = None
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert '--age' in result.output
    assert 'INTEGER RANGE' in result.output
    assert '--score' in result.output
    assert 'FLOAT RANGE' in result.output
    typer.core.rich = rich

def test_params():
    if False:
        print('Hello World!')
    result = runner.invoke(app, ['5', '--age', '20', '--score', '90'])
    assert result.exit_code == 0
    assert 'ID is 5' in result.output
    assert '--age is 20' in result.output
    assert '--score is 90.0' in result.output

def test_invalid_id():
    if False:
        while True:
            i = 10
    result = runner.invoke(app, ['1002'])
    assert result.exit_code != 0
    assert "Invalid value for 'ID': 1002 is not in the range 0<=x<=1000." in result.output or "Invalid value for 'ID': 1002 is not in the valid range of 0 to 1000." in result.output

def test_invalid_age():
    if False:
        while True:
            i = 10
    result = runner.invoke(app, ['5', '--age', '15'])
    assert result.exit_code != 0
    assert "Invalid value for '--age': 15 is not in the range x>=18" in result.output or "Invalid value for '--age': 15 is smaller than the minimum valid value 18." in result.output

def test_invalid_score():
    if False:
        for i in range(10):
            print('nop')
    result = runner.invoke(app, ['5', '--age', '20', '--score', '100.5'])
    assert result.exit_code != 0
    assert "Invalid value for '--score': 100.5 is not in the range x<=100." in result.output or "Invalid value for '--score': 100.5 is bigger than the maximum valid value" in result.output

def test_negative_score():
    if False:
        print('Hello World!')
    result = runner.invoke(app, ['5', '--age', '20', '--score', '-5'])
    assert result.exit_code == 0
    assert 'ID is 5' in result.output
    assert '--age is 20' in result.output
    assert '--score is -5.0' in result.output

def test_script():
    if False:
        while True:
            i = 10
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    assert 'Usage' in result.stdout