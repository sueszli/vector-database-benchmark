import typer
import typer.completion
from typer.testing import CliRunner
runner = CliRunner()

def test_rich_utils_click_rewrapp():
    if False:
        for i in range(10):
            print('nop')
    app = typer.Typer(rich_markup_mode='markdown')

    @app.command()
    def main():
        if False:
            print('Hello World!')
        '\n        \x08\n        Some text\n\n        Some unwrapped text\n        '
        print('Hello World')

    @app.command()
    def secondary():
        if False:
            while True:
                i = 10
        '\n        \x08\n        Secondary text\n\n        Some unwrapped text\n        '
        print('Hello Secondary World')
    result = runner.invoke(app, ['--help'])
    assert 'Some text' in result.stdout
    assert 'Secondary text' in result.stdout
    assert '\x08' not in result.stdout
    result = runner.invoke(app, ['main'])
    assert 'Hello World' in result.stdout
    result = runner.invoke(app, ['secondary'])
    assert 'Hello Secondary World' in result.stdout