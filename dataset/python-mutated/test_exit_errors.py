import errno
import typer
import typer.completion
from typer.testing import CliRunner
runner = CliRunner()

def test_eoferror():
    if False:
        for i in range(10):
            print('nop')
    app = typer.Typer()

    @app.command()
    def main():
        if False:
            i = 10
            return i + 15
        raise EOFError()
    result = runner.invoke(app)
    assert result.exit_code == 1

def test_oserror():
    if False:
        i = 10
        return i + 15
    app = typer.Typer()

    @app.command()
    def main():
        if False:
            return 10
        e = OSError()
        e.errno = errno.EPIPE
        raise e
    result = runner.invoke(app)
    assert result.exit_code == 1

def test_oserror_no_epipe():
    if False:
        return 10
    app = typer.Typer()

    @app.command()
    def main():
        if False:
            i = 10
            return i + 15
        raise OSError()
    result = runner.invoke(app)
    assert result.exit_code == 1