"""
Utilities for Prefect CLI commands
"""
import functools
import traceback
import typer
import typer.core
from click.exceptions import ClickException
from prefect.exceptions import MissingProfileError
from prefect.settings import PREFECT_TEST_MODE

def exit_with_error(message, code=1, **kwargs):
    if False:
        return 10
    '\n    Utility to print a stylized error message and exit with a non-zero code\n    '
    from prefect.cli.root import app
    kwargs.setdefault('style', 'red')
    app.console.print(message, **kwargs)
    raise typer.Exit(code)

def exit_with_success(message, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Utility to print a stylized success message and exit with a zero code\n    '
    from prefect.cli.root import app
    kwargs.setdefault('style', 'green')
    app.console.print(message, **kwargs)
    raise typer.Exit(0)

def with_cli_exception_handling(fn):
    if False:
        return 10

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            return fn(*args, **kwargs)
        except (typer.Exit, typer.Abort, ClickException):
            raise
        except MissingProfileError as exc:
            exit_with_error(exc)
        except Exception:
            if PREFECT_TEST_MODE.value():
                raise
            traceback.print_exc()
            exit_with_error('An exception occurred.')
    return wrapper