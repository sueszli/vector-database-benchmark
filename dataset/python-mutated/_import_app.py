from __future__ import annotations
import os
import runpy
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from textual.app import App

class AppFail(Exception):
    pass

def shebang_python(candidate: Path) -> bool:
    if False:
        while True:
            i = 10
    "Does the given file look like it's run with Python?\n\n    Args:\n        candidate: The candidate file to check.\n\n    Returns:\n        ``True`` if it looks to #! python, ``False`` if not.\n    "
    try:
        with candidate.open('rb') as source:
            first_line = source.readline()
    except IOError:
        return False
    return first_line.startswith(b'#!') and b'python' in first_line

def import_app(import_name: str) -> App:
    if False:
        return 10
    'Import an app from a path or import name.\n\n    Args:\n        import_name: A name to import, such as `foo.bar`, or a path ending with .py.\n\n    Raises:\n        AppFail: If the app could not be found for any reason.\n\n    Returns:\n        A Textual application\n    '
    import importlib
    import inspect
    from textual.app import WINDOWS, App
    (import_name, *argv) = shlex.split(import_name, posix=not WINDOWS)
    (drive, import_name) = os.path.splitdrive(import_name)
    (lib, _colon, name) = import_name.partition(':')
    if drive:
        lib = os.path.join(drive, os.sep, lib)
    if lib.endswith('.py') or shebang_python(Path(lib)):
        path = os.path.abspath(lib)
        sys.path.append(str(Path(path).parent))
        try:
            global_vars = runpy.run_path(path, {})
        except Exception as error:
            raise AppFail(str(error))
        sys.argv[:] = [path, *argv]
        if name:
            try:
                app = global_vars[name]
            except KeyError:
                raise AppFail(f'App {name!r} not found in {lib!r}')
        elif 'app' in global_vars:
            try:
                app = global_vars['app']
            except KeyError:
                raise AppFail(f'App {name!r} not found in {lib!r}')
        else:
            apps = [value for value in global_vars.values() if isinstance(value, App) or ((inspect.isclass(value) and issubclass(value, App)) and value is not App)]
            if not apps:
                raise AppFail(f'Unable to find app in {lib!r}, try specifying app with "foo.py:app"')
            if len(apps) > 1:
                raise AppFail(f'Multiple apps found {lib!r}, try specifying app with "foo.py:app"')
            app = apps[0]
        app._BASE_PATH = path
    else:
        sys.path.append('')
        try:
            module = importlib.import_module(lib)
        except ImportError as error:
            raise AppFail(str(error))
        find_app = name or 'app'
        try:
            app = getattr(module, find_app or 'app')
        except AttributeError:
            raise AppFail(f'Unable to find {find_app!r} in {module!r}')
        sys.argv[:] = [import_name, *argv]
    if inspect.isclass(app) and issubclass(app, App):
        app = app()
    return cast(App, app)