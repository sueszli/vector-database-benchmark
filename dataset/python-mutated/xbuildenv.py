from pathlib import Path
import typer
from ..create_xbuildenv import create
from ..install_xbuildenv import install
from ..logger import logger
app = typer.Typer(hidden=True, no_args_is_help=True)

@app.callback()
def callback():
    if False:
        print('Hello World!')
    '\n    Create or install cross build environment\n    '

@app.command('install')
def _install(path: Path=typer.Option('.pyodide-xbuildenv', help='path to xbuildenv directory'), download: bool=typer.Option(False, help='download xbuildenv before installing'), url: str=typer.Option(None, help='URL to download xbuildenv from')) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Install xbuildenv.\n\n    The installed environment is the same as the one that would result from\n    `PYODIDE_PACKAGES='scipy' make` except that it is much faster.\n    The goal is to enable out-of-tree builds for binary packages that depend\n    on numpy or scipy.\n    Note: this is a private endpoint that should not be used outside of the Pyodide Makefile.\n    "
    install(path, download=download, url=url)
    logger.info(f'xbuildenv installed at {path.resolve()}')

@app.command('create')
def _create(path: Path=typer.Argument('.pyodide-xbuildenv', help='path to xbuildenv directory'), root: Path=typer.Option(None, help='path to pyodide root directory, if not given, will be inferred'), skip_missing_files: bool=typer.Option(False, help='skip if cross build files are missing instead of raising an error. This is useful for testing.')) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create xbuildenv.\n\n    The create environment is then used to cross-compile packages out-of-tree.\n    Note: this is a private endpoint that should not be used outside of the Pyodide Makefile.\n    '
    create(path, pyodide_root=root, skip_missing_files=skip_missing_files)
    logger.info(f'xbuildenv created at {path.resolve()}')