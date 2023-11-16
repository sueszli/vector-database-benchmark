import shutil
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from ._py_compile import _compile
from .common import make_zip_archive
REMOVED_FILES = ('ensurepip/', 'venv/', 'lib2to3/', '_osx_support.py', '_aix_support.py', 'curses/', 'dbm/', 'idlelib/', 'tkinter/', 'turtle.py', 'turtledemo')
UNVENDORED_FILES = ('test/', 'distutils/', 'sqlite3', 'ssl.py', 'lzma.py', '_pydecimal.py', 'pydoc_data')
JS_STUB_FILES = ('webbrowser.py',)

def default_filterfunc(root: Path, verbose: bool=False) -> Callable[[str, list[str]], set[str]]:
    if False:
        while True:
            i = 10
    '\n    The default filter function used by `create_zipfile`.\n\n    This function filters out several modules that are:\n\n    - not supported in Pyodide due to browser limitations (e.g. `tkinter`)\n    - unvendored from the standard library (e.g. `sqlite3`)\n    '

    def _should_skip(path: Path) -> bool:
        if False:
            return 10
        'Skip common files that are not needed in the zip file.'
        name = path.name
        if path.is_dir() and name in ('__pycache__', 'dist'):
            return True
        if path.is_dir() and name.endswith(('.egg-info', '.dist-info')):
            return True
        if path.is_file() and name in ('LICENSE', 'LICENSE.txt', 'setup.py', '.gitignore'):
            return True
        if path.is_file() and name.endswith(('pyi', 'toml', 'cfg', 'md', 'rst')):
            return True
        return False

    def filterfunc(path: Path | str, names: list[str]) -> set[str]:
        if False:
            i = 10
            return i + 15
        filtered_files = {(root / f).resolve() for f in REMOVED_FILES + UNVENDORED_FILES}
        if root.name.startswith('python3'):
            filtered_files.update({root / f for f in JS_STUB_FILES})
        path = Path(path).resolve()
        if _should_skip(path):
            return set(names)
        _names = []
        for name in names:
            fullpath = path / name
            if _should_skip(fullpath) or fullpath in filtered_files:
                if verbose:
                    print(f'Skipping {fullpath}')
                _names.append(name)
        return set(_names)
    return filterfunc

def create_zipfile(libdirs: list[Path], output: Path | str='python', pycompile: bool=False, filterfunc: Callable[[str, list[str]], set[str]] | None=None, compression_level: int=6) -> None:
    if False:
        return 10
    '\n    Bundle Python standard libraries into a zip file.\n\n    The basic idea of this function is similar to the standard library\'s\n    {ref}`zipfile.PyZipFile` class.\n\n    However, we need some additional functionality for Pyodide. For example:\n\n    - We need to remove some unvendored modules, e.g. `sqlite3`\n    - We need an option to "not" compile the files in the zip file\n\n    hence this function.\n\n    Parameters\n    ----------\n    libdirs\n        List of paths to the directory containing the Python standard library or extra packages.\n\n    output\n        Path to the output zip file. Defaults to python.zip.\n\n    pycompile\n        Whether to compile the .py files into .pyc, by default False\n\n    filterfunc\n        A function that filters the files to be included in the zip file.\n        This function will be passed to {ref}`shutil.copytree` \'s ignore argument.\n        By default, Pyodide\'s default filter function is used.\n\n    compression_level\n        Level of zip compression to apply. 0 means no compression. If a strictly\n        positive integer is provided, ZIP_DEFLATED option is used.\n\n    Returns\n    -------\n    BytesIO\n        A BytesIO object containing the zip file.\n    '
    archive = Path(output)
    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for libdir in libdirs:
            libdir = Path(libdir)
            if filterfunc is None:
                _filterfunc = default_filterfunc(libdir)
            shutil.copytree(libdir, temp_dir, ignore=_filterfunc, dirs_exist_ok=True)
        make_zip_archive(archive, temp_dir, compression_level=compression_level)
    if pycompile:
        _compile(archive, archive, verbose=False, keep=False, compression_level=compression_level)