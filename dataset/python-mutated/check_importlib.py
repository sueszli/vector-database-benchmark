import importlib.abc
import pathlib
import sys
import zipfile
if sys.version_info >= (3, 9):

    def traverse(t: importlib.abc.Traversable) -> None:
        if False:
            return 10
        pass
    traverse(pathlib.Path())
    traverse(zipfile.Path(''))