import os
from pathlib import Path

def make_file(path: str) -> None:
    if False:
        i = 10
        return i + 15
    Path(path).touch()

def make_valid_symlink(path: str) -> None:
    if False:
        i = 10
        return i + 15
    target = path + '1'
    make_file(target)
    os.symlink(target, path)

def make_broken_symlink(path: str) -> None:
    if False:
        return 10
    os.symlink('foo', path)

def make_dir(path: str) -> None:
    if False:
        while True:
            i = 10
    os.mkdir(path)