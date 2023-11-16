from __future__ import annotations
import os
import tarfile
from pathlib import Path
from shutil import rmtree

def is_relative_to(child_path: str | os.PathLike, root_path: str | os.PathLike):
    if False:
        while True:
            i = 10
    return Path(root_path).resolve() in Path(child_path).resolve().parents

def untar(archive_path: str, output_path: str, overwrite=True, strip=0) -> None:
    if False:
        for i in range(10):
            print('nop')
    if overwrite and os.path.exists(output_path):
        rmtree(output_path)
    with tarfile.open(archive_path, mode='r') as archive:
        for member in archive.getmembers():
            if not is_relative_to(Path(output_path, member.name), output_path):
                strip = -1
            member.path = member.path.split('/', strip)[-1]
        archive.extractall(output_path)