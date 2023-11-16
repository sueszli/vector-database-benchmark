"""Path operations."""
from __future__ import annotations
import json
import os
import re
import shutil
from pathlib import Path
from reflex import constants
join = os.linesep.join

def rm(path: str):
    if False:
        return 10
    'Remove a file or directory.\n\n    Args:\n        path: The path to the file or directory.\n    '
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)

def cp(src: str, dest: str, overwrite: bool=True) -> bool:
    if False:
        while True:
            i = 10
    'Copy a file or directory.\n\n    Args:\n        src: The path to the file or directory.\n        dest: The path to the destination.\n        overwrite: Whether to overwrite the destination.\n\n    Returns:\n        Whether the copy was successful.\n    '
    if src == dest:
        return False
    if not overwrite and os.path.exists(dest):
        return False
    if os.path.isdir(src):
        rm(dest)
        shutil.copytree(src, dest)
    else:
        shutil.copyfile(src, dest)
    return True

def mv(src: str, dest: str, overwrite: bool=True) -> bool:
    if False:
        return 10
    'Move a file or directory.\n\n    Args:\n        src: The path to the file or directory.\n        dest: The path to the destination.\n        overwrite: Whether to overwrite the destination.\n\n    Returns:\n        Whether the move was successful.\n    '
    if src == dest:
        return False
    if not overwrite and os.path.exists(dest):
        return False
    rm(dest)
    shutil.move(src, dest)
    return True

def mkdir(path: str):
    if False:
        print('Hello World!')
    'Create a directory.\n\n    Args:\n        path: The path to the directory.\n    '
    os.makedirs(path, exist_ok=True)

def ln(src: str, dest: str, overwrite: bool=False) -> bool:
    if False:
        while True:
            i = 10
    'Create a symbolic link.\n\n    Args:\n        src: The path to the file or directory.\n        dest: The path to the destination.\n        overwrite: Whether to overwrite the destination.\n\n    Returns:\n        Whether the link was successful.\n    '
    if src == dest:
        return False
    if not overwrite and (os.path.exists(dest) or os.path.islink(dest)):
        return False
    if os.path.isdir(src):
        rm(dest)
        os.symlink(src, dest, target_is_directory=True)
    else:
        os.symlink(src, dest)
    return True

def which(program: str) -> str | None:
    if False:
        i = 10
        return i + 15
    'Find the path to an executable.\n\n    Args:\n        program: The name of the executable.\n\n    Returns:\n        The path to the executable.\n    '
    return shutil.which(program)

def get_node_bin_path() -> str | None:
    if False:
        return 10
    'Get the node binary dir path.\n\n    Returns:\n        The path to the node bin folder.\n    '
    if not os.path.exists(constants.Node.BIN_PATH):
        str_path = which('node')
        return str(Path(str_path).parent) if str_path else str_path
    return constants.Node.BIN_PATH

def get_node_path() -> str | None:
    if False:
        return 10
    'Get the node binary path.\n\n    Returns:\n        The path to the node binary file.\n    '
    if not os.path.exists(constants.Node.PATH):
        return which('node')
    return constants.Node.PATH

def get_npm_path() -> str | None:
    if False:
        return 10
    'Get npm binary path.\n\n    Returns:\n        The path to the npm binary file.\n    '
    if not os.path.exists(constants.Node.PATH):
        return which('npm')
    return constants.Node.NPM_PATH

def update_json_file(file_path: str, update_dict: dict[str, int | str]):
    if False:
        print('Hello World!')
    'Update the contents of a json file.\n\n    Args:\n        file_path: the path to the JSON file.\n        update_dict: object to update json.\n    '
    fp = Path(file_path)
    fp.touch(exist_ok=True)
    fp.write_text('{}') if fp.stat().st_size == 0 else None
    json_object = {}
    if fp.stat().st_size == 0:
        with open(fp) as f:
            json_object = json.load(f)
    json_object.update(update_dict)
    with open(fp, 'w') as f:
        json.dump(json_object, f, ensure_ascii=False)

def find_replace(directory: str, find: str, replace: str):
    if False:
        for i in range(10):
            print('nop')
    'Recursively find and replace text in files in a directory.\n\n    Args:\n        directory: The directory to search.\n        find: The text to find.\n        replace: The text to replace.\n    '
    for (root, _dirs, files) in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            text = re.sub(find, replace, text)
            with open(filepath, 'w') as f:
                f.write(text)