import hashlib
import json
from typing import Any
import click
from deepdiff import DeepDiff
SECRET_MASK = '**********'

def hash_config(configuration: dict) -> str:
    if False:
        while True:
            i = 10
    'Computes a SHA256 hash from a dictionnary.\n\n    Args:\n        configuration (dict): The configuration to hash\n\n    Returns:\n        str: _description_\n    '
    stringified = json.dumps(configuration, sort_keys=True)
    return hashlib.sha256(stringified.encode('utf-8')).hexdigest()

def exclude_secrets_from_diff(obj: Any, path: str) -> bool:
    if False:
        print('Hello World!')
    'Callback function used with DeepDiff to ignore secret values from the diff.\n\n    Args:\n        obj (Any): Object for which a diff will be computed.\n        path (str): unused.\n\n    Returns:\n        bool: Whether to ignore the object from the diff.\n    '
    if isinstance(obj, str):
        return True if SECRET_MASK in obj else False
    else:
        return False

def compute_diff(a: Any, b: Any) -> DeepDiff:
    if False:
        return 10
    'Wrapper around the DeepDiff computation.\n\n    Args:\n        a (Any): Object to compare with b.\n        b (Any): Object to compare with a.\n\n    Returns:\n        DeepDiff: the computed diff object.\n    '
    return DeepDiff(a, b, view='tree', exclude_obj_callback=exclude_secrets_from_diff)

def display_diff_line(diff_line: str) -> None:
    if False:
        while True:
            i = 10
    'Prettify a diff line and print it to standard output.\n\n    Args:\n        diff_line (str): The diff line to display.\n    '
    if 'changed from' in diff_line:
        color = 'yellow'
        prefix = 'E'
    elif 'added' in diff_line:
        color = 'green'
        prefix = '+'
    elif 'removed' in diff_line:
        color = 'red'
        prefix = '-'
    else:
        prefix = ''
        color = None
    click.echo(click.style(f'\t{prefix} - {diff_line}', fg=color))