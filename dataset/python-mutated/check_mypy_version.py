import re
import sys
from pathlib import Path
from mypy.plugin import Plugin

def get_correct_mypy_version():
    if False:
        i = 10
        return i + 15
    (match,) = re.finditer('mypy==(\\d+(?:\\.\\d+)*)', (Path(__file__).parent.parent / '.ci' / 'docker' / 'requirements-ci.txt').read_text())
    (version,) = match.groups()
    return version

def plugin(version: str):
    if False:
        i = 10
        return i + 15
    correct_version = get_correct_mypy_version()
    if version != correct_version:
        print(f'You are using mypy version {version}, which is not supported\nin the PyTorch repo. Please switch to mypy version {correct_version}.\n\nFor example, if you installed mypy via pip, run this:\n\n    pip install mypy=={correct_version}\n\nOr if you installed mypy via conda, run this:\n\n    conda install -c conda-forge mypy={correct_version}\n', file=sys.stderr)
    return Plugin