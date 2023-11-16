"""
Module to update db migration information in Airflow
"""
from __future__ import annotations
import os
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from alembic.script import ScriptDirectory
from tabulate import tabulate
from airflow import __version__ as airflow_version
from airflow.utils.db import _get_alembic_config
if TYPE_CHECKING:
    from alembic.script import Script
airflow_version = re.match('(\\d+\\.\\d+\\.\\d+).*', airflow_version).group(1)
project_root = Path(__file__).parents[2].resolve()

def replace_text_between(file: Path, start: str, end: str, replacement_text: str):
    if False:
        return 10
    original_text = file.read_text()
    leading_text = original_text.split(start)[0]
    trailing_text = original_text.split(end)[1]
    file.write_text(leading_text + start + replacement_text + end + trailing_text)

def wrap_backticks(val):
    if False:
        while True:
            i = 10

    def _wrap_backticks(x):
        if False:
            i = 10
            return i + 15
        return f'``{x}``'
    return ',\n'.join(map(_wrap_backticks, val)) if isinstance(val, (tuple, list)) else _wrap_backticks(val)

def update_doc(file, data):
    if False:
        print('Hello World!')
    replace_text_between(file=file, start=' .. Beginning of auto-generated table\n', end=' .. End of auto-generated table\n', replacement_text='\n' + tabulate(headers={'revision': 'Revision ID', 'down_revision': 'Revises ID', 'version': 'Airflow Version', 'description': 'Description'}, tabular_data=data, tablefmt='grid', stralign='left', disable_numparse=True) + '\n\n')

def has_version(content):
    if False:
        print('Hello World!')
    return re.search('^airflow_version\\s*=.*', content, flags=re.MULTILINE) is not None

def insert_version(old_content, file):
    if False:
        for i in range(10):
            print('nop')
    new_content = re.sub('(^depends_on.*)', lambda x: f"{x.group(1)}\nairflow_version = '{airflow_version}'", old_content, flags=re.MULTILINE)
    file.write_text(new_content)

def revision_suffix(rev: Script):
    if False:
        i = 10
        return i + 15
    if rev.is_head:
        return ' (head)'
    if rev.is_base:
        return ' (base)'
    if rev.is_merge_point:
        return ' (merge_point)'
    if rev.is_branch_point:
        return ' (branch_point)'
    return ''

def ensure_airflow_version(revisions: Iterable[Script]):
    if False:
        for i in range(10):
            print('nop')
    for rev in revisions:
        assert rev.module.__file__ is not None
        file = Path(rev.module.__file__)
        content = file.read_text()
        if not has_version(content):
            insert_version(content, file)

def get_revisions() -> Iterable[Script]:
    if False:
        i = 10
        return i + 15
    config = _get_alembic_config()
    script = ScriptDirectory.from_config(config)
    yield from script.walk_revisions()

def update_docs(revisions: Iterable[Script]):
    if False:
        return 10
    doc_data = []
    for rev in revisions:
        doc_data.append(dict(revision=wrap_backticks(rev.revision) + revision_suffix(rev), down_revision=wrap_backticks(rev.down_revision), version=wrap_backticks(rev.module.airflow_version), description='\n'.join(textwrap.wrap(rev.doc, width=60))))
    update_doc(file=project_root / 'docs' / 'apache-airflow' / 'migrations-ref.rst', data=doc_data)

def ensure_mod_prefix(mod_name, idx, version):
    if False:
        while True:
            i = 10
    parts = [f'{idx + 1:04}', *version]
    match = re.match('([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_(.+)', mod_name)
    if match:
        parts.append(match.group(5))
    else:
        match = re.match('([a-z0-9]+)_(.+)', mod_name)
        if match:
            parts.append(match.group(2))
    return '_'.join(parts)

def ensure_filenames_are_sorted(revisions):
    if False:
        return 10
    renames = []
    is_branched = False
    unmerged_heads = []
    for (idx, rev) in enumerate(revisions):
        mod_path = Path(rev.module.__file__)
        version = rev.module.airflow_version.split('.')[0:3]
        correct_mod_basename = ensure_mod_prefix(mod_path.name, idx, version)
        if mod_path.name != correct_mod_basename:
            renames.append((mod_path, Path(mod_path.parent, correct_mod_basename)))
        if is_branched and rev.is_merge_point:
            is_branched = False
        if rev.is_branch_point:
            is_branched = True
        elif rev.is_head:
            unmerged_heads.append(rev.revision)
    if is_branched:
        head_prefixes = [x[0:4] for x in unmerged_heads]
        alembic_command = "alembic merge -m 'merge heads " + ', '.join(head_prefixes) + "' " + ' '.join(unmerged_heads)
        raise SystemExit(f'You have multiple alembic heads; please merge them with the `alembic merge` command and re-run pre-commit. It should fail once more before succeeding. \nhint: `{alembic_command}`')
    for (old, new) in renames:
        os.rename(old, new)
if __name__ == '__main__':
    revisions = list(reversed(list(get_revisions())))
    ensure_airflow_version(revisions=revisions)
    revisions = list(reversed(list(get_revisions())))
    ensure_filenames_are_sorted(revisions=revisions)
    revisions = list(get_revisions())
    update_docs(revisions=revisions)