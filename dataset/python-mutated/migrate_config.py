from __future__ import annotations
import re
import textwrap
import cfgv
import yaml
from pre_commit.clientlib import InvalidConfigError
from pre_commit.yaml import yaml_load

def _is_header_line(line: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return line.startswith(('#', '---')) or not line.strip()

def _migrate_map(contents: str) -> str:
    if False:
        return 10
    if isinstance(yaml_load(contents), list):
        lines = contents.splitlines(True)
        i = 0
        while i < len(lines) and _is_header_line(lines[i]):
            i += 1
        header = ''.join(lines[:i])
        rest = ''.join(lines[i:])
        try:
            trial_contents = f'{header}repos:\n{rest}'
            yaml_load(trial_contents)
            contents = trial_contents
        except yaml.YAMLError:
            contents = f"{header}repos:\n{textwrap.indent(rest, ' ' * 4)}"
    return contents

def _migrate_sha_to_rev(contents: str) -> str:
    if False:
        return 10
    return re.sub('(\\n\\s+)sha:', '\\1rev:', contents)

def _migrate_python_venv(contents: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return re.sub('(\\n\\s+)language: python_venv\\b', '\\1language: python', contents)

def migrate_config(config_file: str, quiet: bool=False) -> int:
    if False:
        for i in range(10):
            print('nop')
    with open(config_file) as f:
        orig_contents = contents = f.read()
    with cfgv.reraise_as(InvalidConfigError):
        with cfgv.validate_context(f'File {config_file}'):
            try:
                yaml_load(orig_contents)
            except Exception as e:
                raise cfgv.ValidationError(str(e))
    contents = _migrate_map(contents)
    contents = _migrate_sha_to_rev(contents)
    contents = _migrate_python_venv(contents)
    if contents != orig_contents:
        with open(config_file, 'w') as f:
            f.write(contents)
        print('Configuration has been migrated.')
    elif not quiet:
        print('Configuration is already migrated.')
    return 0