from __future__ import annotations
import hashlib
from pathlib import Path
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit(f'This file is intended to be executed as an executable program. You cannot use it as a module.To execute this script, run ./{__file__} [FILE] ...')
AIRFLOW_SOURCES_ROOT = Path(__file__).parents[3].resolve()
BREEZE_SOURCES_ROOT = AIRFLOW_SOURCES_ROOT / 'dev' / 'breeze'

def get_package_setup_metadata_hash() -> str:
    if False:
        return 10
    "\n    Retrieves hash of pyproject.toml file.\n\n    This is used in order to determine if we need to upgrade Breeze, because some\n    setup files changed. Blake2b algorithm will not be flagged by security checkers\n    as insecure algorithm (in Python 3.9 and above we can use `usedforsecurity=False`\n    to disable it, but for now it's better to use more secure algorithms.\n    "
    try:
        the_hash = hashlib.new('blake2b')
        the_hash.update((BREEZE_SOURCES_ROOT / 'pyproject.toml').read_bytes())
        return the_hash.hexdigest()
    except FileNotFoundError as e:
        return f'Missing file {e.filename}'

def process_breeze_readme():
    if False:
        return 10
    breeze_readme = BREEZE_SOURCES_ROOT / 'README.md'
    lines = breeze_readme.read_text().splitlines(keepends=True)
    result_lines = []
    for line in lines:
        if line.startswith('Package config hash:'):
            line = f'Package config hash: {get_package_setup_metadata_hash()}\n'
        result_lines.append(line)
    breeze_readme.write_text(''.join(result_lines))
if __name__ == '__main__':
    process_breeze_readme()