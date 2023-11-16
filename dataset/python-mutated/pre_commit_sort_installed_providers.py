from __future__ import annotations
from pathlib import Path
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit(f'This file is intended to be executed as an executable program. You cannot use it as a module.To run this script, run the ./{__file__} command')
AIRFLOW_SOURCES = Path(__file__).parents[3].resolve()

def stable_sort(x):
    if False:
        i = 10
        return i + 15
    return (x.casefold(), x)

def sort_uniq(sequence):
    if False:
        while True:
            i = 10
    return sorted(set(sequence), key=stable_sort)
if __name__ == '__main__':
    installed_providers_path = Path(AIRFLOW_SOURCES) / 'airflow' / 'providers' / 'installed_providers.txt'
    content = installed_providers_path.read_text().splitlines(keepends=True)
    sorted_content = sort_uniq(content)
    installed_providers_path.write_text(''.join(sorted_content))