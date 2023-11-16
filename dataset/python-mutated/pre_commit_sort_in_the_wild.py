from __future__ import annotations
import re
import sys
from pathlib import Path
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit(f'This file is intended to be executed as an executable program. You cannot use it as a module.To run this script, run the ./{__file__} command')
AIRFLOW_SOURCES = Path(__file__).parents[3].resolve()
NUMBER_MATCH = re.compile('(^\\d+\\.)')

def stable_sort(x):
    if False:
        return 10
    return (x.casefold(), x)
if __name__ == '__main__':
    inthewild_path = Path(AIRFLOW_SOURCES) / 'INTHEWILD.md'
    content = inthewild_path.read_text()
    header = []
    companies = []
    in_header = True
    for (index, line) in enumerate(content.splitlines(keepends=True)):
        if in_header:
            header.append(line)
            if 'Currently, **officially** using Airflow:' in line:
                in_header = False
        else:
            if line.strip() == '':
                continue
            match = NUMBER_MATCH.match(line)
            if not match:
                print(f"\x1b[0;31mERROR: The {index + 1} line in `INTHEWILD.md` should begin with '1.'. Please fix it !\x1b[0m\n")
                print(line)
                print()
                sys.exit(1)
            if not line.startswith('1.'):
                print(f"\x1b[0;33mWARNING: The {index + 1} line in `INTHEWILD.md` should begin with '1.' but it starts with {match.group(1)} Replacing the number with 1.\x1b[0m\n")
                old_line = line
                line = '1.' + line.split('.', maxsplit=1)[1]
                print(f'{old_line.strip()} => {line.strip()}')
            companies.append(line)
    companies.sort(key=stable_sort)
    inthewild_path.write_text(''.join(header) + '\n' + ''.join(companies))