"""
Take normal chart CHANGELOG entries and build ArtifactHub changelog annotations.
Only outputs the annotations for the latest release in the CHANGELOG.

e.g from:

New Features
------------

- Add resources for `cleanup` and `createuser` jobs (#19263)

to:

- kind: added
  description: Add resources for `cleanup` and `createuser` jobs
  links:
    - name: "#19263"
      url: https://github.com/apache/airflow/pull/19263
"""
from __future__ import annotations
import re
import yaml
TYPE_MAPPING = {'New Features': ('added', None), 'Improvements': ('changed', None), 'Bug Fixes': ('fixed', None), 'Doc only changes': ('changed', 'Docs'), 'Misc': ('changed', 'Misc')}
PREFIXES_TO_STRIP = ['Chart:', 'Chart Docs:']

def parse_line(line: str) -> tuple[str | None, int | None]:
    if False:
        print('Hello World!')
    match = re.search('^- (.*?)(?:\\(#(\\d+)\\)){0,1}$', line)
    if not match:
        return (None, None)
    (desc, pr_number) = match.groups()
    return (desc.strip(), int(pr_number))

def print_entry(section: str, description: str, pr_number: int | None):
    if False:
        while True:
            i = 10
    for unwanted_prefix in PREFIXES_TO_STRIP:
        if description.lower().startswith(unwanted_prefix.lower()):
            description = description[len(unwanted_prefix):].strip()
    (kind, prefix) = TYPE_MAPPING[section]
    if prefix:
        description = f'{prefix}: {description}'
    entry: dict[str, str | list] = {'kind': kind, 'description': description}
    if pr_number:
        entry['links'] = [{'name': f'#{pr_number}', 'url': f'https://github.com/apache/airflow/pull/{pr_number}'}]
    print(yaml.dump([entry]))
in_first_release = False
past_significant_changes = False
section = ''
with open('chart/RELEASE_NOTES.rst') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(('"""', '----', '^^^^')):
            pass
        elif line.startswith('Airflow Helm Chart'):
            if in_first_release:
                break
            in_first_release = True
        elif not past_significant_changes:
            if line == 'New Features':
                section = line
                past_significant_changes = True
        elif not line.startswith('- '):
            section = line
        else:
            (description, pr) = parse_line(line)
            if description:
                print_entry(section, description, pr)