"""
Utility that sorts the names in the auto mappings defines in the auto modules in alphabetical order.

Use from the root of the repo with:

```bash
python utils/sort_auto_mappings.py
```

to auto-fix all the auto mappings (used in `make style`).

To only check if the mappings are properly sorted (as used in `make quality`), do:

```bash
python utils/sort_auto_mappings.py --check_only
```
"""
import argparse
import os
import re
from typing import Optional
PATH_TO_AUTO_MODULE = 'src/transformers/models/auto'
_re_intro_mapping = re.compile('[A-Z_]+_MAPPING(\\s+|_[A-Z_]+\\s+)=\\s+OrderedDict')
_re_identifier = re.compile('\\s*\\(\\s*"(\\S[^"]+)"')

def sort_auto_mapping(fname: str, overwrite: bool=False) -> Optional[bool]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Sort all auto mappings in a file.\n\n    Args:\n        fname (`str`): The name of the file where we want to sort auto-mappings.\n        overwrite (`bool`, *optional*, defaults to `False`): Whether or not to fix and overwrite the file.\n\n    Returns:\n        `Optional[bool]`: Returns `None` if `overwrite=True`. Otherwise returns `True` if the file has an auto-mapping\n        improperly sorted, `False` if the file is okay.\n    '
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    new_lines = []
    line_idx = 0
    while line_idx < len(lines):
        if _re_intro_mapping.search(lines[line_idx]) is not None:
            indent = len(re.search('^(\\s*)\\S', lines[line_idx]).groups()[0]) + 8
            while not lines[line_idx].startswith(' ' * indent + '('):
                new_lines.append(lines[line_idx])
                line_idx += 1
            blocks = []
            while lines[line_idx].strip() != ']':
                if lines[line_idx].strip() == '(':
                    start_idx = line_idx
                    while not lines[line_idx].startswith(' ' * indent + ')'):
                        line_idx += 1
                    blocks.append('\n'.join(lines[start_idx:line_idx + 1]))
                else:
                    blocks.append(lines[line_idx])
                line_idx += 1
            blocks = sorted(blocks, key=lambda x: _re_identifier.search(x).groups()[0])
            new_lines += blocks
        else:
            new_lines.append(lines[line_idx])
            line_idx += 1
    if overwrite:
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
    else:
        return '\n'.join(new_lines) != content

def sort_all_auto_mappings(overwrite: bool=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sort all auto mappings in the library.\n\n    Args:\n        overwrite (`bool`, *optional*, defaults to `False`): Whether or not to fix and overwrite the file.\n    '
    fnames = [os.path.join(PATH_TO_AUTO_MODULE, f) for f in os.listdir(PATH_TO_AUTO_MODULE) if f.endswith('.py')]
    diffs = [sort_auto_mapping(fname, overwrite=overwrite) for fname in fnames]
    if not overwrite and any(diffs):
        failures = [f for (f, d) in zip(fnames, diffs) if d]
        raise ValueError(f"The following files have auto mappings that need sorting: {', '.join(failures)}. Run `make style` to fix this.")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_only', action='store_true', help='Whether to only check or fix style.')
    args = parser.parse_args()
    sort_all_auto_mappings(not args.check_only)