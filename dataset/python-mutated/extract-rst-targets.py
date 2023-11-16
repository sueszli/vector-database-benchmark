import os
import re
from typing import Dict, Iterator
tgt_pat = re.compile('^.. _(\\S+?):$', re.MULTILINE)
title_pat = re.compile('^(.+)\n[-=^#*]{5,}$', re.MULTILINE)

def find_explicit_targets(text: str) -> Iterator[str]:
    if False:
        while True:
            i = 10
    for m in tgt_pat.finditer(text):
        yield m.group(1)

def find_page_title(text: str) -> str:
    if False:
        while True:
            i = 10
    for m in title_pat.finditer(text):
        return m.group(1)
    return ''

def main() -> Dict[str, Dict[str, str]]:
    if False:
        for i in range(10):
            print('nop')
    refs = {}
    docs = {}
    base = os.path.dirname(os.path.abspath(__file__))
    for (dirpath, dirnames, filenames) in os.walk(base):
        if 'generated' in dirnames:
            dirnames.remove('generated')
        for f in filenames:
            if f.endswith('.rst'):
                with open(os.path.join(dirpath, f)) as stream:
                    raw = stream.read()
                href = os.path.relpath(stream.name, base).replace(os.sep, '/')
                href = href.rpartition('.')[0] + '/'
                docs[href.rstrip('/')] = find_page_title(raw)
                first_line = raw.lstrip('\n').partition('\n')[0]
                first_target_added = False
                for explicit_target in find_explicit_targets(raw):
                    if not first_target_added:
                        first_target_added = True
                        if first_line.startswith(f'.. _{explicit_target}:'):
                            refs[explicit_target] = href
                            continue
                    refs[explicit_target] = href + f"#{explicit_target.replace('_', '-')}"
    return {'ref': refs, 'doc': docs}
if __name__ == '__main__':
    import json
    print(json.dumps(main(), indent=2))