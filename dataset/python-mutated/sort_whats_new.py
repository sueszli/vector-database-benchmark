import re
import sys
from collections import defaultdict
LABEL_ORDER = ['MajorFeature', 'Feature', 'Efficiency', 'Enhancement', 'Fix', 'API']

def entry_sort_key(s):
    if False:
        for i in range(10):
            print('nop')
    if s.startswith('- |'):
        return LABEL_ORDER.index(s.split('|')[1])
    else:
        return -1
text = ''.join((l for l in sys.stdin if l.startswith('- ') or l.startswith(' ')))
bucketed = defaultdict(list)
for entry in re.split('\n(?=- )', text.strip()):
    modules = re.findall(':(?:func|meth|mod|class):`(?:[^<`]*<|~)?(?:sklearn.)?([a-z]\\w+)', entry)
    modules = set(modules)
    if len(modules) > 1:
        key = 'Multiple modules'
    elif modules:
        key = ':mod:`sklearn.%s`' % next(iter(modules))
    else:
        key = 'Miscellaneous'
    bucketed[key].append(entry)
    entry = entry.strip() + '\n'
everything = []
for (key, bucket) in sorted(bucketed.items()):
    everything.append(key + '\n' + '.' * len(key))
    bucket.sort(key=entry_sort_key)
    everything.extend(bucket)
print('\n\n'.join(everything))