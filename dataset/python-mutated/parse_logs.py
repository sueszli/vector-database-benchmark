import csv
import os
import re
import sys
assert len(sys.argv) == 2
full_log = open(sys.argv[1]).read()
gist_url = ''
m = re.search('https://gist.github.com/[a-f0-9]+', full_log)
if m is not None:
    gist_url = m.group(0)
entries = re.split('(?:cuda (?:train|eval) +([^ ]+)|WARNING:root:([^ ]+) failed to load)', full_log)[1:]

def chunker(seq, size):
    if False:
        i = 10
        return i + 15
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
c = 0
i = 0
out = csv.DictWriter(sys.stdout, ['bench', 'name', 'result', 'component', 'context', 'explain', 'frame_time', 'backend_time', 'graph_count', 'op_count', 'graph_breaks', 'unique_graph_breaks'], dialect='excel')
out.writeheader()
out.writerow({'explain': gist_url})

def normalize_file(f):
    if False:
        while True:
            i = 10
    if 'site-packages/' in f:
        return f.split('site-packages/', 2)[1]
    else:
        return os.path.relpath(f)
bench = 'torchbench'
for (name, name2, log) in chunker(entries, 3):
    if name is None:
        name = name2
    if name.startswith('Albert'):
        bench = 'huggingface'
    elif name.startswith('adv_inc'):
        bench = 'timm_models'
    r = 'UNKNOWN'
    explain = ''
    component = ''
    context = ''
    if 'PASS' in log:
        r = 'PASS'
    if 'TIMEOUT' in log:
        r = 'FAIL TIMEOUT'
    if 'Accuracy failed' in log:
        r = 'FAIL ACCURACY'
    log = log.split('The above exception was the direct cause of the following exception')[0]
    split = log.split('Traceback (most recent call last)', maxsplit=1)
    if len(split) == 2:
        log = split[1]
    log = log.split('Original traceback:')[0]
    m = re.search('File "([^"]+)", line ([0-9]+), in .+\\n +(.+)\\n([A-Za-z]+(?:Error|Exception|NotImplementedError): ?.*)', log)
    if m is not None:
        r = 'FAIL'
        component = f'{normalize_file(m.group(1))}:{m.group(2)}'
        context = m.group(3)
        explain = f'{m.group(4)}'
    else:
        m = re.search('File "([^"]+)", line ([0-9]+), in .+\\n +(.+)\\nAssertionError', log)
        if m is not None:
            r = 'FAIL'
            component = f'{normalize_file(m.group(1))}:{m.group(2)}'
            context = m.group(3)
            explain = 'AssertionError'
    if 'FAIL' in log:
        r = 'FAIL'
    if r == 'UNKNOWN':
        c += 1
    backend_time = None
    frame_time = None
    if 'TIMING:' in log:
        result = re.search('TIMING:(.*)\n', log).group(1)
        split_str = result.split('backend_compile:')
        if len(split_str) == 2:
            backend_time = float(split_str[1])
            frame_time = float(split_str[0].split('entire_frame_compile:')[1])
    if 'STATS:' in log:
        result = re.search('STATS:(.*)\n', log).group(1)
        split_all = result.split('|')
    graph_count = None
    op_count = None
    graph_breaks = None
    unique_graph_breaks = None
    if (m := re.search('Dynamo produced (\\d+) graphs covering (\\d+) ops with (\\d+) graph breaks \\((\\d+) unique\\)', log)):
        graph_count = m.group(1)
        op_count = m.group(2)
        graph_breaks = m.group(3)
        unique_graph_breaks = m.group(4)
    if len(context) > 78:
        context = ''
    if '/tmp/' in component:
        component = 'generated code'
        context = ''
    out.writerow({'bench': bench, 'name': name, 'result': r, 'component': component, 'context': context, 'explain': explain, 'frame_time': frame_time, 'backend_time': backend_time, 'graph_count': graph_count, 'op_count': op_count, 'graph_breaks': graph_breaks, 'unique_graph_breaks': unique_graph_breaks})
    i += 1
if c:
    print(f'failed to classify {c} entries', file=sys.stderr)