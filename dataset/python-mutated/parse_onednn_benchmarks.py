"""Parses results from run_onednn_benchmarks.sh.

Example results:
Showing runtimes in microseconds. `?` means not available.
               Model,  Batch,        Vanilla,         oneDNN,    Speedup
          bert-large,      1,              x,              y,        x/y
          bert-large,     16,            ...,            ...,        ...
           inception,      1,            ...,            ...,        ...
           inception,     16,            ...,            ...,        ...
                                        ⋮
        ssd-resnet34,      1,              ?,            ...,          ?
        ssd-resnet34,     16,              ?,            ...,          ?

Vanilla TF can't run ssd-resnet34 on CPU because it doesn't support NCHW format.
"""
import enum
import re
import sys
db = dict()
models = set()
batch_sizes = set()
State = enum.Enum('State', 'FIND_CONFIG_OR_MODEL FIND_RUNNING_TIME')

def parse_results(lines):
    if False:
        i = 10
        return i + 15
    'Parses benchmark results from run_onednn_benchmarks.sh.\n\n  Stores results in a global dict.\n\n  Args:\n    lines: Array of strings corresponding to each line of the output from\n      run_onednn_benchmarks.sh\n\n  Raises:\n    RuntimeError: If the program reaches an unknown state.\n  '
    idx = 0
    (batch, onednn, model) = (None, None, None)
    state = State.FIND_CONFIG_OR_MODEL
    while idx < len(lines):
        if state is State.FIND_CONFIG_OR_MODEL:
            config = re.match("\\+ echo 'BATCH=(?P<batch>[\\d]+), ONEDNN=(?P<onednn>[\\d]+)", lines[idx])
            if config:
                batch = int(config.group('batch'))
                onednn = int(config.group('onednn'))
                batch_sizes.add(batch)
            else:
                model_re = re.search('tf-graphs\\/(?P<model>[\\w\\d_-]+).pb', lines[idx])
                assert model_re
                model = model_re.group('model')
                models.add(model)
                state = State.FIND_RUNNING_TIME
        elif state is State.FIND_RUNNING_TIME:
            match = re.search('no stats: (?P<avg>[\\d.]+)', lines[idx])
            state = State.FIND_CONFIG_OR_MODEL
            if match:
                avg = float(match.group('avg'))
                key = (model, batch, onednn)
                assert None not in key
                db[key] = avg
            else:
                continue
        else:
            raise RuntimeError('Reached the unreachable code.')
        idx = idx + 1

def main():
    if False:
        i = 10
        return i + 15
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        lines = f.readlines()
    parse_results(lines)
    print('Showing runtimes in microseconds. `?` means not available.')
    print('%20s, %6s, %14s, %14s, %10s' % ('Model', 'Batch', 'Vanilla', 'oneDNN', 'Speedup'))
    for model in sorted(models):
        for batch in sorted(batch_sizes):
            key = (model, batch, 0)
            eigen = db[key] if key in db else '?'
            key = (model, batch, 1)
            onednn = db[key] if key in db else '?'
            speedup = '%10.2f' % (eigen / onednn) if '?' not in (eigen, onednn) else '?'
            print('%20s, %6d, %14s, %14s, %10s' % (model, batch, str(eigen), str(onednn), speedup))
if __name__ == '__main__':
    main()