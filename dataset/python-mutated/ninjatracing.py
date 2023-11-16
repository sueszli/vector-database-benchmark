"""Converts one (or several) .ninja_log files into chrome's about:tracing format.

If clang -ftime-trace .json files are found adjacent to generated files they
are embedded.

Usage:
    ninja -C $BUILDDIR
    python ninjatracing.py $BUILDDIR/.ninja_log > trace.json

Then load trace.json into Chrome or into https://ui.perfetto.dev/ to see
the profiling results.
"""
import json
import os
import argparse
import re
import sys

class Target:
    """Represents a single line read for a .ninja_log file. Start and end times
    are milliseconds."""

    def __init__(self, start, end):
        if False:
            i = 10
            return i + 15
        self.start = int(start)
        self.end = int(end)
        self.targets = []

def read_targets(log, show_all):
    if False:
        while True:
            i = 10
    'Reads all targets from .ninja_log file |log_file|, sorted by start\n    time'
    header = log.readline()
    m = re.search('^# ninja log v(\\d+)\\n$', header)
    assert m, 'unrecognized ninja log version %r' % header
    version = int(m.group(1))
    assert 5 <= version <= 6, 'unsupported ninja log version %d' % version
    if version == 6:
        next(log)
    targets = {}
    last_end_seen = 0
    for line in log:
        (start, end, _, name, cmdhash) = line.strip().split('\t')
        if not show_all and int(end) < last_end_seen:
            targets = {}
        last_end_seen = int(end)
        targets.setdefault(cmdhash, Target(start, end)).targets.append(name)
    return sorted(targets.values(), key=lambda job: job.end, reverse=True)

class Threads:
    """Tries to reconstruct the parallelism from a .ninja_log"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.workers = []

    def alloc(self, target):
        if False:
            for i in range(10):
                print('nop')
        'Places target in an available thread, or adds a new thread.'
        for worker in range(len(self.workers)):
            if self.workers[worker] >= target.end:
                self.workers[worker] = target.start
                return worker
        self.workers.append(target.start)
        return len(self.workers) - 1

def read_events(trace, options):
    if False:
        print('Hello World!')
    'Reads all events from time-trace json file |trace|.'
    trace_data = json.load(trace)

    def include_event(event, options):
        if False:
            return 10
        'Only include events if they are complete events, are longer than\n        granularity, and are not totals.'
        return event['ph'] == 'X' and event['dur'] >= options['granularity'] and (not event['name'].startswith('Total'))
    return [x for x in trace_data['traceEvents'] if include_event(x, options)]

def trace_to_dicts(target, trace, options, pid, tid):
    if False:
        for i in range(10):
            print('nop')
    'Read a file-like object |trace| containing -ftime-trace data and yields\n    about:tracing dict per eligible event in that log.'
    for event in read_events(trace, options):
        ninja_time = (target.end - target.start) * 1000
        if event['dur'] > ninja_time:
            print('Inconsistent timing found (clang time > ninja time). Please ensure that timings are from consistent builds.')
            sys.exit(1)
        event['pid'] = pid
        event['tid'] = tid
        event['ts'] += target.start * 1000
        yield event

def embed_time_trace(ninja_log_dir, target, pid, tid, options):
    if False:
        i = 10
        return i + 15
    'Produce time trace output for the specified ninja target. Expects\n    time-trace file to be in .json file named based on .o file.'
    for t in target.targets:
        o_path = os.path.join(ninja_log_dir, t)
        json_trace_path = os.path.splitext(o_path)[0] + '.json'
        try:
            with open(json_trace_path, 'r') as trace:
                for time_trace_event in trace_to_dicts(target, trace, options, pid, tid):
                    yield time_trace_event
        except OSError:
            pass

def log_to_dicts(log, pid, options):
    if False:
        print('Hello World!')
    'Reads a file-like object |log| containing a .ninja_log, and yields one\n    about:tracing dict per command found in the log.'
    threads = Threads()
    for target in read_targets(log, options['showall']):
        tid = threads.alloc(target)
        yield {'name': '%0s' % ', '.join(target.targets), 'cat': 'targets', 'ph': 'X', 'ts': target.start * 1000, 'dur': (target.end - target.start) * 1000, 'pid': pid, 'tid': tid, 'args': {}}
        if options.get('embed_time_trace', False):
            try:
                ninja_log_dir = os.path.dirname(log.name)
            except AttributeError:
                continue
            for time_trace in embed_time_trace(ninja_log_dir, target, pid, tid, options):
                yield time_trace

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    usage = __doc__
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('logfiles', nargs='*', help=argparse.SUPPRESS)
    parser.add_argument('-a', '--showall', action='store_true', dest='showall', default=False, help='report on last build step for all outputs. Default is to report just on the last (possibly incremental) build')
    parser.add_argument('-g', '--granularity', type=int, default=50000, dest='granularity', help='minimum length time-trace event to embed in microseconds. Default: 50000')
    parser.add_argument('-e', '--embed-time-trace', action='store_true', default=False, dest='embed_time_trace', help='embed clang -ftime-trace json file found adjacent to a target file')
    options = parser.parse_args()
    entries = []
    for (pid, log_file) in enumerate(options.logfiles):
        with open(log_file, 'r') as log:
            entries += list(log_to_dicts(log, pid, vars(options)))
    json.dump(entries, sys.stdout)
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))