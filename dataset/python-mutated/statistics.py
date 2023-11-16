import math
from collections import Counter
from hypothesis._settings import Phase
from hypothesis.utils.dynamicvariables import DynamicVariable
collector = DynamicVariable(None)

def note_statistics(stats_dict):
    if False:
        for i in range(10):
            print('nop')
    callback = collector.value
    if callback is not None:
        callback(stats_dict)

def describe_targets(best_targets):
    if False:
        while True:
            i = 10
    'Return a list of lines describing the results of `target`, if any.'
    if not best_targets:
        return []
    elif len(best_targets) == 1:
        (label, score) = next(iter(best_targets.items()))
        return [f'Highest target score: {score:g}  (label={label!r})']
    else:
        lines = ['Highest target scores:']
        for (label, score) in sorted(best_targets.items(), key=lambda x: x[::-1]):
            lines.append(f'{score:>16g}  (label={label!r})')
        return lines

def format_ms(times):
    if False:
        while True:
            i = 10
    'Format `times` into a string representing approximate milliseconds.\n\n    `times` is a collection of durations in seconds.\n    '
    ordered = sorted(times)
    n = len(ordered) - 1
    assert n >= 0
    lower = int(ordered[int(math.floor(n * 0.05))] * 1000)
    upper = int(ordered[int(math.ceil(n * 0.95))] * 1000)
    if upper == 0:
        return '< 1ms'
    elif lower == upper:
        return f'~ {lower}ms'
    else:
        return f'~ {lower}-{upper} ms'

def describe_statistics(stats_dict):
    if False:
        print('Hello World!')
    "Return a multi-line string describing the passed run statistics.\n\n    `stats_dict` must be a dictionary of data in the format collected by\n    `hypothesis.internal.conjecture.engine.ConjectureRunner.statistics`.\n\n    We DO NOT promise that this format will be stable or supported over\n    time, but do aim to make it reasonably useful for downstream users.\n    It's also meant to support benchmarking for research purposes.\n\n    This function is responsible for the report which is printed in the\n    terminal for our pytest --hypothesis-show-statistics option.\n    "
    lines = [stats_dict['nodeid'] + ':\n'] if 'nodeid' in stats_dict else []
    prev_failures = 0
    for phase in (p.name for p in list(Phase)[1:]):
        d = stats_dict.get(phase + '-phase', {})
        cases = d.get('test-cases', [])
        if not cases:
            continue
        statuses = Counter((t['status'] for t in cases))
        runtime_ms = format_ms((t['runtime'] for t in cases))
        drawtime_ms = format_ms((t['drawtime'] for t in cases))
        lines.append(f"  - during {phase} phase ({d['duration-seconds']:.2f} seconds):\n    - Typical runtimes: {runtime_ms}, of which {drawtime_ms} in data generation\n    - {statuses['valid']} passing examples, {statuses['interesting']} failing examples, {statuses['invalid'] + statuses['overrun']} invalid examples")
        distinct_failures = d['distinct-failures'] - prev_failures
        if distinct_failures:
            plural = distinct_failures > 1
            lines.append('    - Found {}{} distinct error{} in this phase'.format(distinct_failures, ' more' * bool(prev_failures), 's' * plural))
        prev_failures = d['distinct-failures']
        if phase == 'generate':
            events = Counter(sum((t['events'] for t in cases), []))
            if events:
                lines.append('    - Events:')
                lines += [f'      * {100 * v / len(cases):.2f}%, {k}' for (k, v) in sorted(events.items(), key=lambda x: (-x[1], x[0]))]
        if phase == 'shrink':
            lines.append('    - Tried {} shrinks of which {} were successful'.format(len(cases), d['shrinks-successful']))
        lines.append('')
    target_lines = describe_targets(stats_dict.get('targets', {}))
    if target_lines:
        lines.append('  - ' + target_lines[0])
        lines.extend(('    ' + l for l in target_lines[1:]))
    lines.append('  - Stopped because ' + stats_dict['stopped-because'])
    return '\n'.join(lines)