import sys
import types
from collections import defaultdict
from functools import lru_cache, reduce
from os import sep
from pathlib import Path
from hypothesis._settings import Phase, Verbosity
from hypothesis.internal.escalation import is_hypothesis_file

@lru_cache(maxsize=None)
def should_trace_file(fname):
    if False:
        while True:
            i = 10
    return not (is_hypothesis_file(fname) or fname.startswith('<'))
MONITORING_TOOL_ID = 3
if sys.version_info[:2] >= (3, 12):
    MONITORING_EVENTS = {sys.monitoring.events.LINE: 'trace_line'}

class Tracer:
    """A super-simple branch coverage tracer."""
    __slots__ = ('branches', '_previous_location')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.branches = set()
        self._previous_location = None

    def trace(self, frame, event, arg):
        if False:
            for i in range(10):
                print('nop')
        if event == 'call':
            return self.trace
        elif event == 'line':
            fname = frame.f_code.co_filename
            if should_trace_file(fname):
                current_location = (fname, frame.f_lineno)
                self.branches.add((self._previous_location, current_location))
                self._previous_location = current_location

    def trace_line(self, code: types.CodeType, line_number: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        fname = code.co_filename
        if should_trace_file(fname):
            current_location = (fname, line_number)
            self.branches.add((self._previous_location, current_location))
            self._previous_location = current_location

    def __enter__(self):
        if False:
            return 10
        if sys.version_info[:2] < (3, 12):
            assert sys.gettrace() is None
            sys.settrace(self.trace)
            return self
        sys.monitoring.use_tool_id(MONITORING_TOOL_ID, 'scrutineer')
        for (event, callback_name) in MONITORING_EVENTS.items():
            sys.monitoring.set_events(MONITORING_TOOL_ID, event)
            callback = getattr(self, callback_name)
            sys.monitoring.register_callback(MONITORING_TOOL_ID, event, callback)
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            return 10
        if sys.version_info[:2] < (3, 12):
            sys.settrace(None)
            return
        sys.monitoring.free_tool_id(MONITORING_TOOL_ID)
        for event in MONITORING_EVENTS:
            sys.monitoring.register_callback(MONITORING_TOOL_ID, event, None)
UNHELPFUL_LOCATIONS = (f'{sep}contextlib.py', f'{sep}inspect.py', f'{sep}re.py', f'{sep}re{sep}__init__.py', f'{sep}warnings.py', f'{sep}_pytest{sep}assertion{sep}__init__.py', f'{sep}_pytest{sep}assertion{sep}rewrite.py', f'{sep}_pytest{sep}_io{sep}saferepr.py', f'{sep}pluggy{sep}_result.py')

def get_explaining_locations(traces):
    if False:
        i = 10
        return i + 15
    if not traces:
        return {}
    unions = {origin: set().union(*values) for (origin, values) in traces.items()}
    seen_passing = {None}.union(*unions.pop(None, set()))
    always_failing_never_passing = {origin: reduce(set.intersection, [set().union(*v) for v in values]) - seen_passing for (origin, values) in traces.items() if origin is not None}
    cf_graphs = {origin: defaultdict(set) for origin in unions}
    for (origin, seen_arcs) in unions.items():
        for (src, dst) in seen_arcs:
            cf_graphs[origin][src].add(dst)
        assert cf_graphs[origin][None], 'Expected start node with >=1 successor'
    explanations = defaultdict(set)
    for origin in unions:
        queue = {None}
        seen = set()
        while queue:
            assert queue.isdisjoint(seen), f'Intersection: {queue & seen}'
            src = queue.pop()
            seen.add(src)
            if src in always_failing_never_passing[origin]:
                explanations[origin].add(src)
            else:
                queue.update(cf_graphs[origin][src] - seen)
    return {origin: {loc for loc in afnp_locs if not loc[0].endswith(UNHELPFUL_LOCATIONS)} for (origin, afnp_locs) in explanations.items()}
LIB_DIR = str(Path(sys.executable).parent / 'lib')
EXPLANATION_STUB = ('Explanation:', '    These lines were always and only run by failing examples:')

def make_report(explanations, cap_lines_at=5):
    if False:
        i = 10
        return i + 15
    report = defaultdict(list)
    for (origin, locations) in explanations.items():
        report_lines = [f'        {fname}:{lineno}' for (fname, lineno) in locations]
        report_lines.sort(key=lambda line: (line.startswith(LIB_DIR), line))
        if len(report_lines) > cap_lines_at + 1:
            msg = '        (and {} more with settings.verbosity >= verbose)'
            report_lines[cap_lines_at:] = [msg.format(len(report_lines[cap_lines_at:]))]
        if report_lines:
            report[origin] = list(EXPLANATION_STUB) + report_lines
    return report

def explanatory_lines(traces, settings):
    if False:
        return 10
    if Phase.explain in settings.phases and sys.gettrace() and (not traces):
        return defaultdict(list)
    explanations = get_explaining_locations(traces)
    max_lines = 5 if settings.verbosity <= Verbosity.normal else float('inf')
    return make_report(explanations, cap_lines_at=max_lines)