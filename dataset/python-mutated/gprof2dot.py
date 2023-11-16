"""Generate a dot graph from the output of several profilers."""
__author__ = 'Jose Fonseca et al'
import sys
import math
import os.path
import re
import textwrap
import optparse
import xml.parsers.expat
import collections
import locale
import json
import fnmatch
assert sys.version_info[0] >= 3
MULTIPLICATION_SIGN = chr(215)

def times(x):
    if False:
        print('Hello World!')
    return '%u%s' % (x, MULTIPLICATION_SIGN)

def percentage(p):
    if False:
        while True:
            i = 10
    return '%.02f%%' % (p * 100.0,)

def add(a, b):
    if False:
        return 10
    return a + b

def fail(a, b):
    if False:
        i = 10
        return i + 15
    assert False
tol = 2 ** (-23)

def ratio(numerator, denominator):
    if False:
        i = 10
        return i + 15
    try:
        ratio = float(numerator) / float(denominator)
    except ZeroDivisionError:
        return 1.0
    if ratio < 0.0:
        if ratio < -tol:
            sys.stderr.write('warning: negative ratio (%s/%s)\n' % (numerator, denominator))
        return 0.0
    if ratio > 1.0:
        if ratio > 1.0 + tol:
            sys.stderr.write('warning: ratio greater than one (%s/%s)\n' % (numerator, denominator))
        return 1.0
    return ratio

class UndefinedEvent(Exception):
    """Raised when attempting to get an event which is undefined."""

    def __init__(self, event):
        if False:
            print('Hello World!')
        Exception.__init__(self)
        self.event = event

    def __str__(self):
        if False:
            return 10
        return 'unspecified event %s' % self.event.name

class Event(object):
    """Describe a kind of event, and its basic operations."""

    def __init__(self, name, null, aggregator, formatter=str):
        if False:
            return 10
        self.name = name
        self._null = null
        self._aggregator = aggregator
        self._formatter = formatter

    def __repr__(self):
        if False:
            return 10
        return self.name

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self is other

    def __hash__(self):
        if False:
            print('Hello World!')
        return id(self)

    def null(self):
        if False:
            i = 10
            return i + 15
        return self._null

    def aggregate(self, val1, val2):
        if False:
            print('Hello World!')
        'Aggregate two event values.'
        assert val1 is not None
        assert val2 is not None
        return self._aggregator(val1, val2)

    def format(self, val):
        if False:
            print('Hello World!')
        'Format an event value.'
        assert val is not None
        return self._formatter(val)
CALLS = Event('Calls', 0, add, times)
SAMPLES = Event('Samples', 0, add, times)
SAMPLES2 = Event('Samples', 0, add, times)
TOTAL_SAMPLES = Event('Samples', 0, add, times)
TIME = Event('Time', 0.0, add, lambda x: '(' + str(x) + ')')
TIME_RATIO = Event('Time ratio', 0.0, add, lambda x: '(' + percentage(x) + ')')
TOTAL_TIME = Event('Total time', 0.0, fail)
TOTAL_TIME_RATIO = Event('Total time ratio', 0.0, fail, percentage)
labels = {'self-time': TIME, 'self-time-percentage': TIME_RATIO, 'total-time': TOTAL_TIME, 'total-time-percentage': TOTAL_TIME_RATIO}
defaultLabelNames = ['total-time-percentage', 'self-time-percentage']
totalMethod = 'callratios'

class Object(object):
    """Base class for all objects in profile which can store events."""

    def __init__(self, events=None):
        if False:
            print('Hello World!')
        if events is None:
            self.events = {}
        else:
            self.events = events

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return id(self)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self is other

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return id(self) < id(other)

    def __contains__(self, event):
        if False:
            while True:
                i = 10
        return event in self.events

    def __getitem__(self, event):
        if False:
            i = 10
            return i + 15
        try:
            return self.events[event]
        except KeyError:
            raise UndefinedEvent(event)

    def __setitem__(self, event, value):
        if False:
            i = 10
            return i + 15
        if value is None:
            if event in self.events:
                del self.events[event]
        else:
            self.events[event] = value

class Call(Object):
    """A call between functions.

    There should be at most one call object for every pair of functions.
    """

    def __init__(self, callee_id):
        if False:
            while True:
                i = 10
        Object.__init__(self)
        self.callee_id = callee_id
        self.ratio = None
        self.weight = None

class Function(Object):
    """A function."""

    def __init__(self, id, name):
        if False:
            print('Hello World!')
        Object.__init__(self)
        self.id = id
        self.name = name
        self.module = None
        self.process = None
        self.calls = {}
        self.called = None
        self.weight = None
        self.cycle = None
        self.filename = None

    def add_call(self, call):
        if False:
            print('Hello World!')
        if call.callee_id in self.calls:
            sys.stderr.write('warning: overwriting call from function %s to %s\n' % (str(self.id), str(call.callee_id)))
        self.calls[call.callee_id] = call

    def get_call(self, callee_id):
        if False:
            while True:
                i = 10
        if not callee_id in self.calls:
            call = Call(callee_id)
            call[SAMPLES] = 0
            call[SAMPLES2] = 0
            call[CALLS] = 0
            self.calls[callee_id] = call
        return self.calls[callee_id]
    _parenthesis_re = re.compile('\\([^()]*\\)')
    _angles_re = re.compile('<[^<>]*>')
    _const_re = re.compile('\\s+const$')

    def stripped_name(self):
        if False:
            print('Hello World!')
        'Remove extraneous information from C++ demangled function names.'
        name = self.name
        while True:
            (name, n) = self._parenthesis_re.subn('', name)
            if not n:
                break
        name = self._const_re.sub('', name)
        while True:
            (name, n) = self._angles_re.subn('', name)
            if not n:
                break
        return name

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.name

    def dump(self, sep1=',\n\t', sep2=':=', sep3='\n'):
        if False:
            print('Hello World!')
        ' Returns as a string all information available in this Function object\n            separators sep1:between entries\n                       sep2:between attribute name and value,\n                       sep3: inserted at end\n        '
        return sep1.join((sep2.join([k, str(v)]) for (k, v) in sorted(self.__dict__.items()))) + sep3

class Cycle(Object):
    """A cycle made from recursive function calls."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        Object.__init__(self)
        self.functions = set()

    def add_function(self, function):
        if False:
            i = 10
            return i + 15
        assert function not in self.functions
        self.functions.add(function)
        if function.cycle is not None:
            for other in function.cycle.functions:
                if function not in self.functions:
                    self.add_function(other)
        function.cycle = self

class Profile(Object):
    """The whole profile."""

    def __init__(self):
        if False:
            print('Hello World!')
        Object.__init__(self)
        self.functions = {}
        self.cycles = []

    def add_function(self, function):
        if False:
            return 10
        if function.id in self.functions:
            sys.stderr.write('warning: overwriting function %s (id %s)\n' % (function.name, str(function.id)))
        self.functions[function.id] = function

    def add_cycle(self, cycle):
        if False:
            i = 10
            return i + 15
        self.cycles.append(cycle)

    def validate(self):
        if False:
            for i in range(10):
                print('nop')
        'Validate the edges.'
        for function in self.functions.values():
            for callee_id in list(function.calls.keys()):
                assert function.calls[callee_id].callee_id == callee_id
                if callee_id not in self.functions:
                    sys.stderr.write('warning: call to undefined function %s from function %s\n' % (str(callee_id), function.name))
                    del function.calls[callee_id]

    def find_cycles(self):
        if False:
            return 10
        "Find cycles using Tarjan's strongly connected components algorithm."
        stack = []
        data = {}
        order = 0
        for function in self.functions.values():
            order = self._tarjan(function, order, stack, data)
        cycles = []
        for function in self.functions.values():
            if function.cycle is not None and function.cycle not in cycles:
                cycles.append(function.cycle)
        self.cycles = cycles
        if 0:
            for cycle in cycles:
                sys.stderr.write('Cycle:\n')
                for member in cycle.functions:
                    sys.stderr.write('\tFunction %s\n' % member.name)

    def prune_root(self, roots, depth=-1):
        if False:
            while True:
                i = 10
        visited = set()
        frontier = set([(root_node, depth) for root_node in roots])
        while len(frontier) > 0:
            (node, node_depth) = frontier.pop()
            visited.add(node)
            if node_depth == 0:
                continue
            f = self.functions[node]
            newNodes = set(f.calls.keys()) - visited
            frontier = frontier.union({(new_node, node_depth - 1) for new_node in newNodes})
        subtreeFunctions = {}
        for n in visited:
            f = self.functions[n]
            newCalls = {}
            for c in f.calls.keys():
                if c in visited:
                    newCalls[c] = f.calls[c]
            f.calls = newCalls
            subtreeFunctions[n] = f
        self.functions = subtreeFunctions

    def prune_leaf(self, leafs, depth=-1):
        if False:
            return 10
        edgesUp = collections.defaultdict(set)
        for f in self.functions.keys():
            for n in self.functions[f].calls.keys():
                edgesUp[n].add(f)
        visited = set()
        frontier = set([(leaf_node, depth) for leaf_node in leafs])
        while len(frontier) > 0:
            (node, node_depth) = frontier.pop()
            visited.add(node)
            if node_depth == 0:
                continue
            newNodes = edgesUp[node] - visited
            frontier = frontier.union({(new_node, node_depth - 1) for new_node in newNodes})
        downTree = set(self.functions.keys())
        upTree = visited
        path = downTree.intersection(upTree)
        pathFunctions = {}
        for n in path:
            f = self.functions[n]
            newCalls = {}
            for c in f.calls.keys():
                if c in path:
                    newCalls[c] = f.calls[c]
            f.calls = newCalls
            pathFunctions[n] = f
        self.functions = pathFunctions

    def getFunctionIds(self, funcName):
        if False:
            i = 10
            return i + 15
        function_names = {v.name: k for (k, v) in self.functions.items()}
        return [function_names[name] for name in fnmatch.filter(function_names.keys(), funcName)]

    def getFunctionId(self, funcName):
        if False:
            return 10
        for f in self.functions:
            if self.functions[f].name == funcName:
                return f
        return False

    def printFunctionIds(self, selector=None, file=sys.stderr):
        if False:
            i = 10
            return i + 15
        ' Print to file function entries selected by fnmatch.fnmatch like in\n            method getFunctionIds, with following extensions:\n             - selector starts with "%": dump all information available\n             - selector is \'+\' or \'-\': select all function entries\n        '
        if selector is None or selector in ('+', '*'):
            v = ',\n'.join(('%s:\t%s' % (kf, self.functions[kf].name) for kf in self.functions.keys()))
        elif selector[0] == '%':
            selector = selector[1:]
            function_info = {k: v for (k, v) in self.functions.items() if fnmatch.fnmatch(v.name, selector)}
            v = ',\n'.join(('%s\t({k})\t(%s)::\n\t%s' % (v.name, type(v), v.dump()) for (k, v) in function_info.items()))
        else:
            function_names = (v.name for v in self.functions.values())
            v = ',\n'.join((nm for nm in fnmatch.filter(function_names, selector)))
        file.write(v + '\n')
        file.flush()

    class _TarjanData:

        def __init__(self, order):
            if False:
                while True:
                    i = 10
            self.order = order
            self.lowlink = order
            self.onstack = False

    def _tarjan(self, function, order, stack, data):
        if False:
            return 10
        "Tarjan's strongly connected components algorithm.\n\n        See also:\n        - http://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm\n        "
        try:
            func_data = data[function.id]
            return order
        except KeyError:
            func_data = self._TarjanData(order)
            data[function.id] = func_data
        order += 1
        pos = len(stack)
        stack.append(function)
        func_data.onstack = True
        for call in function.calls.values():
            try:
                callee_data = data[call.callee_id]
                if callee_data.onstack:
                    func_data.lowlink = min(func_data.lowlink, callee_data.order)
            except KeyError:
                callee = self.functions[call.callee_id]
                order = self._tarjan(callee, order, stack, data)
                callee_data = data[call.callee_id]
                func_data.lowlink = min(func_data.lowlink, callee_data.lowlink)
        if func_data.lowlink == func_data.order:
            members = stack[pos:]
            del stack[pos:]
            if len(members) > 1:
                cycle = Cycle()
                for member in members:
                    cycle.add_function(member)
                    data[member.id].onstack = False
            else:
                for member in members:
                    data[member.id].onstack = False
        return order

    def call_ratios(self, event):
        if False:
            i = 10
            return i + 15
        cycle_totals = {}
        for cycle in self.cycles:
            cycle_totals[cycle] = 0.0
        function_totals = {}
        for function in self.functions.values():
            function_totals[function] = 0.0
        for function in self.functions.values():
            for call in function.calls.values():
                if call.callee_id != function.id:
                    callee = self.functions[call.callee_id]
                    if event in call.events:
                        function_totals[callee] += call[event]
                        if callee.cycle is not None and callee.cycle is not function.cycle:
                            cycle_totals[callee.cycle] += call[event]
                    else:
                        sys.stderr.write('call_ratios: No data for ' + function.name + ' call to ' + callee.name + '\n')
        for function in self.functions.values():
            for call in function.calls.values():
                assert call.ratio is None
                if call.callee_id != function.id:
                    callee = self.functions[call.callee_id]
                    if event in call.events:
                        if callee.cycle is not None and callee.cycle is not function.cycle:
                            total = cycle_totals[callee.cycle]
                        else:
                            total = function_totals[callee]
                        call.ratio = ratio(call[event], total)
                    else:
                        call.ratio = 0.0

    def integrate(self, outevent, inevent):
        if False:
            for i in range(10):
                print('nop')
        'Propagate function time ratio along the function calls.\n\n        Must be called after finding the cycles.\n\n        See also:\n        - http://citeseer.ist.psu.edu/graham82gprof.html\n        '
        assert outevent not in self
        for function in self.functions.values():
            assert outevent not in function
            assert inevent in function
            for call in function.calls.values():
                assert outevent not in call
                if call.callee_id != function.id:
                    assert call.ratio is not None
        for cycle in self.cycles:
            total = inevent.null()
            for function in self.functions.values():
                total = inevent.aggregate(total, function[inevent])
            self[inevent] = total
        total = inevent.null()
        for function in self.functions.values():
            total = inevent.aggregate(total, function[inevent])
            self._integrate_function(function, outevent, inevent)
        self[outevent] = total

    def _integrate_function(self, function, outevent, inevent):
        if False:
            i = 10
            return i + 15
        if function.cycle is not None:
            return self._integrate_cycle(function.cycle, outevent, inevent)
        else:
            if outevent not in function:
                total = function[inevent]
                for call in function.calls.values():
                    if call.callee_id != function.id:
                        total += self._integrate_call(call, outevent, inevent)
                function[outevent] = total
            return function[outevent]

    def _integrate_call(self, call, outevent, inevent):
        if False:
            for i in range(10):
                print('nop')
        assert outevent not in call
        assert call.ratio is not None
        callee = self.functions[call.callee_id]
        subtotal = call.ratio * self._integrate_function(callee, outevent, inevent)
        call[outevent] = subtotal
        return subtotal

    def _integrate_cycle(self, cycle, outevent, inevent):
        if False:
            i = 10
            return i + 15
        if outevent not in cycle:
            total = inevent.null()
            for member in cycle.functions:
                subtotal = member[inevent]
                for call in member.calls.values():
                    callee = self.functions[call.callee_id]
                    if callee.cycle is not cycle:
                        subtotal += self._integrate_call(call, outevent, inevent)
                total += subtotal
            cycle[outevent] = total
            callees = {}
            for function in self.functions.values():
                if function.cycle is not cycle:
                    for call in function.calls.values():
                        callee = self.functions[call.callee_id]
                        if callee.cycle is cycle:
                            try:
                                callees[callee] += call.ratio
                            except KeyError:
                                callees[callee] = call.ratio
            for member in cycle.functions:
                member[outevent] = outevent.null()
            for (callee, call_ratio) in callees.items():
                ranks = {}
                call_ratios = {}
                partials = {}
                self._rank_cycle_function(cycle, callee, ranks)
                self._call_ratios_cycle(cycle, callee, ranks, call_ratios, set())
                partial = self._integrate_cycle_function(cycle, callee, call_ratio, partials, ranks, call_ratios, outevent, inevent)
                max_partial = max(partials.values())
                assert abs(partial - max_partial) <= 1e-07 * max_partial
                assert abs(call_ratio * total - partial) <= 0.001 * call_ratio * total
        return cycle[outevent]

    def _rank_cycle_function(self, cycle, function, ranks):
        if False:
            print('Hello World!')
        "Dijkstra's shortest paths algorithm.\n\n        See also:\n        - http://en.wikipedia.org/wiki/Dijkstra's_algorithm\n        "
        import heapq
        Q = []
        Qd = {}
        p = {}
        visited = set([function])
        ranks[function] = 0
        for call in function.calls.values():
            if call.callee_id != function.id:
                callee = self.functions[call.callee_id]
                if callee.cycle is cycle:
                    ranks[callee] = 1
                    item = [ranks[callee], function, callee]
                    heapq.heappush(Q, item)
                    Qd[callee] = item
        while Q:
            (cost, parent, member) = heapq.heappop(Q)
            if member not in visited:
                p[member] = parent
                visited.add(member)
                for call in member.calls.values():
                    if call.callee_id != member.id:
                        callee = self.functions[call.callee_id]
                        if callee.cycle is cycle:
                            member_rank = ranks[member]
                            rank = ranks.get(callee)
                            if rank is not None:
                                if rank > 1 + member_rank:
                                    rank = 1 + member_rank
                                    ranks[callee] = rank
                                    Qd_callee = Qd[callee]
                                    Qd_callee[0] = rank
                                    Qd_callee[1] = member
                                    heapq._siftdown(Q, 0, Q.index(Qd_callee))
                            else:
                                rank = 1 + member_rank
                                ranks[callee] = rank
                                item = [rank, member, callee]
                                heapq.heappush(Q, item)
                                Qd[callee] = item

    def _call_ratios_cycle(self, cycle, function, ranks, call_ratios, visited):
        if False:
            return 10
        if function not in visited:
            visited.add(function)
            for call in function.calls.values():
                if call.callee_id != function.id:
                    callee = self.functions[call.callee_id]
                    if callee.cycle is cycle:
                        if ranks[callee] > ranks[function]:
                            call_ratios[callee] = call_ratios.get(callee, 0.0) + call.ratio
                            self._call_ratios_cycle(cycle, callee, ranks, call_ratios, visited)

    def _integrate_cycle_function(self, cycle, function, partial_ratio, partials, ranks, call_ratios, outevent, inevent):
        if False:
            return 10
        if function not in partials:
            partial = partial_ratio * function[inevent]
            for call in function.calls.values():
                if call.callee_id != function.id:
                    callee = self.functions[call.callee_id]
                    if callee.cycle is not cycle:
                        assert outevent in call
                        partial += partial_ratio * call[outevent]
                    elif ranks[callee] > ranks[function]:
                        callee_partial = self._integrate_cycle_function(cycle, callee, partial_ratio, partials, ranks, call_ratios, outevent, inevent)
                        call_ratio = ratio(call.ratio, call_ratios[callee])
                        call_partial = call_ratio * callee_partial
                        try:
                            call[outevent] += call_partial
                        except UndefinedEvent:
                            call[outevent] = call_partial
                        partial += call_partial
            partials[function] = partial
            try:
                function[outevent] += partial
            except UndefinedEvent:
                function[outevent] = partial
        return partials[function]

    def aggregate(self, event):
        if False:
            return 10
        'Aggregate an event for the whole profile.'
        total = event.null()
        for function in self.functions.values():
            try:
                total = event.aggregate(total, function[event])
            except UndefinedEvent:
                return
        self[event] = total

    def ratio(self, outevent, inevent):
        if False:
            print('Hello World!')
        assert outevent not in self
        assert inevent in self
        for function in self.functions.values():
            assert outevent not in function
            assert inevent in function
            function[outevent] = ratio(function[inevent], self[inevent])
            for call in function.calls.values():
                assert outevent not in call
                if inevent in call:
                    call[outevent] = ratio(call[inevent], self[inevent])
        self[outevent] = 1.0

    def prune(self, node_thres, edge_thres, paths, color_nodes_by_selftime):
        if False:
            for i in range(10):
                print('nop')
        'Prune the profile'
        for function in self.functions.values():
            try:
                function.weight = function[TOTAL_TIME_RATIO]
            except UndefinedEvent:
                pass
            for call in function.calls.values():
                callee = self.functions[call.callee_id]
                if TOTAL_TIME_RATIO in call:
                    call.weight = call[TOTAL_TIME_RATIO]
                else:
                    try:
                        call.weight = min(function[TOTAL_TIME_RATIO], callee[TOTAL_TIME_RATIO])
                    except UndefinedEvent:
                        pass
        for function_id in list(self.functions.keys()):
            function = self.functions[function_id]
            if function.weight is not None:
                if function.weight < node_thres:
                    del self.functions[function_id]
        for function_id in list(self.functions.keys()):
            function = self.functions[function_id]
            if paths and function.filename and (not any((function.filename.startswith(path) for path in paths))):
                del self.functions[function_id]
            elif paths and function.module and (not any((function.module.find(path) > -1 for path in paths))):
                del self.functions[function_id]
        for function in self.functions.values():
            for callee_id in list(function.calls.keys()):
                call = function.calls[callee_id]
                if callee_id not in self.functions or (call.weight is not None and call.weight < edge_thres):
                    del function.calls[callee_id]
        if color_nodes_by_selftime:
            weights = []
            for function in self.functions.values():
                try:
                    weights.append(function[TIME_RATIO])
                except UndefinedEvent:
                    pass
            max_ratio = max(weights or [1])
            for function in self.functions.values():
                try:
                    function.weight = function[TIME_RATIO] / max_ratio
                except (ZeroDivisionError, UndefinedEvent):
                    pass

    def dump(self):
        if False:
            print('Hello World!')
        for function in self.functions.values():
            sys.stderr.write('Function %s:\n' % (function.name,))
            self._dump_events(function.events)
            for call in function.calls.values():
                callee = self.functions[call.callee_id]
                sys.stderr.write('  Call %s:\n' % (callee.name,))
                self._dump_events(call.events)
        for cycle in self.cycles:
            sys.stderr.write('Cycle:\n')
            self._dump_events(cycle.events)
            for function in cycle.functions:
                sys.stderr.write('  Function %s\n' % (function.name,))

    def _dump_events(self, events):
        if False:
            for i in range(10):
                print('nop')
        for (event, value) in events.items():
            sys.stderr.write('    %s: %s\n' % (event.name, event.format(value)))

class Struct:
    """Masquerade a dictionary with a structure-like behavior."""

    def __init__(self, attrs=None):
        if False:
            return 10
        if attrs is None:
            attrs = {}
        self.__dict__['_attrs'] = attrs

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        self._attrs[name] = value

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self._attrs)

    def __repr__(self):
        if False:
            return 10
        return repr(self._attrs)

class ParseError(Exception):
    """Raised when parsing to signal mismatches."""

    def __init__(self, msg, line):
        if False:
            for i in range(10):
                print('nop')
        Exception.__init__(self)
        self.msg = msg
        self.line = line

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s: %r' % (self.msg, self.line)

class Parser:
    """Parser interface."""
    stdinInput = True
    multipleInput = False

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def parse(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

class JsonParser(Parser):
    """Parser for a custom JSON representation of profile data.

    See schema.json for details.
    """

    def __init__(self, stream):
        if False:
            return 10
        Parser.__init__(self)
        self.stream = stream

    def parse(self):
        if False:
            return 10
        obj = json.load(self.stream)
        assert obj['version'] == 0
        profile = Profile()
        profile[SAMPLES] = 0
        fns = obj['functions']
        for functionIndex in range(len(fns)):
            fn = fns[functionIndex]
            function = Function(functionIndex, fn['name'])
            try:
                function.module = fn['module']
            except KeyError:
                pass
            try:
                function.process = fn['process']
            except KeyError:
                pass
            function[SAMPLES] = 0
            function.called = 0
            profile.add_function(function)
        for event in obj['events']:
            callchain = []
            for functionIndex in event['callchain']:
                function = profile.functions[functionIndex]
                callchain.append(function)
            function = profile.functions[event['callchain'][0]]
            function.called = function.called + 1
            cost = event['cost'][0]
            callee = callchain[0]
            callee[SAMPLES] += cost
            profile[SAMPLES] += cost
            for caller in callchain[1:]:
                try:
                    call = caller.calls[callee.id]
                except KeyError:
                    call = Call(callee.id)
                    call[SAMPLES2] = cost
                    caller.add_call(call)
                else:
                    call[SAMPLES2] += cost
                callee = caller
        if False:
            profile.dump()
        profile.validate()
        profile.find_cycles()
        profile.ratio(TIME_RATIO, SAMPLES)
        profile.call_ratios(SAMPLES2)
        profile.integrate(TOTAL_TIME_RATIO, TIME_RATIO)
        return profile

class LineParser(Parser):
    """Base class for parsers that read line-based formats."""

    def __init__(self, stream):
        if False:
            return 10
        Parser.__init__(self)
        self._stream = stream
        self.__line = None
        self.__eof = False
        self.line_no = 0

    def readline(self):
        if False:
            print('Hello World!')
        line = self._stream.readline()
        if not line:
            self.__line = ''
            self.__eof = True
        else:
            self.line_no += 1
        line = line.rstrip('\r\n')
        self.__line = line

    def lookahead(self):
        if False:
            i = 10
            return i + 15
        assert self.__line is not None
        return self.__line

    def consume(self):
        if False:
            return 10
        assert self.__line is not None
        line = self.__line
        self.readline()
        return line

    def eof(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.__line is not None
        return self.__eof
(XML_ELEMENT_START, XML_ELEMENT_END, XML_CHARACTER_DATA, XML_EOF) = range(4)

class XmlToken:

    def __init__(self, type, name_or_data, attrs=None, line=None, column=None):
        if False:
            while True:
                i = 10
        assert type in (XML_ELEMENT_START, XML_ELEMENT_END, XML_CHARACTER_DATA, XML_EOF)
        self.type = type
        self.name_or_data = name_or_data
        self.attrs = attrs
        self.line = line
        self.column = column

    def __str__(self):
        if False:
            print('Hello World!')
        if self.type == XML_ELEMENT_START:
            return '<' + self.name_or_data + ' ...>'
        if self.type == XML_ELEMENT_END:
            return '</' + self.name_or_data + '>'
        if self.type == XML_CHARACTER_DATA:
            return self.name_or_data
        if self.type == XML_EOF:
            return 'end of file'
        assert 0

class XmlTokenizer:
    """Expat based XML tokenizer."""

    def __init__(self, fp, skip_ws=True):
        if False:
            while True:
                i = 10
        self.fp = fp
        self.tokens = []
        self.index = 0
        self.final = False
        self.skip_ws = skip_ws
        self.character_pos = (0, 0)
        self.character_data = ''
        self.parser = xml.parsers.expat.ParserCreate()
        self.parser.StartElementHandler = self.handle_element_start
        self.parser.EndElementHandler = self.handle_element_end
        self.parser.CharacterDataHandler = self.handle_character_data

    def handle_element_start(self, name, attributes):
        if False:
            return 10
        self.finish_character_data()
        (line, column) = self.pos()
        token = XmlToken(XML_ELEMENT_START, name, attributes, line, column)
        self.tokens.append(token)

    def handle_element_end(self, name):
        if False:
            i = 10
            return i + 15
        self.finish_character_data()
        (line, column) = self.pos()
        token = XmlToken(XML_ELEMENT_END, name, None, line, column)
        self.tokens.append(token)

    def handle_character_data(self, data):
        if False:
            for i in range(10):
                print('nop')
        if not self.character_data:
            self.character_pos = self.pos()
        self.character_data += data

    def finish_character_data(self):
        if False:
            print('Hello World!')
        if self.character_data:
            if not self.skip_ws or not self.character_data.isspace():
                (line, column) = self.character_pos
                token = XmlToken(XML_CHARACTER_DATA, self.character_data, None, line, column)
                self.tokens.append(token)
            self.character_data = ''

    def next(self):
        if False:
            while True:
                i = 10
        size = 16 * 1024
        while self.index >= len(self.tokens) and (not self.final):
            self.tokens = []
            self.index = 0
            data = self.fp.read(size)
            self.final = len(data) < size
            self.parser.Parse(data, self.final)
        if self.index >= len(self.tokens):
            (line, column) = self.pos()
            token = XmlToken(XML_EOF, None, None, line, column)
        else:
            token = self.tokens[self.index]
            self.index += 1
        return token

    def pos(self):
        if False:
            print('Hello World!')
        return (self.parser.CurrentLineNumber, self.parser.CurrentColumnNumber)

class XmlTokenMismatch(Exception):

    def __init__(self, expected, found):
        if False:
            for i in range(10):
                print('nop')
        Exception.__init__(self)
        self.expected = expected
        self.found = found

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%u:%u: %s expected, %s found' % (self.found.line, self.found.column, str(self.expected), str(self.found))

class XmlParser(Parser):
    """Base XML document parser."""

    def __init__(self, fp):
        if False:
            for i in range(10):
                print('nop')
        Parser.__init__(self)
        self.tokenizer = XmlTokenizer(fp)
        self.consume()

    def consume(self):
        if False:
            i = 10
            return i + 15
        self.token = self.tokenizer.next()

    def match_element_start(self, name):
        if False:
            while True:
                i = 10
        return self.token.type == XML_ELEMENT_START and self.token.name_or_data == name

    def match_element_end(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.token.type == XML_ELEMENT_END and self.token.name_or_data == name

    def element_start(self, name):
        if False:
            return 10
        while self.token.type == XML_CHARACTER_DATA:
            self.consume()
        if self.token.type != XML_ELEMENT_START:
            raise XmlTokenMismatch(XmlToken(XML_ELEMENT_START, name), self.token)
        if self.token.name_or_data != name:
            raise XmlTokenMismatch(XmlToken(XML_ELEMENT_START, name), self.token)
        attrs = self.token.attrs
        self.consume()
        return attrs

    def element_end(self, name):
        if False:
            print('Hello World!')
        while self.token.type == XML_CHARACTER_DATA:
            self.consume()
        if self.token.type != XML_ELEMENT_END:
            raise XmlTokenMismatch(XmlToken(XML_ELEMENT_END, name), self.token)
        if self.token.name_or_data != name:
            raise XmlTokenMismatch(XmlToken(XML_ELEMENT_END, name), self.token)
        self.consume()

    def character_data(self, strip=True):
        if False:
            return 10
        data = ''
        while self.token.type == XML_CHARACTER_DATA:
            data += self.token.name_or_data
            self.consume()
        if strip:
            data = data.strip()
        return data

class GprofParser(Parser):
    """Parser for GNU gprof output.

    See also:
    - Chapter "Interpreting gprof's Output" from the GNU gprof manual
      http://sourceware.org/binutils/docs-2.18/gprof/Call-Graph.html#Call-Graph
    - File "cg_print.c" from the GNU gprof source code
      http://sourceware.org/cgi-bin/cvsweb.cgi/~checkout~/src/gprof/cg_print.c?rev=1.12&cvsroot=src
    """

    def __init__(self, fp):
        if False:
            return 10
        Parser.__init__(self)
        self.fp = fp
        self.functions = {}
        self.cycles = {}

    def readline(self):
        if False:
            while True:
                i = 10
        line = self.fp.readline()
        if not line:
            sys.stderr.write('error: unexpected end of file\n')
            sys.exit(1)
        line = line.rstrip('\r\n')
        return line
    _int_re = re.compile('^\\d+$')
    _float_re = re.compile('^\\d+\\.\\d+$')

    def translate(self, mo):
        if False:
            i = 10
            return i + 15
        'Extract a structure from a match object, while translating the types in the process.'
        attrs = {}
        groupdict = mo.groupdict()
        for (name, value) in groupdict.items():
            if value is None:
                value = None
            elif self._int_re.match(value):
                value = int(value)
            elif self._float_re.match(value):
                value = float(value)
            attrs[name] = value
        return Struct(attrs)
    _cg_header_re = re.compile('^\\s+called/total\\s+parents\\s*$|' + '^index\\s+%time\\s+self\\s+descendents\\s+called\\+self\\s+name\\s+index\\s*$|' + '^\\s+called/total\\s+children\\s*$|' + '^index\\s+%\\s+(time\\s+)?self\\s+children\\s+called\\s+name\\s*$')
    _cg_ignore_re = re.compile('^\\s+<spontaneous>\\s*$|^.*\\((\\d+)\\)$')
    _cg_primary_re = re.compile('^\\[(?P<index>\\d+)\\]?' + '\\s+(?P<percentage_time>\\d+\\.\\d+)' + '\\s+(?P<self>\\d+\\.\\d+)' + '\\s+(?P<descendants>\\d+\\.\\d+)' + '\\s+(?:(?P<called>\\d+)(?:\\+(?P<called_self>\\d+))?)?' + '\\s+(?P<name>\\S.*?)' + '(?:\\s+<cycle\\s(?P<cycle>\\d+)>)?' + '\\s\\[(\\d+)\\]$')
    _cg_parent_re = re.compile('^\\s+(?P<self>\\d+\\.\\d+)?' + '\\s+(?P<descendants>\\d+\\.\\d+)?' + '\\s+(?P<called>\\d+)(?:/(?P<called_total>\\d+))?' + '\\s+(?P<name>\\S.*?)' + '(?:\\s+<cycle\\s(?P<cycle>\\d+)>)?' + '\\s\\[(?P<index>\\d+)\\]$')
    _cg_child_re = _cg_parent_re
    _cg_cycle_header_re = re.compile('^\\[(?P<index>\\d+)\\]?' + '\\s+(?P<percentage_time>\\d+\\.\\d+)' + '\\s+(?P<self>\\d+\\.\\d+)' + '\\s+(?P<descendants>\\d+\\.\\d+)' + '\\s+(?:(?P<called>\\d+)(?:\\+(?P<called_self>\\d+))?)?' + '\\s+<cycle\\s(?P<cycle>\\d+)\\sas\\sa\\swhole>' + '\\s\\[(\\d+)\\]$')
    _cg_cycle_member_re = re.compile('^\\s+(?P<self>\\d+\\.\\d+)?' + '\\s+(?P<descendants>\\d+\\.\\d+)?' + '\\s+(?P<called>\\d+)(?:\\+(?P<called_self>\\d+))?' + '\\s+(?P<name>\\S.*?)' + '(?:\\s+<cycle\\s(?P<cycle>\\d+)>)?' + '\\s\\[(?P<index>\\d+)\\]$')
    _cg_sep_re = re.compile('^--+$')

    def parse_function_entry(self, lines):
        if False:
            for i in range(10):
                print('nop')
        parents = []
        children = []
        while True:
            if not lines:
                sys.stderr.write('warning: unexpected end of entry\n')
            line = lines.pop(0)
            if line.startswith('['):
                break
            mo = self._cg_parent_re.match(line)
            if not mo:
                if self._cg_ignore_re.match(line):
                    continue
                sys.stderr.write('warning: unrecognized call graph entry: %r\n' % line)
            else:
                parent = self.translate(mo)
                parents.append(parent)
        mo = self._cg_primary_re.match(line)
        if not mo:
            sys.stderr.write('warning: unrecognized call graph entry: %r\n' % line)
            return
        else:
            function = self.translate(mo)
        while lines:
            line = lines.pop(0)
            mo = self._cg_child_re.match(line)
            if not mo:
                if self._cg_ignore_re.match(line):
                    continue
                sys.stderr.write('warning: unrecognized call graph entry: %r\n' % line)
            else:
                child = self.translate(mo)
                children.append(child)
        function.parents = parents
        function.children = children
        self.functions[function.index] = function

    def parse_cycle_entry(self, lines):
        if False:
            for i in range(10):
                print('nop')
        line = lines[0]
        mo = self._cg_cycle_header_re.match(line)
        if not mo:
            sys.stderr.write('warning: unrecognized call graph entry: %r\n' % line)
            return
        cycle = self.translate(mo)
        cycle.functions = []
        for line in lines[1:]:
            mo = self._cg_cycle_member_re.match(line)
            if not mo:
                sys.stderr.write('warning: unrecognized call graph entry: %r\n' % line)
                continue
            call = self.translate(mo)
            cycle.functions.append(call)
        self.cycles[cycle.cycle] = cycle

    def parse_cg_entry(self, lines):
        if False:
            while True:
                i = 10
        if lines[0].startswith('['):
            self.parse_cycle_entry(lines)
        else:
            self.parse_function_entry(lines)

    def parse_cg(self):
        if False:
            for i in range(10):
                print('nop')
        'Parse the call graph.'
        while not self._cg_header_re.match(self.readline()):
            pass
        line = self.readline()
        while self._cg_header_re.match(line):
            line = self.readline()
        entry_lines = []
        while line != '\x0c':
            if line and (not line.isspace()):
                if self._cg_sep_re.match(line):
                    self.parse_cg_entry(entry_lines)
                    entry_lines = []
                else:
                    entry_lines.append(line)
            line = self.readline()

    def parse(self):
        if False:
            i = 10
            return i + 15
        self.parse_cg()
        self.fp.close()
        profile = Profile()
        profile[TIME] = 0.0
        cycles = {}
        for index in self.cycles:
            cycles[index] = Cycle()
        for entry in self.functions.values():
            function = Function(entry.index, entry.name)
            function[TIME] = entry.self
            if entry.called is not None:
                function.called = entry.called
            if entry.called_self is not None:
                call = Call(entry.index)
                call[CALLS] = entry.called_self
                function.called += entry.called_self
            for child in entry.children:
                call = Call(child.index)
                assert child.called is not None
                call[CALLS] = child.called
                if child.index not in self.functions:
                    missing = Function(child.index, child.name)
                    function[TIME] = 0.0
                    function.called = 0
                    profile.add_function(missing)
                function.add_call(call)
            profile.add_function(function)
            if entry.cycle is not None:
                try:
                    cycle = cycles[entry.cycle]
                except KeyError:
                    sys.stderr.write('warning: <cycle %u as a whole> entry missing\n' % entry.cycle)
                    cycle = Cycle()
                    cycles[entry.cycle] = cycle
                cycle.add_function(function)
            profile[TIME] = profile[TIME] + function[TIME]
        for cycle in cycles.values():
            profile.add_cycle(cycle)
        profile.validate()
        profile.ratio(TIME_RATIO, TIME)
        profile.call_ratios(CALLS)
        profile.integrate(TOTAL_TIME, TIME)
        profile.ratio(TOTAL_TIME_RATIO, TOTAL_TIME)
        return profile

class AXEParser(Parser):
    """Parser for VTune Amplifier XE 2013 gprof-cc report output."""

    def __init__(self, fp):
        if False:
            while True:
                i = 10
        Parser.__init__(self)
        self.fp = fp
        self.functions = {}
        self.cycles = {}

    def readline(self):
        if False:
            return 10
        line = self.fp.readline()
        if not line:
            sys.stderr.write('error: unexpected end of file\n')
            sys.exit(1)
        line = line.rstrip('\r\n')
        return line
    _int_re = re.compile('^\\d+$')
    _float_re = re.compile('^\\d+\\.\\d+$')

    def translate(self, mo):
        if False:
            i = 10
            return i + 15
        'Extract a structure from a match object, while translating the types in the process.'
        attrs = {}
        groupdict = mo.groupdict()
        for (name, value) in groupdict.items():
            if value is None:
                value = None
            elif self._int_re.match(value):
                value = int(value)
            elif self._float_re.match(value):
                value = float(value)
            attrs[name] = value
        return Struct(attrs)
    _cg_header_re = re.compile('^Index |^-----+ ')
    _cg_footer_re = re.compile('^Index\\s+Function\\s*$')
    _cg_primary_re = re.compile('^\\[(?P<index>\\d+)\\]?' + '\\s+(?P<percentage_time>\\d+\\.\\d+)' + '\\s+(?P<self>\\d+\\.\\d+)' + '\\s+(?P<descendants>\\d+\\.\\d+)' + '\\s+(?P<name>\\S.*?)' + '(?:\\s+<cycle\\s(?P<cycle>\\d+)>)?' + '\\s+\\[(\\d+)\\]' + '\\s*$')
    _cg_parent_re = re.compile('^\\s+(?P<self>\\d+\\.\\d+)?' + '\\s+(?P<descendants>\\d+\\.\\d+)?' + '\\s+(?P<name>\\S.*?)' + '(?:\\s+<cycle\\s(?P<cycle>\\d+)>)?' + '(?:\\s+\\[(?P<index>\\d+)\\]\\s*)?' + '\\s*$')
    _cg_child_re = _cg_parent_re
    _cg_cycle_header_re = re.compile('^\\[(?P<index>\\d+)\\]?' + '\\s+(?P<percentage_time>\\d+\\.\\d+)' + '\\s+(?P<self>\\d+\\.\\d+)' + '\\s+(?P<descendants>\\d+\\.\\d+)' + '\\s+<cycle\\s(?P<cycle>\\d+)\\sas\\sa\\swhole>' + '\\s+\\[(\\d+)\\]' + '\\s*$')
    _cg_cycle_member_re = re.compile('^\\s+(?P<self>\\d+\\.\\d+)?' + '\\s+(?P<descendants>\\d+\\.\\d+)?' + '\\s+(?P<name>\\S.*?)' + '(?:\\s+<cycle\\s(?P<cycle>\\d+)>)?' + '\\s+\\[(?P<index>\\d+)\\]' + '\\s*$')

    def parse_function_entry(self, lines):
        if False:
            print('Hello World!')
        parents = []
        children = []
        while True:
            if not lines:
                sys.stderr.write('warning: unexpected end of entry\n')
                return
            line = lines.pop(0)
            if line.startswith('['):
                break
            mo = self._cg_parent_re.match(line)
            if not mo:
                sys.stderr.write('warning: unrecognized call graph entry (1): %r\n' % line)
            else:
                parent = self.translate(mo)
                if parent.name != '<spontaneous>':
                    parents.append(parent)
        mo = self._cg_primary_re.match(line)
        if not mo:
            sys.stderr.write('warning: unrecognized call graph entry (2): %r\n' % line)
            return
        else:
            function = self.translate(mo)
        while lines:
            line = lines.pop(0)
            mo = self._cg_child_re.match(line)
            if not mo:
                sys.stderr.write('warning: unrecognized call graph entry (3): %r\n' % line)
            else:
                child = self.translate(mo)
                if child.name != '<spontaneous>':
                    children.append(child)
        if function.name != '<spontaneous>':
            function.parents = parents
            function.children = children
            self.functions[function.index] = function

    def parse_cycle_entry(self, lines):
        if False:
            print('Hello World!')
        parents = []
        while True:
            if not lines:
                sys.stderr.write('warning: unexpected end of cycle entry\n')
                return
            line = lines.pop(0)
            if line.startswith('['):
                break
            mo = self._cg_parent_re.match(line)
            if not mo:
                sys.stderr.write('warning: unrecognized call graph entry (6): %r\n' % line)
            else:
                parent = self.translate(mo)
                if parent.name != '<spontaneous>':
                    parents.append(parent)
        mo = self._cg_cycle_header_re.match(line)
        if not mo:
            sys.stderr.write('warning: unrecognized call graph entry (4): %r\n' % line)
            return
        cycle = self.translate(mo)
        cycle.functions = []
        for line in lines[1:]:
            mo = self._cg_cycle_member_re.match(line)
            if not mo:
                sys.stderr.write('warning: unrecognized call graph entry (5): %r\n' % line)
                continue
            call = self.translate(mo)
            cycle.functions.append(call)
        cycle.parents = parents
        self.cycles[cycle.cycle] = cycle

    def parse_cg_entry(self, lines):
        if False:
            return 10
        if any(('as a whole' in linelooper for linelooper in lines)):
            self.parse_cycle_entry(lines)
        else:
            self.parse_function_entry(lines)

    def parse_cg(self):
        if False:
            print('Hello World!')
        'Parse the call graph.'
        line = self.readline()
        while self._cg_header_re.match(line):
            line = self.readline()
        entry_lines = []
        while not self._cg_footer_re.match(line):
            if line.isspace():
                self.parse_cg_entry(entry_lines)
                entry_lines = []
            else:
                entry_lines.append(line)
            line = self.readline()

    def parse(self):
        if False:
            print('Hello World!')
        sys.stderr.write('warning: for axe format, edge weights are unreliable estimates derived from function total times.\n')
        self.parse_cg()
        self.fp.close()
        profile = Profile()
        profile[TIME] = 0.0
        cycles = {}
        for index in self.cycles:
            cycles[index] = Cycle()
        for entry in self.functions.values():
            function = Function(entry.index, entry.name)
            function[TIME] = entry.self
            function[TOTAL_TIME_RATIO] = entry.percentage_time / 100.0
            for child in entry.children:
                call = Call(child.index)
                call[TOTAL_TIME_RATIO] = function[TOTAL_TIME_RATIO]
                if child.index not in self.functions:
                    missing = Function(child.index, child.name)
                    function[TIME] = 0.0
                    profile.add_function(missing)
                function.add_call(call)
            profile.add_function(function)
            if entry.cycle is not None:
                try:
                    cycle = cycles[entry.cycle]
                except KeyError:
                    sys.stderr.write('warning: <cycle %u as a whole> entry missing\n' % entry.cycle)
                    cycle = Cycle()
                    cycles[entry.cycle] = cycle
                cycle.add_function(function)
            profile[TIME] = profile[TIME] + function[TIME]
        for cycle in cycles.values():
            profile.add_cycle(cycle)
        profile.validate()
        profile.ratio(TIME_RATIO, TIME)
        profile.call_ratios(TOTAL_TIME_RATIO)
        for function in profile.functions.values():
            for call in function.calls.values():
                if call.ratio is not None:
                    callee = profile.functions[call.callee_id]
                    call[TOTAL_TIME_RATIO] = call.ratio * callee[TOTAL_TIME_RATIO]
        return profile

class CallgrindParser(LineParser):
    """Parser for valgrind's callgrind tool.

    See also:
    - https://valgrind.org/docs/manual/cl-format.html
    """
    _call_re = re.compile('^calls=\\s*(\\d+)\\s+((\\d+|\\+\\d+|-\\d+|\\*)\\s+)+$')

    def __init__(self, infile):
        if False:
            for i in range(10):
                print('nop')
        LineParser.__init__(self, infile)
        self.position_ids = {}
        self.positions = {}
        self.num_positions = 1
        self.cost_positions = ['line']
        self.last_positions = [0]
        self.num_events = 0
        self.cost_events = []
        self.profile = Profile()
        self.profile[SAMPLES] = 0

    def parse(self):
        if False:
            return 10
        self.readline()
        self.parse_key('version')
        self.parse_key('creator')
        while self.parse_part():
            pass
        if not self.eof():
            sys.stderr.write('warning: line %u: unexpected line\n' % self.line_no)
            sys.stderr.write('%s\n' % self.lookahead())
        self.profile.validate()
        self.profile.find_cycles()
        self.profile.ratio(TIME_RATIO, SAMPLES)
        self.profile.call_ratios(SAMPLES2)
        self.profile.integrate(TOTAL_TIME_RATIO, TIME_RATIO)
        return self.profile

    def parse_part(self):
        if False:
            print('Hello World!')
        if not self.parse_header_line():
            return False
        while self.parse_header_line():
            pass
        if not self.parse_body_line():
            return False
        while self.parse_body_line():
            pass
        return True

    def parse_header_line(self):
        if False:
            print('Hello World!')
        return self.parse_empty() or self.parse_comment() or self.parse_part_detail() or self.parse_description() or self.parse_event_specification() or self.parse_cost_line_def() or self.parse_cost_summary()
    _detail_keys = set(('cmd', 'pid', 'thread', 'part'))

    def parse_part_detail(self):
        if False:
            i = 10
            return i + 15
        return self.parse_keys(self._detail_keys)

    def parse_description(self):
        if False:
            i = 10
            return i + 15
        return self.parse_key('desc') is not None

    def parse_event_specification(self):
        if False:
            while True:
                i = 10
        event = self.parse_key('event')
        if event is None:
            return False
        return True

    def parse_cost_line_def(self):
        if False:
            while True:
                i = 10
        pair = self.parse_keys(('events', 'positions'))
        if pair is None:
            return False
        (key, value) = pair
        items = value.split()
        if key == 'events':
            self.num_events = len(items)
            self.cost_events = items
        if key == 'positions':
            self.num_positions = len(items)
            self.cost_positions = items
            self.last_positions = [0] * self.num_positions
        return True

    def parse_cost_summary(self):
        if False:
            print('Hello World!')
        pair = self.parse_keys(('summary', 'totals'))
        if pair is None:
            return False
        return True

    def parse_body_line(self):
        if False:
            print('Hello World!')
        return self.parse_empty() or self.parse_comment() or self.parse_cost_line() or self.parse_position_spec() or self.parse_association_spec()
    __subpos_re = '(0x[0-9a-fA-F]+|\\d+|\\+\\d+|-\\d+|\\*)'
    _cost_re = re.compile('^' + __subpos_re + '( +' + __subpos_re + ')*' + '( +\\d+)*' + '$')

    def parse_cost_line(self, calls=None):
        if False:
            i = 10
            return i + 15
        line = self.lookahead().rstrip()
        mo = self._cost_re.match(line)
        if not mo:
            return False
        function = self.get_function()
        if calls is None:
            try:
                self.positions['cob'] = self.positions['ob']
            except KeyError:
                pass
        values = line.split()
        assert len(values) <= self.num_positions + self.num_events
        positions = values[0:self.num_positions]
        events = values[self.num_positions:]
        events += ['0'] * (self.num_events - len(events))
        for i in range(self.num_positions):
            position = positions[i]
            if position == '*':
                position = self.last_positions[i]
            elif position[0] in '-+':
                position = self.last_positions[i] + int(position)
            elif position.startswith('0x'):
                position = int(position, 16)
            else:
                position = int(position)
            self.last_positions[i] = position
        events = [float(event) for event in events]
        if calls is None:
            function[SAMPLES] += events[0]
            self.profile[SAMPLES] += events[0]
        else:
            callee = self.get_callee()
            callee.called += calls
            try:
                call = function.calls[callee.id]
            except KeyError:
                call = Call(callee.id)
                call[CALLS] = calls
                call[SAMPLES2] = events[0]
                function.add_call(call)
            else:
                call[CALLS] += calls
                call[SAMPLES2] += events[0]
        self.consume()
        return True

    def parse_association_spec(self):
        if False:
            return 10
        line = self.lookahead()
        if not line.startswith('calls='):
            return False
        (_, values) = line.split('=', 1)
        values = values.strip().split()
        calls = int(values[0])
        call_position = values[1:]
        self.consume()
        self.parse_cost_line(calls)
        return True
    _position_re = re.compile('^(?P<position>[cj]?(?:ob|fl|fi|fe|fn))=\\s*(?:\\((?P<id>\\d+)\\))?(?:\\s*(?P<name>.+))?')
    _position_table_map = {'ob': 'ob', 'fl': 'fl', 'fi': 'fl', 'fe': 'fl', 'fn': 'fn', 'cob': 'ob', 'cfl': 'fl', 'cfi': 'fl', 'cfe': 'fl', 'cfn': 'fn', 'jfi': 'fl'}
    _position_map = {'ob': 'ob', 'fl': 'fl', 'fi': 'fl', 'fe': 'fl', 'fn': 'fn', 'cob': 'cob', 'cfl': 'cfl', 'cfi': 'cfl', 'cfe': 'cfl', 'cfn': 'cfn', 'jfi': 'jfi'}

    def parse_position_spec(self):
        if False:
            print('Hello World!')
        line = self.lookahead()
        if line.startswith('jump=') or line.startswith('jcnd='):
            self.consume()
            return True
        mo = self._position_re.match(line)
        if not mo:
            return False
        (position, id, name) = mo.groups()
        if id:
            table = self._position_table_map[position]
            if name:
                self.position_ids[table, id] = name
            else:
                name = self.position_ids.get((table, id), '')
        self.positions[self._position_map[position]] = name
        self.consume()
        return True

    def parse_empty(self):
        if False:
            return 10
        if self.eof():
            return False
        line = self.lookahead()
        if line.strip():
            return False
        self.consume()
        return True

    def parse_comment(self):
        if False:
            while True:
                i = 10
        line = self.lookahead()
        if not line.startswith('#'):
            return False
        self.consume()
        return True
    _key_re = re.compile('^(\\w+):')

    def parse_key(self, key):
        if False:
            for i in range(10):
                print('nop')
        pair = self.parse_keys((key,))
        if not pair:
            return None
        (key, value) = pair
        return value

    def parse_keys(self, keys):
        if False:
            while True:
                i = 10
        line = self.lookahead()
        mo = self._key_re.match(line)
        if not mo:
            return None
        (key, value) = line.split(':', 1)
        if key not in keys:
            return None
        value = value.strip()
        self.consume()
        return (key, value)

    def make_function(self, module, filename, name):
        if False:
            return 10
        id = name
        try:
            function = self.profile.functions[id]
        except KeyError:
            function = Function(id, name)
            if module:
                function.module = os.path.basename(module)
            function[SAMPLES] = 0
            function.called = 0
            self.profile.add_function(function)
        return function

    def get_function(self):
        if False:
            print('Hello World!')
        module = self.positions.get('ob', '')
        filename = self.positions.get('fl', '')
        function = self.positions.get('fn', '')
        return self.make_function(module, filename, function)

    def get_callee(self):
        if False:
            print('Hello World!')
        module = self.positions.get('cob', '')
        filename = self.positions.get('cfi', '')
        function = self.positions.get('cfn', '')
        return self.make_function(module, filename, function)

    def readline(self):
        if False:
            for i in range(10):
                print('nop')
        while True:
            LineParser.readline(self)
            if self.eof() or not self.lookahead().startswith('#'):
                break

class PerfParser(LineParser):
    """Parser for linux perf callgraph output.

    It expects output generated with

        perf record -g
        perf script | gprof2dot.py --format=perf
    """

    def __init__(self, infile):
        if False:
            return 10
        LineParser.__init__(self, infile)
        self.profile = Profile()

    def readline(self):
        if False:
            i = 10
            return i + 15
        while True:
            LineParser.readline(self)
            if self.eof() or not self.lookahead().startswith('#'):
                break

    def parse(self):
        if False:
            for i in range(10):
                print('nop')
        self.readline()
        profile = self.profile
        profile[SAMPLES] = 0
        while not self.eof():
            self.parse_event()
        profile.validate()
        profile.find_cycles()
        profile.ratio(TIME_RATIO, SAMPLES)
        profile.call_ratios(SAMPLES2)
        if totalMethod == 'callratios':
            profile.integrate(TOTAL_TIME_RATIO, TIME_RATIO)
        elif totalMethod == 'callstacks':
            profile[TOTAL_SAMPLES] = profile[SAMPLES]
            profile.ratio(TOTAL_TIME_RATIO, TOTAL_SAMPLES)
            for function in profile.functions.values():
                for call in function.calls.values():
                    if call.ratio is not None:
                        callee = profile.functions[call.callee_id]
                        call[TOTAL_TIME_RATIO] = call.ratio * callee[TOTAL_TIME_RATIO]
        else:
            assert False
        return profile

    def parse_event(self):
        if False:
            print('Hello World!')
        if self.eof():
            return
        line = self.consume()
        assert line
        callchain = self.parse_callchain()
        if not callchain:
            return
        callee = callchain[0]
        callee[SAMPLES] += 1
        self.profile[SAMPLES] += 1
        for caller in callchain[1:]:
            try:
                call = caller.calls[callee.id]
            except KeyError:
                call = Call(callee.id)
                call[SAMPLES2] = 1
                caller.add_call(call)
            else:
                call[SAMPLES2] += 1
            callee = caller
        stack = set(callchain)
        for function in stack:
            function[TOTAL_SAMPLES] += 1

    def parse_callchain(self):
        if False:
            while True:
                i = 10
        callchain = []
        while self.lookahead():
            function = self.parse_call()
            if function is None:
                break
            callchain.append(function)
        if self.lookahead() == '':
            self.consume()
        return callchain
    call_re = re.compile('^\\s+(?P<address>[0-9a-fA-F]+)\\s+(?P<symbol>.*)\\s+\\((?P<module>.*)\\)$')
    addr2_re = re.compile('\\+0x[0-9a-fA-F]+$')

    def parse_call(self):
        if False:
            i = 10
            return i + 15
        line = self.consume()
        mo = self.call_re.match(line)
        assert mo
        if not mo:
            return None
        function_name = mo.group('symbol')
        if function_name:
            function_name = re.sub(self.addr2_re, '', function_name)
        if not function_name or function_name == '[unknown]':
            function_name = mo.group('address')
        module = mo.group('module')
        function_id = function_name + ':' + module
        try:
            function = self.profile.functions[function_id]
        except KeyError:
            function = Function(function_id, function_name)
            function.module = os.path.basename(module)
            function[SAMPLES] = 0
            function[TOTAL_SAMPLES] = 0
            self.profile.add_function(function)
        return function

class OprofileParser(LineParser):
    """Parser for oprofile callgraph output.

    See also:
    - http://oprofile.sourceforge.net/doc/opreport.html#opreport-callgraph
    """
    _fields_re = {'samples': '(\\d+)', '%': '(\\S+)', 'linenr info': '(?P<source>\\(no location information\\)|\\S+:\\d+)', 'image name': '(?P<image>\\S+(?:\\s\\(tgid:[^)]*\\))?)', 'app name': '(?P<application>\\S+)', 'symbol name': '(?P<symbol>\\(no symbols\\)|.+?)'}

    def __init__(self, infile):
        if False:
            i = 10
            return i + 15
        LineParser.__init__(self, infile)
        self.entries = {}
        self.entry_re = None

    def add_entry(self, callers, function, callees):
        if False:
            for i in range(10):
                print('nop')
        try:
            entry = self.entries[function.id]
        except KeyError:
            self.entries[function.id] = (callers, function, callees)
        else:
            (callers_total, function_total, callees_total) = entry
            self.update_subentries_dict(callers_total, callers)
            function_total.samples += function.samples
            self.update_subentries_dict(callees_total, callees)

    def update_subentries_dict(self, totals, partials):
        if False:
            i = 10
            return i + 15
        for partial in partials.values():
            try:
                total = totals[partial.id]
            except KeyError:
                totals[partial.id] = partial
            else:
                total.samples += partial.samples

    def parse(self):
        if False:
            print('Hello World!')
        self.readline()
        self.parse_header()
        while self.lookahead():
            self.parse_entry()
        profile = Profile()
        reverse_call_samples = {}
        profile[SAMPLES] = 0
        for (_callers, _function, _callees) in self.entries.values():
            function = Function(_function.id, _function.name)
            function[SAMPLES] = _function.samples
            profile.add_function(function)
            profile[SAMPLES] += _function.samples
            if _function.application:
                function.process = os.path.basename(_function.application)
            if _function.image:
                function.module = os.path.basename(_function.image)
            total_callee_samples = 0
            for _callee in _callees.values():
                total_callee_samples += _callee.samples
            for _callee in _callees.values():
                if not _callee.self:
                    call = Call(_callee.id)
                    call[SAMPLES2] = _callee.samples
                    function.add_call(call)
        profile.validate()
        profile.find_cycles()
        profile.ratio(TIME_RATIO, SAMPLES)
        profile.call_ratios(SAMPLES2)
        profile.integrate(TOTAL_TIME_RATIO, TIME_RATIO)
        return profile

    def parse_header(self):
        if False:
            return 10
        while not self.match_header():
            self.consume()
        line = self.lookahead()
        fields = re.split('\\s\\s+', line)
        entry_re = '^\\s*' + '\\s+'.join([self._fields_re[field] for field in fields]) + '(?P<self>\\s+\\[self\\])?$'
        self.entry_re = re.compile(entry_re)
        self.skip_separator()

    def parse_entry(self):
        if False:
            i = 10
            return i + 15
        callers = self.parse_subentries()
        if self.match_primary():
            function = self.parse_subentry()
            if function is not None:
                callees = self.parse_subentries()
                self.add_entry(callers, function, callees)
        self.skip_separator()

    def parse_subentries(self):
        if False:
            i = 10
            return i + 15
        subentries = {}
        while self.match_secondary():
            subentry = self.parse_subentry()
            subentries[subentry.id] = subentry
        return subentries

    def parse_subentry(self):
        if False:
            for i in range(10):
                print('nop')
        entry = Struct()
        line = self.consume()
        mo = self.entry_re.match(line)
        if not mo:
            raise ParseError('failed to parse', line)
        fields = mo.groupdict()
        entry.samples = int(mo.group(1))
        if 'source' in fields and fields['source'] != '(no location information)':
            source = fields['source']
            (filename, lineno) = source.split(':')
            entry.filename = filename
            entry.lineno = int(lineno)
        else:
            source = ''
            entry.filename = None
            entry.lineno = None
        entry.image = fields.get('image', '')
        entry.application = fields.get('application', '')
        if 'symbol' in fields and fields['symbol'] != '(no symbols)':
            entry.symbol = fields['symbol']
        else:
            entry.symbol = ''
        if entry.symbol.startswith('"') and entry.symbol.endswith('"'):
            entry.symbol = entry.symbol[1:-1]
        entry.id = ':'.join((entry.application, entry.image, source, entry.symbol))
        entry.self = fields.get('self', None) != None
        if entry.self:
            entry.id += ':self'
        if entry.symbol:
            entry.name = entry.symbol
        else:
            entry.name = entry.image
        return entry

    def skip_separator(self):
        if False:
            return 10
        while not self.match_separator():
            self.consume()
        self.consume()

    def match_header(self):
        if False:
            i = 10
            return i + 15
        line = self.lookahead()
        return line.startswith('samples')

    def match_separator(self):
        if False:
            for i in range(10):
                print('nop')
        line = self.lookahead()
        return line == '-' * len(line)

    def match_primary(self):
        if False:
            i = 10
            return i + 15
        line = self.lookahead()
        return not line[:1].isspace()

    def match_secondary(self):
        if False:
            return 10
        line = self.lookahead()
        return line[:1].isspace()

class HProfParser(LineParser):
    """Parser for java hprof output

    See also:
    - http://java.sun.com/developer/technicalArticles/Programming/HPROF.html
    """
    trace_re = re.compile('\\t(.*)\\((.*):(.*)\\)')
    trace_id_re = re.compile('^TRACE (\\d+):$')

    def __init__(self, infile):
        if False:
            i = 10
            return i + 15
        LineParser.__init__(self, infile)
        self.traces = {}
        self.samples = {}

    def parse(self):
        if False:
            i = 10
            return i + 15
        self.readline()
        while not self.lookahead().startswith('------'):
            self.consume()
        while not self.lookahead().startswith('TRACE '):
            self.consume()
        self.parse_traces()
        while not self.lookahead().startswith('CPU'):
            self.consume()
        self.parse_samples()
        profile = Profile()
        profile[SAMPLES] = 0
        functions = {}
        for (id, trace) in self.traces.items():
            if not id in self.samples:
                continue
            mtime = self.samples[id][0]
            last = None
            for (func, file, line) in trace:
                if not func in functions:
                    function = Function(func, func)
                    function[SAMPLES] = 0
                    profile.add_function(function)
                    functions[func] = function
                function = functions[func]
                if not last:
                    function[SAMPLES] += mtime
                    profile[SAMPLES] += mtime
                else:
                    c = function.get_call(last)
                    c[SAMPLES2] += mtime
                last = func
        profile.validate()
        profile.find_cycles()
        profile.ratio(TIME_RATIO, SAMPLES)
        profile.call_ratios(SAMPLES2)
        profile.integrate(TOTAL_TIME_RATIO, TIME_RATIO)
        return profile

    def parse_traces(self):
        if False:
            return 10
        while self.lookahead().startswith('TRACE '):
            self.parse_trace()

    def parse_trace(self):
        if False:
            return 10
        l = self.consume()
        mo = self.trace_id_re.match(l)
        tid = mo.group(1)
        last = None
        trace = []
        while self.lookahead().startswith('\t'):
            l = self.consume()
            match = self.trace_re.search(l)
            if not match:
                break
            else:
                (function_name, file, line) = match.groups()
                trace += [(function_name, file, line)]
        self.traces[int(tid)] = trace

    def parse_samples(self):
        if False:
            for i in range(10):
                print('nop')
        self.consume()
        self.consume()
        while not self.lookahead().startswith('CPU'):
            (rank, percent_self, percent_accum, count, traceid, method) = self.lookahead().split()
            self.samples[int(traceid)] = (int(count), method)
            self.consume()

class SysprofParser(XmlParser):

    def __init__(self, stream):
        if False:
            i = 10
            return i + 15
        XmlParser.__init__(self, stream)

    def parse(self):
        if False:
            for i in range(10):
                print('nop')
        objects = {}
        nodes = {}
        self.element_start('profile')
        while self.token.type == XML_ELEMENT_START:
            if self.token.name_or_data == 'objects':
                assert not objects
                objects = self.parse_items('objects')
            elif self.token.name_or_data == 'nodes':
                assert not nodes
                nodes = self.parse_items('nodes')
            else:
                self.parse_value(self.token.name_or_data)
        self.element_end('profile')
        return self.build_profile(objects, nodes)

    def parse_items(self, name):
        if False:
            i = 10
            return i + 15
        assert name[-1] == 's'
        items = {}
        self.element_start(name)
        while self.token.type == XML_ELEMENT_START:
            (id, values) = self.parse_item(name[:-1])
            assert id not in items
            items[id] = values
        self.element_end(name)
        return items

    def parse_item(self, name):
        if False:
            while True:
                i = 10
        attrs = self.element_start(name)
        id = int(attrs['id'])
        values = self.parse_values()
        self.element_end(name)
        return (id, values)

    def parse_values(self):
        if False:
            return 10
        values = {}
        while self.token.type == XML_ELEMENT_START:
            name = self.token.name_or_data
            value = self.parse_value(name)
            assert name not in values
            values[name] = value
        return values

    def parse_value(self, tag):
        if False:
            print('Hello World!')
        self.element_start(tag)
        value = self.character_data()
        self.element_end(tag)
        if value.isdigit():
            return int(value)
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        return value

    def build_profile(self, objects, nodes):
        if False:
            i = 10
            return i + 15
        profile = Profile()
        profile[SAMPLES] = 0
        for (id, object) in objects.items():
            if object['self'] == 0:
                continue
            function = Function(id, object['name'])
            function[SAMPLES] = object['self']
            profile.add_function(function)
            profile[SAMPLES] += function[SAMPLES]
        for (id, node) in nodes.items():
            if node['self'] == 0:
                continue
            parent_id = node['parent']
            while parent_id != 0:
                parent = nodes[parent_id]
                caller_id = parent['object']
                if objects[caller_id]['self'] != 0:
                    break
                parent_id = parent['parent']
            if parent_id == 0:
                continue
            callee_id = node['object']
            assert objects[caller_id]['self']
            assert objects[callee_id]['self']
            function = profile.functions[caller_id]
            samples = node['self']
            try:
                call = function.calls[callee_id]
            except KeyError:
                call = Call(callee_id)
                call[SAMPLES2] = samples
                function.add_call(call)
            else:
                call[SAMPLES2] += samples
        profile.validate()
        profile.find_cycles()
        profile.ratio(TIME_RATIO, SAMPLES)
        profile.call_ratios(SAMPLES2)
        profile.integrate(TOTAL_TIME_RATIO, TIME_RATIO)
        return profile

class XPerfParser(Parser):
    """Parser for CSVs generated by XPerf, from Microsoft Windows Performance Tools.
    """

    def __init__(self, stream):
        if False:
            return 10
        Parser.__init__(self)
        self.stream = stream
        self.profile = Profile()
        self.profile[SAMPLES] = 0
        self.column = {}

    def parse(self):
        if False:
            print('Hello World!')
        import csv
        reader = csv.reader(self.stream, delimiter=',', quotechar=None, escapechar=None, doublequote=False, skipinitialspace=True, lineterminator='\r\n', quoting=csv.QUOTE_NONE)
        header = True
        for row in reader:
            if header:
                self.parse_header(row)
                header = False
            else:
                self.parse_row(row)
        self.profile.validate()
        self.profile.find_cycles()
        self.profile.ratio(TIME_RATIO, SAMPLES)
        self.profile.call_ratios(SAMPLES2)
        self.profile.integrate(TOTAL_TIME_RATIO, TIME_RATIO)
        return self.profile

    def parse_header(self, row):
        if False:
            return 10
        for column in range(len(row)):
            name = row[column]
            assert name not in self.column
            self.column[name] = column

    def parse_row(self, row):
        if False:
            while True:
                i = 10
        fields = {}
        for (name, column) in self.column.items():
            value = row[column]
            for factory in (int, float):
                try:
                    value = factory(value)
                except ValueError:
                    pass
                else:
                    break
            fields[name] = value
        process = fields['Process Name']
        symbol = fields['Module'] + '!' + fields['Function']
        weight = fields['Weight']
        count = fields['Count']
        if process == 'Idle':
            return
        function = self.get_function(process, symbol)
        function[SAMPLES] += weight * count
        self.profile[SAMPLES] += weight * count
        stack = fields['Stack']
        if stack != '?':
            stack = stack.split('/')
            assert stack[0] == '[Root]'
            if stack[-1] != symbol:
                stack.append(symbol)
            caller = None
            for symbol in stack[1:]:
                callee = self.get_function(process, symbol)
                if caller is not None:
                    try:
                        call = caller.calls[callee.id]
                    except KeyError:
                        call = Call(callee.id)
                        call[SAMPLES2] = count
                        caller.add_call(call)
                    else:
                        call[SAMPLES2] += count
                caller = callee

    def get_function(self, process, symbol):
        if False:
            for i in range(10):
                print('nop')
        function_id = process + '!' + symbol
        try:
            function = self.profile.functions[function_id]
        except KeyError:
            (module, name) = symbol.split('!', 1)
            function = Function(function_id, name)
            function.process = process
            function.module = module
            function[SAMPLES] = 0
            self.profile.add_function(function)
        return function

class SleepyParser(Parser):
    """Parser for GNU gprof output.

    See also:
    - http://www.codersnotes.com/sleepy/
    - http://sleepygraph.sourceforge.net/
    """
    stdinInput = False

    def __init__(self, filename):
        if False:
            return 10
        Parser.__init__(self)
        from zipfile import ZipFile
        self.database = ZipFile(filename)
        self.symbols = {}
        self.calls = {}
        self.profile = Profile()
    _symbol_re = re.compile('^(?P<id>\\w+)' + '\\s+"(?P<module>[^"]*)"' + '\\s+"(?P<procname>[^"]*)"' + '\\s+"(?P<sourcefile>[^"]*)"' + '\\s+(?P<sourceline>\\d+)$')

    def openEntry(self, name):
        if False:
            return 10
        for database_name in self.database.namelist():
            if name.lower() == database_name.lower():
                name = database_name
                break
        return self.database.open(name, 'r')

    def parse_symbols(self):
        if False:
            while True:
                i = 10
        for line in self.openEntry('Symbols.txt'):
            line = line.decode('UTF-8').rstrip('\r\n')
            mo = self._symbol_re.match(line)
            if mo:
                (symbol_id, module, procname, sourcefile, sourceline) = mo.groups()
                function_id = ':'.join([module, procname])
                try:
                    function = self.profile.functions[function_id]
                except KeyError:
                    function = Function(function_id, procname)
                    function.module = module
                    function[SAMPLES] = 0
                    self.profile.add_function(function)
                self.symbols[symbol_id] = function

    def parse_callstacks(self):
        if False:
            while True:
                i = 10
        for line in self.openEntry('Callstacks.txt'):
            line = line.decode('UTF-8').rstrip('\r\n')
            fields = line.split()
            samples = float(fields[0])
            callstack = fields[1:]
            callstack = [self.symbols[symbol_id] for symbol_id in callstack]
            callee = callstack[0]
            callee[SAMPLES] += samples
            self.profile[SAMPLES] += samples
            for caller in callstack[1:]:
                try:
                    call = caller.calls[callee.id]
                except KeyError:
                    call = Call(callee.id)
                    call[SAMPLES2] = samples
                    caller.add_call(call)
                else:
                    call[SAMPLES2] += samples
                callee = caller

    def parse(self):
        if False:
            print('Hello World!')
        profile = self.profile
        profile[SAMPLES] = 0
        self.parse_symbols()
        self.parse_callstacks()
        profile.validate()
        profile.find_cycles()
        profile.ratio(TIME_RATIO, SAMPLES)
        profile.call_ratios(SAMPLES2)
        profile.integrate(TOTAL_TIME_RATIO, TIME_RATIO)
        return profile

class PstatsParser:
    """Parser python profiling statistics saved with te pstats module."""
    stdinInput = False
    multipleInput = True

    def __init__(self, *filename):
        if False:
            while True:
                i = 10
        import pstats
        try:
            self.stats = pstats.Stats(*filename)
        except ValueError:
            sys.stderr.write('error: failed to load %s, maybe they are generated by different python version?\n' % ', '.join(filename))
            sys.exit(1)
        self.profile = Profile()
        self.function_ids = {}

    def get_function_name(self, key):
        if False:
            for i in range(10):
                print('nop')
        (filename, line, name) = key
        module = os.path.splitext(filename)[0]
        module = os.path.basename(module)
        return '%s:%d:%s' % (module, line, name)

    def get_function(self, key):
        if False:
            for i in range(10):
                print('nop')
        try:
            id = self.function_ids[key]
        except KeyError:
            id = len(self.function_ids)
            name = self.get_function_name(key)
            function = Function(id, name)
            function.filename = key[0]
            self.profile.functions[id] = function
            self.function_ids[key] = id
        else:
            function = self.profile.functions[id]
        return function

    def parse(self):
        if False:
            return 10
        self.profile[TIME] = 0.0
        self.profile[TOTAL_TIME] = self.stats.total_tt
        for (fn, (cc, nc, tt, ct, callers)) in self.stats.stats.items():
            callee = self.get_function(fn)
            callee.called = nc
            callee[TOTAL_TIME] = ct
            callee[TIME] = tt
            self.profile[TIME] += tt
            self.profile[TOTAL_TIME] = max(self.profile[TOTAL_TIME], ct)
            for (fn, value) in callers.items():
                caller = self.get_function(fn)
                call = Call(callee.id)
                if isinstance(value, tuple):
                    for i in range(0, len(value), 4):
                        (nc, cc, tt, ct) = value[i:i + 4]
                        if CALLS in call:
                            call[CALLS] += cc
                        else:
                            call[CALLS] = cc
                        if TOTAL_TIME in call:
                            call[TOTAL_TIME] += ct
                        else:
                            call[TOTAL_TIME] = ct
                else:
                    call[CALLS] = value
                    call[TOTAL_TIME] = ratio(value, nc) * ct
                caller.add_call(call)
        if False:
            self.stats.print_stats()
            self.stats.print_callees()
        self.profile.validate()
        self.profile.ratio(TIME_RATIO, TIME)
        self.profile.ratio(TOTAL_TIME_RATIO, TOTAL_TIME)
        return self.profile

class DtraceParser(LineParser):
    """Parser for linux perf callgraph output.

    It expects output generated with

        # Refer to https://github.com/brendangregg/FlameGraph#dtrace
        # 60 seconds of user-level stacks, including time spent in-kernel, for PID 12345 at 97 Hertz
        sudo dtrace -x ustackframes=100 -n 'profile-97 /pid == 12345/ { @[ustack()] = count(); } tick-60s { exit(0); }' -o out.user_stacks

        # The dtrace output
        gprof2dot.py -f dtrace out.user_stacks

        # Notice: sometimes, the dtrace outputs format may be latin-1, and gprof2dot will fail to parse it.
        # To solve this problem, you should use iconv to convert to UTF-8 explicitly.
        # TODO: add an encoding flag to tell gprof2dot how to decode the profile file.
        iconv -f ISO-8859-1 -t UTF-8 out.user_stacks | gprof2dot.py -f dtrace
    """

    def __init__(self, infile):
        if False:
            while True:
                i = 10
        LineParser.__init__(self, infile)
        self.profile = Profile()

    def readline(self):
        if False:
            return 10
        while True:
            LineParser.readline(self)
            if self.eof():
                break
            line = self.lookahead().strip()
            if line.startswith('CPU'):
                LineParser.readline(self)
            elif not line == '':
                break

    def parse(self):
        if False:
            print('Hello World!')
        self.readline()
        profile = self.profile
        profile[SAMPLES] = 0
        while not self.eof():
            self.parse_event()
        profile.validate()
        profile.find_cycles()
        profile.ratio(TIME_RATIO, SAMPLES)
        profile.call_ratios(SAMPLES2)
        if totalMethod == 'callratios':
            profile.integrate(TOTAL_TIME_RATIO, TIME_RATIO)
        elif totalMethod == 'callstacks':
            profile[TOTAL_SAMPLES] = profile[SAMPLES]
            profile.ratio(TOTAL_TIME_RATIO, TOTAL_SAMPLES)
            for function in profile.functions.values():
                for call in function.calls.values():
                    if call.ratio is not None:
                        callee = profile.functions[call.callee_id]
                        call[TOTAL_TIME_RATIO] = call.ratio * callee[TOTAL_TIME_RATIO]
        else:
            assert False
        return profile

    def parse_event(self):
        if False:
            print('Hello World!')
        if self.eof():
            return
        (callchain, count) = self.parse_callchain()
        if not callchain:
            return
        callee = callchain[0]
        callee[SAMPLES] += count
        self.profile[SAMPLES] += count
        for caller in callchain[1:]:
            try:
                call = caller.calls[callee.id]
            except KeyError:
                call = Call(callee.id)
                call[SAMPLES2] = count
                caller.add_call(call)
            else:
                call[SAMPLES2] += count
            callee = caller
        stack = set(callchain)
        for function in stack:
            function[TOTAL_SAMPLES] += count

    def parse_callchain(self):
        if False:
            while True:
                i = 10
        callchain = []
        count = 0
        while self.lookahead():
            (function, count) = self.parse_call()
            if function is None:
                break
            callchain.append(function)
        return (callchain, count)
    call_re = re.compile('^\\s+(?P<module>.*)`(?P<symbol>.*)')
    addr2_re = re.compile('\\+0x[0-9a-fA-F]+$')

    def parse_call(self):
        if False:
            for i in range(10):
                print('nop')
        line = self.consume()
        mo = self.call_re.match(line)
        if not mo:
            return (None, int(line.strip()))
        function_name = mo.group('symbol')
        if function_name:
            function_name = re.sub(self.addr2_re, '', function_name)
        module = mo.group('module')
        function_id = function_name + ':' + module
        try:
            function = self.profile.functions[function_id]
        except KeyError:
            function = Function(function_id, function_name)
            function.module = os.path.basename(module)
            function[SAMPLES] = 0
            function[TOTAL_SAMPLES] = 0
            self.profile.add_function(function)
        return (function, None)
formats = {'axe': AXEParser, 'callgrind': CallgrindParser, 'hprof': HProfParser, 'json': JsonParser, 'oprofile': OprofileParser, 'perf': PerfParser, 'prof': GprofParser, 'pstats': PstatsParser, 'sleepy': SleepyParser, 'sysprof': SysprofParser, 'xperf': XPerfParser, 'dtrace': DtraceParser}

class Theme:

    def __init__(self, bgcolor=(0.0, 0.0, 1.0), mincolor=(0.0, 0.0, 0.0), maxcolor=(0.0, 0.0, 1.0), fontname='Arial', fontcolor='white', nodestyle='filled', minfontsize=10.0, maxfontsize=10.0, minpenwidth=0.5, maxpenwidth=4.0, gamma=2.2, skew=1.0):
        if False:
            return 10
        self.bgcolor = bgcolor
        self.mincolor = mincolor
        self.maxcolor = maxcolor
        self.fontname = fontname
        self.fontcolor = fontcolor
        self.nodestyle = nodestyle
        self.minfontsize = minfontsize
        self.maxfontsize = maxfontsize
        self.minpenwidth = minpenwidth
        self.maxpenwidth = maxpenwidth
        self.gamma = gamma
        self.skew = skew

    def graph_bgcolor(self):
        if False:
            for i in range(10):
                print('nop')
        return self.hsl_to_rgb(*self.bgcolor)

    def graph_fontname(self):
        if False:
            i = 10
            return i + 15
        return self.fontname

    def graph_fontcolor(self):
        if False:
            i = 10
            return i + 15
        return self.fontcolor

    def node_bgcolor(self, weight):
        if False:
            for i in range(10):
                print('nop')
        return self.color(weight)

    def node_fgcolor(self, weight):
        if False:
            for i in range(10):
                print('nop')
        if self.nodestyle == 'filled':
            return self.graph_bgcolor()
        else:
            return self.color(weight)

    def node_fontsize(self, weight):
        if False:
            i = 10
            return i + 15
        return self.fontsize(weight)

    def node_style(self):
        if False:
            while True:
                i = 10
        return self.nodestyle

    def edge_color(self, weight):
        if False:
            print('Hello World!')
        return self.color(weight)

    def edge_fontsize(self, weight):
        if False:
            print('Hello World!')
        return self.fontsize(weight)

    def edge_penwidth(self, weight):
        if False:
            while True:
                i = 10
        return max(weight * self.maxpenwidth, self.minpenwidth)

    def edge_arrowsize(self, weight):
        if False:
            i = 10
            return i + 15
        return 0.5 * math.sqrt(self.edge_penwidth(weight))

    def fontsize(self, weight):
        if False:
            while True:
                i = 10
        return max(weight ** 2 * self.maxfontsize, self.minfontsize)

    def color(self, weight):
        if False:
            i = 10
            return i + 15
        weight = min(max(weight, 0.0), 1.0)
        (hmin, smin, lmin) = self.mincolor
        (hmax, smax, lmax) = self.maxcolor
        if self.skew < 0:
            raise ValueError('Skew must be greater than 0')
        elif self.skew == 1.0:
            h = hmin + weight * (hmax - hmin)
            s = smin + weight * (smax - smin)
            l = lmin + weight * (lmax - lmin)
        else:
            base = self.skew
            h = hmin + (hmax - hmin) * (-1.0 + base ** weight) / (base - 1.0)
            s = smin + (smax - smin) * (-1.0 + base ** weight) / (base - 1.0)
            l = lmin + (lmax - lmin) * (-1.0 + base ** weight) / (base - 1.0)
        return self.hsl_to_rgb(h, s, l)

    def hsl_to_rgb(self, h, s, l):
        if False:
            print('Hello World!')
        'Convert a color from HSL color-model to RGB.\n\n        See also:\n        - http://www.w3.org/TR/css3-color/#hsl-color\n        '
        h = h % 1.0
        s = min(max(s, 0.0), 1.0)
        l = min(max(l, 0.0), 1.0)
        if l <= 0.5:
            m2 = l * (s + 1.0)
        else:
            m2 = l + s - l * s
        m1 = l * 2.0 - m2
        r = self._hue_to_rgb(m1, m2, h + 1.0 / 3.0)
        g = self._hue_to_rgb(m1, m2, h)
        b = self._hue_to_rgb(m1, m2, h - 1.0 / 3.0)
        r **= self.gamma
        g **= self.gamma
        b **= self.gamma
        return (r, g, b)

    def _hue_to_rgb(self, m1, m2, h):
        if False:
            i = 10
            return i + 15
        if h < 0.0:
            h += 1.0
        elif h > 1.0:
            h -= 1.0
        if h * 6 < 1.0:
            return m1 + (m2 - m1) * h * 6.0
        elif h * 2 < 1.0:
            return m2
        elif h * 3 < 2.0:
            return m1 + (m2 - m1) * (2.0 / 3.0 - h) * 6.0
        else:
            return m1
TEMPERATURE_COLORMAP = Theme(mincolor=(2.0 / 3.0, 0.8, 0.25), maxcolor=(0.0, 1.0, 0.5), gamma=1.0)
PINK_COLORMAP = Theme(mincolor=(0.0, 1.0, 0.9), maxcolor=(0.0, 1.0, 0.5))
GRAY_COLORMAP = Theme(mincolor=(0.0, 0.0, 0.85), maxcolor=(0.0, 0.0, 0.0))
BW_COLORMAP = Theme(minfontsize=8.0, maxfontsize=24.0, mincolor=(0.0, 0.0, 0.0), maxcolor=(0.0, 0.0, 0.0), minpenwidth=0.1, maxpenwidth=8.0)
PRINT_COLORMAP = Theme(minfontsize=18.0, maxfontsize=30.0, fontcolor='black', nodestyle='solid', mincolor=(0.0, 0.0, 0.0), maxcolor=(0.0, 0.0, 0.0), minpenwidth=0.1, maxpenwidth=8.0)
themes = {'color': TEMPERATURE_COLORMAP, 'pink': PINK_COLORMAP, 'gray': GRAY_COLORMAP, 'bw': BW_COLORMAP, 'print': PRINT_COLORMAP}

def sorted_iteritems(d):
    if False:
        for i in range(10):
            print('nop')
    keys = list(d.keys())
    keys.sort()
    for key in keys:
        value = d[key]
        yield (key, value)

class DotWriter:
    """Writer for the DOT language.

    See also:
    - "The DOT Language" specification
      http://www.graphviz.org/doc/info/lang.html
    """
    strip = False
    wrap = False

    def __init__(self, fp):
        if False:
            for i in range(10):
                print('nop')
        self.fp = fp

    def wrap_function_name(self, name):
        if False:
            return 10
        'Split the function name on multiple lines.'
        if len(name) > 32:
            ratio = 2.0 / 3.0
            height = max(int(len(name) / (1.0 - ratio) + 0.5), 1)
            width = max(len(name) / height, 32)
            name = textwrap.fill(name, width, break_long_words=False)
        name = name.replace(', ', ',')
        name = name.replace('> >', '>>')
        name = name.replace('> >', '>>')
        return name
    show_function_events = [TOTAL_TIME_RATIO, TIME_RATIO]
    show_edge_events = [TOTAL_TIME_RATIO, CALLS]

    def graph(self, profile, theme):
        if False:
            i = 10
            return i + 15
        self.begin_graph()
        fontname = theme.graph_fontname()
        fontcolor = theme.graph_fontcolor()
        nodestyle = theme.node_style()
        self.attr('graph', fontname=fontname, ranksep=0.25, nodesep=0.125)
        self.attr('node', fontname=fontname, shape='box', style=nodestyle, fontcolor=fontcolor, width=0, height=0)
        self.attr('edge', fontname=fontname)
        for (_, function) in sorted_iteritems(profile.functions):
            labels = []
            if function.process is not None:
                labels.append(function.process)
            if function.module is not None:
                labels.append(function.module)
            if self.strip:
                function_name = function.stripped_name()
            else:
                function_name = function.name
            MAX_FUNCTION_NAME = 4096
            if len(function_name) >= MAX_FUNCTION_NAME:
                sys.stderr.write('warning: truncating function name with %u chars (%s)\n' % (len(function_name), function_name[:32] + '...'))
                function_name = function_name[:MAX_FUNCTION_NAME - 1] + chr(8230)
            if self.wrap:
                function_name = self.wrap_function_name(function_name)
            labels.append(function_name)
            for event in self.show_function_events:
                if event in function.events:
                    label = event.format(function[event])
                    labels.append(label)
            if function.called is not None:
                labels.append('%u%s' % (function.called, MULTIPLICATION_SIGN))
            if function.weight is not None:
                weight = function.weight
            else:
                weight = 0.0
            label = '\n'.join(labels)
            self.node(function.id, label=label, color=self.color(theme.node_bgcolor(weight)), fontcolor=self.color(theme.node_fgcolor(weight)), fontsize='%.2f' % theme.node_fontsize(weight), tooltip=function.filename)
            for (_, call) in sorted_iteritems(function.calls):
                callee = profile.functions[call.callee_id]
                labels = []
                for event in self.show_edge_events:
                    if event in call.events:
                        label = event.format(call[event])
                        labels.append(label)
                if call.weight is not None:
                    weight = call.weight
                elif callee.weight is not None:
                    weight = callee.weight
                else:
                    weight = 0.0
                label = '\n'.join(labels)
                self.edge(function.id, call.callee_id, label=label, color=self.color(theme.edge_color(weight)), fontcolor=self.color(theme.edge_color(weight)), fontsize='%.2f' % theme.edge_fontsize(weight), penwidth='%.2f' % theme.edge_penwidth(weight), labeldistance='%.2f' % theme.edge_penwidth(weight), arrowsize='%.2f' % theme.edge_arrowsize(weight))
        self.end_graph()

    def begin_graph(self):
        if False:
            for i in range(10):
                print('nop')
        self.write('digraph {\n')

    def end_graph(self):
        if False:
            while True:
                i = 10
        self.write('}\n')

    def attr(self, what, **attrs):
        if False:
            while True:
                i = 10
        self.write('\t')
        self.write(what)
        self.attr_list(attrs)
        self.write(';\n')

    def node(self, node, **attrs):
        if False:
            for i in range(10):
                print('nop')
        self.write('\t')
        self.id(node)
        self.attr_list(attrs)
        self.write(';\n')

    def edge(self, src, dst, **attrs):
        if False:
            for i in range(10):
                print('nop')
        self.write('\t')
        self.id(src)
        self.write(' -> ')
        self.id(dst)
        self.attr_list(attrs)
        self.write(';\n')

    def attr_list(self, attrs):
        if False:
            print('Hello World!')
        if not attrs:
            return
        self.write(' [')
        first = True
        for (name, value) in sorted_iteritems(attrs):
            if value is None:
                continue
            if first:
                first = False
            else:
                self.write(', ')
            self.id(name)
            self.write('=')
            self.id(value)
        self.write(']')

    def id(self, id):
        if False:
            while True:
                i = 10
        if isinstance(id, (int, float)):
            s = str(id)
        elif isinstance(id, str):
            if id.isalnum() and (not id.startswith('0x')):
                s = id
            else:
                s = self.escape(id)
        else:
            raise TypeError
        self.write(s)

    def color(self, rgb):
        if False:
            return 10
        (r, g, b) = rgb

        def float2int(f):
            if False:
                return 10
            if f <= 0.0:
                return 0
            if f >= 1.0:
                return 255
            return int(255.0 * f + 0.5)
        return '#' + ''.join(['%02x' % float2int(c) for c in (r, g, b)])

    def escape(self, s):
        if False:
            for i in range(10):
                print('nop')
        s = s.replace('\\', '\\\\')
        s = s.replace('\n', '\\n')
        s = s.replace('\t', '\\t')
        s = s.replace('"', '\\"')
        return '"' + s + '"'

    def write(self, s):
        if False:
            print('Hello World!')
        self.fp.write(s)

def naturalJoin(values):
    if False:
        return 10
    if len(values) >= 2:
        return ', '.join(values[:-1]) + ' or ' + values[-1]
    else:
        return ''.join(values)

def main(argv=sys.argv[1:]):
    if False:
        return 10
    'Main program.'
    global totalMethod
    formatNames = list(formats.keys())
    formatNames.sort()
    themeNames = list(themes.keys())
    themeNames.sort()
    labelNames = list(labels.keys())
    labelNames.sort()
    optparser = optparse.OptionParser(usage='\n\t%prog [options] [file] ...')
    optparser.add_option('-o', '--output', metavar='FILE', type='string', dest='output', help='output filename [stdout]')
    optparser.add_option('-n', '--node-thres', metavar='PERCENTAGE', type='float', dest='node_thres', default=0.5, help='eliminate nodes below this threshold [default: %default]')
    optparser.add_option('-e', '--edge-thres', metavar='PERCENTAGE', type='float', dest='edge_thres', default=0.1, help='eliminate edges below this threshold [default: %default]')
    optparser.add_option('-f', '--format', type='choice', choices=formatNames, dest='format', default='prof', help='profile format: %s [default: %%default]' % naturalJoin(formatNames))
    optparser.add_option('--total', type='choice', choices=('callratios', 'callstacks'), dest='totalMethod', default=totalMethod, help='preferred method of calculating total time: callratios or callstacks (currently affects only perf format) [default: %default]')
    optparser.add_option('-c', '--colormap', type='choice', choices=themeNames, dest='theme', default='color', help='color map: %s [default: %%default]' % naturalJoin(themeNames))
    optparser.add_option('-s', '--strip', action='store_true', dest='strip', default=False, help='strip function parameters, template parameters, and const modifiers from demangled C++ function names')
    optparser.add_option('--color-nodes-by-selftime', action='store_true', dest='color_nodes_by_selftime', default=False, help='color nodes by self time, rather than by total time (sum of self and descendants)')
    optparser.add_option('--colour-nodes-by-selftime', action='store_true', dest='color_nodes_by_selftime', help=optparse.SUPPRESS_HELP)
    optparser.add_option('-w', '--wrap', action='store_true', dest='wrap', default=False, help='wrap function names')
    optparser.add_option('--show-samples', action='store_true', dest='show_samples', default=False, help='show function samples')
    optparser.add_option('--node-label', metavar='MEASURE', type='choice', choices=labelNames, action='append', dest='node_labels', help='measurements to on show the node (can be specified multiple times): %s [default: %s]' % (naturalJoin(labelNames), ', '.join(defaultLabelNames)))
    optparser.add_option('--list-functions', type='string', dest='list_functions', default=None, help="list functions available for selection in -z or -l, requires selector argument\n( use '+' to select all).\nRecall that the selector argument is used with Unix/Bash globbing/pattern matching,\nand that entries are formatted '<pkg>:<linenum>:<function>'. When argument starts\nwith '%', a dump of all available information is performed for selected entries,\n after removal of leading '%'.\n")
    optparser.add_option('-z', '--root', type='string', dest='root', default='', help='prune call graph to show only descendants of specified root function')
    optparser.add_option('-l', '--leaf', type='string', dest='leaf', default='', help='prune call graph to show only ancestors of specified leaf function')
    optparser.add_option('--depth', type='int', dest='depth', default=-1, help='prune call graph to show only descendants or ancestors until specified depth')
    optparser.add_option('--skew', type='float', dest='theme_skew', default=1.0, help='skew the colorization curve.  Values < 1.0 give more variety to lower percentages.  Values > 1.0 give less variety to lower percentages')
    optparser.add_option('-p', '--path', action='append', type='string', dest='filter_paths', help='Filter all modules not in a specified path')
    (options, args) = optparser.parse_args(argv)
    if len(args) > 1 and options.format != 'pstats':
        optparser.error('incorrect number of arguments')
    try:
        theme = themes[options.theme]
    except KeyError:
        optparser.error("invalid colormap '%s'" % options.theme)
    if options.theme_skew:
        theme.skew = options.theme_skew
    totalMethod = options.totalMethod
    try:
        Format = formats[options.format]
    except KeyError:
        optparser.error("invalid format '%s'" % options.format)
    if Format.stdinInput:
        if not args:
            fp = sys.stdin
        else:
            fp = open(args[0], 'rt', encoding='UTF-8')
        parser = Format(fp)
    elif Format.multipleInput:
        if not args:
            optparser.error('at least a file must be specified for %s input' % options.format)
        parser = Format(*args)
    else:
        if len(args) != 1:
            optparser.error('exactly one file must be specified for %s input' % options.format)
        parser = Format(args[0])
    profile = parser.parse()
    if options.output is None:
        output = open(sys.stdout.fileno(), mode='wt', encoding='UTF-8', closefd=False)
    else:
        output = open(options.output, 'wt', encoding='UTF-8')
    dot = DotWriter(output)
    dot.strip = options.strip
    dot.wrap = options.wrap
    labelNames = options.node_labels or defaultLabelNames
    dot.show_function_events = [labels[l] for l in labelNames]
    if options.show_samples:
        dot.show_function_events.append(SAMPLES)
    profile.prune(options.node_thres / 100.0, options.edge_thres / 100.0, options.filter_paths, options.color_nodes_by_selftime)
    if options.list_functions:
        profile.printFunctionIds(selector=options.list_functions)
        sys.exit(0)
    if options.root:
        rootIds = profile.getFunctionIds(options.root)
        if not rootIds:
            sys.stderr.write('root node ' + options.root + ' not found (might already be pruned : try -e0 -n0 flags)\n')
            sys.exit(1)
        profile.prune_root(rootIds, options.depth)
    if options.leaf:
        leafIds = profile.getFunctionIds(options.leaf)
        if not leafIds:
            sys.stderr.write('leaf node ' + options.leaf + ' not found (maybe already pruned : try -e0 -n0 flags)\n')
            sys.exit(1)
        profile.prune_leaf(leafIds, options.depth)
    dot.graph(profile, theme)
if __name__ == '__main__':
    main()