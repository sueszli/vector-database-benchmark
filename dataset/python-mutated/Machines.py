"""
Python Lexical Analyser

Classes for building NFAs and DFAs
"""
from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
maxint = 2 ** 31 - 1
if not cython.compiled:
    try:
        unichr
    except NameError:
        unichr = chr
LOWEST_PRIORITY = -maxint

class Machine(object):
    """A collection of Nodes representing an NFA or DFA."""

    def __init__(self):
        if False:
            return 10
        self.states = []
        self.initial_states = {}
        self.next_state_number = 1

    def __del__(self):
        if False:
            i = 10
            return i + 15
        for state in self.states:
            state.destroy()

    def new_state(self):
        if False:
            i = 10
            return i + 15
        'Add a new state to the machine and return it.'
        s = Node()
        n = self.next_state_number
        self.next_state_number = n + 1
        s.number = n
        self.states.append(s)
        return s

    def new_initial_state(self, name):
        if False:
            while True:
                i = 10
        state = self.new_state()
        self.make_initial_state(name, state)
        return state

    def make_initial_state(self, name, state):
        if False:
            for i in range(10):
                print('nop')
        self.initial_states[name] = state

    def get_initial_state(self, name):
        if False:
            while True:
                i = 10
        return self.initial_states[name]

    def dump(self, file):
        if False:
            print('Hello World!')
        file.write('Plex.Machine:\n')
        if self.initial_states is not None:
            file.write('   Initial states:\n')
            for (name, state) in sorted(self.initial_states.items()):
                file.write("      '%s': %d\n" % (name, state.number))
        for s in self.states:
            s.dump(file)

class Node(object):
    """A state of an NFA or DFA."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.transitions = TransitionMap()
        self.action_priority = LOWEST_PRIORITY
        self.action = None
        self.number = 0
        self.epsilon_closure = None

    def destroy(self):
        if False:
            while True:
                i = 10
        self.transitions = None
        self.action = None
        self.epsilon_closure = None

    def add_transition(self, event, new_state):
        if False:
            return 10
        self.transitions.add(event, new_state)

    def link_to(self, state):
        if False:
            i = 10
            return i + 15
        'Add an epsilon-move from this state to another state.'
        self.add_transition('', state)

    def set_action(self, action, priority):
        if False:
            print('Hello World!')
        'Make this an accepting state with the given action. If\n        there is already an action, choose the action with highest\n        priority.'
        if priority > self.action_priority:
            self.action = action
            self.action_priority = priority

    def get_action(self):
        if False:
            for i in range(10):
                print('nop')
        return self.action

    def get_action_priority(self):
        if False:
            return 10
        return self.action_priority

    def is_accepting(self):
        if False:
            for i in range(10):
                print('nop')
        return self.action is not None

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'State %d' % self.number

    def dump(self, file):
        if False:
            while True:
                i = 10
        file.write('   State %d:\n' % self.number)
        self.transitions.dump(file)
        action = self.action
        priority = self.action_priority
        if action is not None:
            file.write('      %s [priority %d]\n' % (action, priority))

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.number < other.number

    def __hash__(self):
        if False:
            print('Hello World!')
        return id(self) & maxint

class FastMachine(object):
    """
    FastMachine is a deterministic machine represented in a way that
    allows fast scanning.
    """

    def __init__(self):
        if False:
            return 10
        self.initial_states = {}
        self.states = []
        self.next_number = 1
        self.new_state_template = {'': None, 'bol': None, 'eol': None, 'eof': None, 'else': None}

    def __del__(self):
        if False:
            print('Hello World!')
        for state in self.states:
            state.clear()

    def new_state(self, action=None):
        if False:
            while True:
                i = 10
        number = self.next_number
        self.next_number = number + 1
        result = self.new_state_template.copy()
        result['number'] = number
        result['action'] = action
        self.states.append(result)
        return result

    def make_initial_state(self, name, state):
        if False:
            i = 10
            return i + 15
        self.initial_states[name] = state

    @cython.locals(code0=cython.int, code1=cython.int, maxint=cython.int, state=dict)
    def add_transitions(self, state, event, new_state, maxint=maxint):
        if False:
            while True:
                i = 10
        if type(event) is tuple:
            (code0, code1) = event
            if code0 == -maxint:
                state['else'] = new_state
            elif code1 != maxint:
                while code0 < code1:
                    state[unichr(code0)] = new_state
                    code0 += 1
        else:
            state[event] = new_state

    def get_initial_state(self, name):
        if False:
            while True:
                i = 10
        return self.initial_states[name]

    def dump(self, file):
        if False:
            while True:
                i = 10
        file.write('Plex.FastMachine:\n')
        file.write('   Initial states:\n')
        for (name, state) in sorted(self.initial_states.items()):
            file.write('      %s: %s\n' % (repr(name), state['number']))
        for state in self.states:
            self.dump_state(state, file)

    def dump_state(self, state, file):
        if False:
            print('Hello World!')
        file.write('   State %d:\n' % state['number'])
        self.dump_transitions(state, file)
        action = state['action']
        if action is not None:
            file.write('      %s\n' % action)

    def dump_transitions(self, state, file):
        if False:
            for i in range(10):
                print('nop')
        chars_leading_to_state = {}
        special_to_state = {}
        for (c, s) in state.items():
            if len(c) == 1:
                chars = chars_leading_to_state.get(id(s), None)
                if chars is None:
                    chars = []
                    chars_leading_to_state[id(s)] = chars
                chars.append(c)
            elif len(c) <= 4:
                special_to_state[c] = s
        ranges_to_state = {}
        for state in self.states:
            char_list = chars_leading_to_state.get(id(state), None)
            if char_list:
                ranges = self.chars_to_ranges(char_list)
                ranges_to_state[ranges] = state
        for ranges in sorted(ranges_to_state):
            key = self.ranges_to_string(ranges)
            state = ranges_to_state[ranges]
            file.write('      %s --> State %d\n' % (key, state['number']))
        for key in ('bol', 'eol', 'eof', 'else'):
            state = special_to_state.get(key, None)
            if state:
                file.write('      %s --> State %d\n' % (key, state['number']))

    @cython.locals(char_list=list, i=cython.Py_ssize_t, n=cython.Py_ssize_t, c1=cython.long, c2=cython.long)
    def chars_to_ranges(self, char_list):
        if False:
            for i in range(10):
                print('nop')
        char_list.sort()
        i = 0
        n = len(char_list)
        result = []
        while i < n:
            c1 = ord(char_list[i])
            c2 = c1
            i += 1
            while i < n and ord(char_list[i]) == c2 + 1:
                i += 1
                c2 += 1
            result.append((chr(c1), chr(c2)))
        return tuple(result)

    def ranges_to_string(self, range_list):
        if False:
            i = 10
            return i + 15
        return ','.join(map(self.range_to_string, range_list))

    def range_to_string(self, range_tuple):
        if False:
            while True:
                i = 10
        (c1, c2) = range_tuple
        if c1 == c2:
            return repr(c1)
        else:
            return '%s..%s' % (repr(c1), repr(c2))