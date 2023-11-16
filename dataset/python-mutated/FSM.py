"""This module implements a Finite State Machine (FSM). In addition to state
this FSM also maintains a user defined "memory". So this FSM can be used as a
Push-down Automata (PDA) since a PDA is a FSM + memory.

The following describes how the FSM works, but you will probably also need to
see the example function to understand how the FSM is used in practice.

You define an FSM by building tables of transitions. For a given input symbol
the process() method uses these tables to decide what action to call and what
the next state will be. The FSM has a table of transitions that associate:

        (input_symbol, current_state) --> (action, next_state)

Where "action" is a function you define. The symbols and states can be any
objects. You use the add_transition() and add_transition_list() methods to add
to the transition table. The FSM also has a table of transitions that
associate:

        (current_state) --> (action, next_state)

You use the add_transition_any() method to add to this transition table. The
FSM also has one default transition that is not associated with any specific
input_symbol or state. You use the set_default_transition() method to set the
default transition.

When an action function is called it is passed a reference to the FSM. The
action function may then access attributes of the FSM such as input_symbol,
current_state, or "memory". The "memory" attribute can be any object that you
want to pass along to the action functions. It is not used by the FSM itself.
For parsing you would typically pass a list to be used as a stack.

The processing sequence is as follows. The process() method is given an
input_symbol to process. The FSM will search the table of transitions that
associate:

        (input_symbol, current_state) --> (action, next_state)

If the pair (input_symbol, current_state) is found then process() will call the
associated action function and then set the current state to the next_state.

If the FSM cannot find a match for (input_symbol, current_state) it will then
search the table of transitions that associate:

        (current_state) --> (action, next_state)

If the current_state is found then the process() method will call the
associated action function and then set the current state to the next_state.
Notice that this table lacks an input_symbol. It lets you define transitions
for a current_state and ANY input_symbol. Hence, it is called the "any" table.
Remember, it is always checked after first searching the table for a specific
(input_symbol, current_state).

For the case where the FSM did not match either of the previous two cases the
FSM will try to use the default transition. If the default transition is
defined then the process() method will call the associated action function and
then set the current state to the next_state. This lets you define a default
transition as a catch-all case. You can think of it as an exception handler.
There can be only one default transition.

Finally, if none of the previous cases are defined for an input_symbol and
current_state then the FSM will raise an exception. This may be desirable, but
you can always prevent this just by defining a default transition.

Noah Spurrier 20020822

PEXPECT LICENSE

    This license is approved by the OSI and FSF as GPL-compatible.
        http://opensource.org/licenses/isc-license.txt

    Copyright (c) 2012, Noah Spurrier <noah@noah.org>
    PERMISSION TO USE, COPY, MODIFY, AND/OR DISTRIBUTE THIS SOFTWARE FOR ANY
    PURPOSE WITH OR WITHOUT FEE IS HEREBY GRANTED, PROVIDED THAT THE ABOVE
    COPYRIGHT NOTICE AND THIS PERMISSION NOTICE APPEAR IN ALL COPIES.
    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""

class ExceptionFSM(Exception):
    """This is the FSM Exception class."""

    def __init__(self, value):
        if False:
            return 10
        self.value = value

    def __str__(self):
        if False:
            print('Hello World!')
        return 'ExceptionFSM: ' + str(self.value)

class FSM:
    """This is a Finite State Machine (FSM).
    """

    def __init__(self, initial_state, memory=None):
        if False:
            for i in range(10):
                print('nop')
        'This creates the FSM. You set the initial state here. The "memory"\n        attribute is any object that you want to pass along to the action\n        functions. It is not used by the FSM. For parsing you would typically\n        pass a list to be used as a stack. '
        self.state_transitions = {}
        self.state_transitions_any = {}
        self.default_transition = None
        self.input_symbol = None
        self.initial_state = initial_state
        self.current_state = self.initial_state
        self.next_state = None
        self.action = None
        self.memory = memory

    def reset(self):
        if False:
            print('Hello World!')
        'This sets the current_state to the initial_state and sets\n        input_symbol to None. The initial state was set by the constructor\n        __init__(). '
        self.current_state = self.initial_state
        self.input_symbol = None

    def add_transition(self, input_symbol, state, action=None, next_state=None):
        if False:
            while True:
                i = 10
        'This adds a transition that associates:\n\n                (input_symbol, current_state) --> (action, next_state)\n\n        The action may be set to None in which case the process() method will\n        ignore the action and only set the next_state. The next_state may be\n        set to None in which case the current state will be unchanged.\n\n        You can also set transitions for a list of symbols by using\n        add_transition_list(). '
        if next_state is None:
            next_state = state
        self.state_transitions[input_symbol, state] = (action, next_state)

    def add_transition_list(self, list_input_symbols, state, action=None, next_state=None):
        if False:
            for i in range(10):
                print('nop')
        'This adds the same transition for a list of input symbols.\n        You can pass a list or a string. Note that it is handy to use\n        string.digits, string.whitespace, string.letters, etc. to add\n        transitions that match character classes.\n\n        The action may be set to None in which case the process() method will\n        ignore the action and only set the next_state. The next_state may be\n        set to None in which case the current state will be unchanged. '
        if next_state is None:
            next_state = state
        for input_symbol in list_input_symbols:
            self.add_transition(input_symbol, state, action, next_state)

    def add_transition_any(self, state, action=None, next_state=None):
        if False:
            return 10
        'This adds a transition that associates:\n\n                (current_state) --> (action, next_state)\n\n        That is, any input symbol will match the current state.\n        The process() method checks the "any" state associations after it first\n        checks for an exact match of (input_symbol, current_state).\n\n        The action may be set to None in which case the process() method will\n        ignore the action and only set the next_state. The next_state may be\n        set to None in which case the current state will be unchanged. '
        if next_state is None:
            next_state = state
        self.state_transitions_any[state] = (action, next_state)

    def set_default_transition(self, action, next_state):
        if False:
            i = 10
            return i + 15
        'This sets the default transition. This defines an action and\n        next_state if the FSM cannot find the input symbol and the current\n        state in the transition list and if the FSM cannot find the\n        current_state in the transition_any list. This is useful as a final\n        fall-through state for catching errors and undefined states.\n\n        The default transition can be removed by setting the attribute\n        default_transition to None. '
        self.default_transition = (action, next_state)

    def get_transition(self, input_symbol, state):
        if False:
            while True:
                i = 10
        'This returns (action, next state) given an input_symbol and state.\n        This does not modify the FSM state, so calling this method has no side\n        effects. Normally you do not call this method directly. It is called by\n        process().\n\n        The sequence of steps to check for a defined transition goes from the\n        most specific to the least specific.\n\n        1. Check state_transitions[] that match exactly the tuple,\n            (input_symbol, state)\n\n        2. Check state_transitions_any[] that match (state)\n            In other words, match a specific state and ANY input_symbol.\n\n        3. Check if the default_transition is defined.\n            This catches any input_symbol and any state.\n            This is a handler for errors, undefined states, or defaults.\n\n        4. No transition was defined. If we get here then raise an exception.\n        '
        if (input_symbol, state) in self.state_transitions:
            return self.state_transitions[input_symbol, state]
        elif state in self.state_transitions_any:
            return self.state_transitions_any[state]
        elif self.default_transition is not None:
            return self.default_transition
        else:
            raise ExceptionFSM('Transition is undefined: (%s, %s).' % (str(input_symbol), str(state)))

    def process(self, input_symbol):
        if False:
            for i in range(10):
                print('nop')
        'This is the main method that you call to process input. This may\n        cause the FSM to change state and call an action. This method calls\n        get_transition() to find the action and next_state associated with the\n        input_symbol and current_state. If the action is None then the action\n        is not called and only the current state is changed. This method\n        processes one complete input symbol. You can process a list of symbols\n        (or a string) by calling process_list(). '
        self.input_symbol = input_symbol
        (self.action, self.next_state) = self.get_transition(self.input_symbol, self.current_state)
        if self.action is not None:
            self.action(self)
        self.current_state = self.next_state
        self.next_state = None

    def process_list(self, input_symbols):
        if False:
            return 10
        'This takes a list and sends each element to process(). The list may\n        be a string or any iterable object. '
        for s in input_symbols:
            self.process(s)
import sys
import string
PY3 = sys.version_info[0] >= 3

def BeginBuildNumber(fsm):
    if False:
        return 10
    fsm.memory.append(fsm.input_symbol)

def BuildNumber(fsm):
    if False:
        i = 10
        return i + 15
    s = fsm.memory.pop()
    s = s + fsm.input_symbol
    fsm.memory.append(s)

def EndBuildNumber(fsm):
    if False:
        while True:
            i = 10
    s = fsm.memory.pop()
    fsm.memory.append(int(s))

def DoOperator(fsm):
    if False:
        print('Hello World!')
    ar = fsm.memory.pop()
    al = fsm.memory.pop()
    if fsm.input_symbol == '+':
        fsm.memory.append(al + ar)
    elif fsm.input_symbol == '-':
        fsm.memory.append(al - ar)
    elif fsm.input_symbol == '*':
        fsm.memory.append(al * ar)
    elif fsm.input_symbol == '/':
        fsm.memory.append(al / ar)

def DoEqual(fsm):
    if False:
        return 10
    print(str(fsm.memory.pop()))

def Error(fsm):
    if False:
        return 10
    print('That does not compute.')
    print(str(fsm.input_symbol))

def main():
    if False:
        print('Hello World!')
    "This is where the example starts and the FSM state transitions are\n    defined. Note that states are strings (such as 'INIT'). This is not\n    necessary, but it makes the example easier to read. "
    f = FSM('INIT', [])
    f.set_default_transition(Error, 'INIT')
    f.add_transition_any('INIT', None, 'INIT')
    f.add_transition('=', 'INIT', DoEqual, 'INIT')
    f.add_transition_list(string.digits, 'INIT', BeginBuildNumber, 'BUILDING_NUMBER')
    f.add_transition_list(string.digits, 'BUILDING_NUMBER', BuildNumber, 'BUILDING_NUMBER')
    f.add_transition_list(string.whitespace, 'BUILDING_NUMBER', EndBuildNumber, 'INIT')
    f.add_transition_list('+-*/', 'INIT', DoOperator, 'INIT')
    print()
    print('Enter an RPN Expression.')
    print('Numbers may be integers. Operators are * / + -')
    print('Use the = sign to evaluate and print the expression.')
    print('For example: ')
    print('    167 3 2 2 * * * 1 - =')
    inputstr = (input if PY3 else raw_input)('> ')
    f.process_list(inputstr)
if __name__ == '__main__':
    main()