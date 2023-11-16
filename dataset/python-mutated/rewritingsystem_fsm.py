class State:
    """
    A representation of a state managed by a ``StateMachine``.

    Attributes:
        name (instance of FreeGroupElement or string) -- State name which is also assigned to the Machine.
        transisitons (OrderedDict) -- Represents all the transitions of the state object.
        state_type (string) -- Denotes the type (accept/start/dead) of the state.
        rh_rule (instance of FreeGroupElement) -- right hand rule for dead state.
        state_machine (instance of StateMachine object) -- The finite state machine that the state belongs to.
    """

    def __init__(self, name, state_machine, state_type=None, rh_rule=None):
        if False:
            while True:
                i = 10
        self.name = name
        self.transitions = {}
        self.state_machine = state_machine
        self.state_type = state_type[0]
        self.rh_rule = rh_rule

    def add_transition(self, letter, state):
        if False:
            while True:
                i = 10
        '\n        Add a transition from the current state to a new state.\n\n        Keyword Arguments:\n            letter -- The alphabet element the current state reads to make the state transition.\n            state -- This will be an instance of the State object which represents a new state after in the transition after the alphabet is read.\n\n        '
        self.transitions[letter] = state

class StateMachine:
    """
    Representation of a finite state machine the manages the states and the transitions of the automaton.

    Attributes:
        states (dictionary) -- Collection of all registered `State` objects.
        name (str) -- Name of the state machine.
    """

    def __init__(self, name, automaton_alphabet):
        if False:
            while True:
                i = 10
        self.name = name
        self.automaton_alphabet = automaton_alphabet
        self.states = {}
        self.add_state('start', state_type='s')

    def add_state(self, state_name, state_type=None, rh_rule=None):
        if False:
            print('Hello World!')
        "\n        Instantiate a state object and stores it in the 'states' dictionary.\n\n        Arguments:\n            state_name (instance of FreeGroupElement or string) -- name of the new states.\n            state_type (string) -- Denotes the type (accept/start/dead) of the state added.\n            rh_rule (instance of FreeGroupElement) -- right hand rule for dead state.\n\n        "
        new_state = State(state_name, self, state_type, rh_rule)
        self.states[state_name] = new_state

    def __repr__(self):
        if False:
            return 10
        return '%s' % self.name