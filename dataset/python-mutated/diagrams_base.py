"""
    transitions.extensions.diagrams_base
    ------------------------------------

    The class BaseGraph implements the common ground for Graphviz backends.
"""
import copy
import abc
import logging
import six
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

@six.add_metaclass(abc.ABCMeta)
class BaseGraph(object):
    """Provides the common foundation for graphs generated either with pygraphviz or graphviz. This abstract class
    should not be instantiated directly. Use .(py)graphviz.(Nested)Graph instead.
    Attributes:
        machine (GraphMachine): The associated GraphMachine
        fsm_graph (object): The AGraph-like object that holds the graphviz information
    """

    def __init__(self, machine):
        if False:
            while True:
                i = 10
        self.machine = machine
        self.fsm_graph = None
        self.generate()

    @abc.abstractmethod
    def generate(self):
        if False:
            print('Hello World!')
        'Triggers the generation of a graph.'

    @abc.abstractmethod
    def set_previous_transition(self, src, dst):
        if False:
            return 10
        "Sets the styling of an edge to 'previous'\n        Args:\n            src (str): Name of the source state\n            dst (str): Name of the destination\n        "

    @abc.abstractmethod
    def reset_styling(self):
        if False:
            i = 10
            return i + 15
        'Resets the styling of the currently generated graph.'

    @abc.abstractmethod
    def set_node_style(self, state, style):
        if False:
            print('Hello World!')
        'Sets the style of nodes associated with a model state\n        Args:\n            state (str, Enum or list): Name of the state(s) or Enum(s)\n            style (str): Name of the style\n        '

    @abc.abstractmethod
    def get_graph(self, title=None, roi_state=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns a graph object.\n        Args:\n            title (str): Title of the generated graph\n            roi_state (State): If not None, the returned graph will only contain edges and states connected to it.\n        Returns:\n             A graph instance with a `draw` that allows to render the graph.\n        '

    def _convert_state_attributes(self, state):
        if False:
            i = 10
            return i + 15
        label = state.get('label', state['name'])
        if self.machine.show_state_attributes:
            if 'tags' in state:
                label += ' [' + ', '.join(state['tags']) + ']'
            if 'on_enter' in state:
                label += '\\l- enter:\\l  + ' + '\\l  + '.join(state['on_enter'])
            if 'on_exit' in state:
                label += '\\l- exit:\\l  + ' + '\\l  + '.join(state['on_exit'])
            if 'timeout' in state:
                label += '\\l- timeout(' + state['timeout'] + 's) -> (' + ', '.join(state['on_timeout']) + ')'
        return label + '\\l'

    def _get_state_names(self, state):
        if False:
            return 10
        if isinstance(state, (list, tuple, set)):
            for res in state:
                for inner in self._get_state_names(res):
                    yield inner
        else:
            yield (self.machine.state_cls.separator.join(self.machine._get_enum_path(state)) if hasattr(state, 'name') else state)

    def _transition_label(self, tran):
        if False:
            while True:
                i = 10
        edge_label = tran.get('label', tran['trigger'])
        if 'dest' not in tran:
            edge_label += ' [internal]'
        if self.machine.show_conditions and any((prop in tran for prop in ['conditions', 'unless'])):
            edge_label = '{edge_label} [{conditions}]'.format(edge_label=edge_label, conditions=' & '.join(tran.get('conditions', []) + ['!' + u for u in tran.get('unless', [])]))
        return edge_label

    def _get_global_name(self, path):
        if False:
            while True:
                i = 10
        if path:
            state = path.pop(0)
            with self.machine(state):
                return self._get_global_name(path)
        else:
            return self.machine.get_global_name()

    def _flatten(self, *lists):
        if False:
            return 10
        return (e for a in lists for e in (self._flatten(*a) if isinstance(a, (tuple, list)) else (a.name if hasattr(a, 'name') else a,)))

    def _get_elements(self):
        if False:
            return 10
        states = []
        transitions = []
        try:
            markup = self.machine.get_markup_config()
            queue = [([], markup)]
            while queue:
                (prefix, scope) = queue.pop(0)
                for transition in scope.get('transitions', []):
                    if prefix:
                        tran = copy.copy(transition)
                        tran['source'] = self.machine.state_cls.separator.join(prefix + [tran['source']])
                        if 'dest' in tran:
                            tran['dest'] = self.machine.state_cls.separator.join(prefix + [tran['dest']])
                    else:
                        tran = transition
                    transitions.append(tran)
                for state in scope.get('children', []) + scope.get('states', []):
                    if not prefix:
                        sta = state
                        states.append(sta)
                    ini = state.get('initial', [])
                    if not isinstance(ini, list):
                        ini = ini.name if hasattr(ini, 'name') else ini
                        tran = dict(trigger='', source=self.machine.state_cls.separator.join(prefix + [state['name']]), dest=self.machine.state_cls.separator.join(prefix + [state['name'], ini]))
                        transitions.append(tran)
                    if state.get('children', []):
                        queue.append((prefix + [state['name']], state))
        except KeyError:
            _LOGGER.error('Graph creation incomplete!')
        return (states, transitions)