"""Visualizing game trees with graphviz.

GameTree builds a `pygraphviz.AGraph` reprensentation of the game tree. The
resulting tree can be directly visualized in Jupyter notebooks or Google Colab
via SVG plotting - or written to a file by calling `draw(filename, prog="dot")`.

See `examples/treeviz_example.py` for a more detailed example.

This module relies on external dependencies, which need to be installed before
use. On a debian system follow these steps:
```
sudo apt-get install graphviz libgraphviz-dev
pip install pygraphviz
```
"""
import collections
import pyspiel
try:
    import pygraphviz
except (ImportError, Exception) as e:
    raise ImportError(str(e) + '\nPlease make sure to install the following dependencies:\nsudo apt-get install graphviz libgraphviz-dev\npip install pygraphviz') from None
_PLAYER_SHAPES = {0: 'square', 1: 'ellipse'}
_PLAYER_COLORS = {-1: 'black', 0: 'blue', 1: 'red'}
_FONTSIZE = 8
_WIDTH = _HEIGHT = 0.25
_ARROWSIZE = 0.5
_MARGIN = 0.01

def default_node_decorator(state):
    if False:
        print('Hello World!')
    'Decorates a state-node of the game tree.\n\n  This method can be called by a custom decorator to prepopulate the attributes\n  dictionary. Then only relevant attributes need to be changed, or added.\n\n  Args:\n    state: The state.\n\n  Returns:\n    `dict` with graphviz node style attributes.\n  '
    player = state.current_player()
    attrs = {'label': '', 'fontsize': _FONTSIZE, 'width': _WIDTH, 'height': _HEIGHT, 'margin': _MARGIN}
    if state.is_terminal():
        attrs['label'] = ', '.join(map(str, state.returns()))
        attrs['shape'] = 'diamond'
    elif state.is_chance_node():
        attrs['shape'] = 'point'
        attrs['width'] = _WIDTH / 2.0
        attrs['height'] = _HEIGHT / 2.0
    else:
        attrs['label'] = str(state.information_state_string())
        attrs['shape'] = _PLAYER_SHAPES.get(player, 'ellipse')
        attrs['color'] = _PLAYER_COLORS.get(player, 'black')
    return attrs

def default_edge_decorator(parent, unused_child, action):
    if False:
        print('Hello World!')
    'Decorates a state-node of the game tree.\n\n  This method can be called by a custom decorator to prepopulate the attributes\n  dictionary. Then only relevant attributes need to be changed, or added.\n\n  Args:\n    parent: The parent state.\n    unused_child: The child state, not used in the default decorator.\n    action: `int` the selected action in the parent state.\n\n  Returns:\n    `dict` with graphviz node style attributes.\n  '
    player = parent.current_player()
    attrs = {'label': ' ' + parent.action_to_string(player, action), 'fontsize': _FONTSIZE, 'arrowsize': _ARROWSIZE}
    attrs['color'] = _PLAYER_COLORS.get(player, 'black')
    return attrs

class GameTree(pygraphviz.AGraph):
    """Builds `pygraphviz.AGraph` of the game tree.

  Attributes:
    game: A `pyspiel.Game` object.
    depth_limit: Maximum depth of the tree. Optional, default=-1 (no limit).
    node_decorator: Decorator function for nodes (states). Optional, default=
      `treeviz.default_node_decorator`.
    edge_decorator: Decorator function for edges (actions). Optional, default=
      `treeviz.default_edge_decorator`.
    group_terminal: Whether to display all terminal states at same level,
      default=False.
    group_infosets: Whether to group infosets together, default=False.
    group_pubsets: Whether to group public sets together, default=False.
    target_pubset: Whether to group all public sets "*" or a specific one.
    infoset_attrs: Attributes to style infoset grouping.
    pubset_attrs: Attributes to style public set grouping.
    kwargs: Keyword arguments passed on to `pygraphviz.AGraph.__init__`.
  """

    def __init__(self, game=None, depth_limit=-1, node_decorator=default_node_decorator, edge_decorator=default_edge_decorator, group_terminal=False, group_infosets=False, group_pubsets=False, target_pubset='*', infoset_attrs=None, pubset_attrs=None, **kwargs):
        if False:
            return 10
        kwargs['directed'] = kwargs.get('directed', True)
        super(GameTree, self).__init__(**kwargs)
        if game is None:
            return
        self.game = game
        self._node_decorator = node_decorator
        self._edge_decorator = edge_decorator
        self._group_infosets = group_infosets
        self._group_pubsets = group_pubsets
        if self._group_infosets:
            if not self.game.get_type().provides_information_state_string:
                raise RuntimeError('Grouping of infosets requested, but the game does not provide information state string.')
        if self._group_pubsets:
            if not self.game.get_type().provides_factored_observation_string:
                raise RuntimeError('Grouping of public sets requested, but the game does not provide factored observations strings.')
        self._infosets = collections.defaultdict(lambda : [])
        self._pubsets = collections.defaultdict(lambda : [])
        self._terminal_nodes = []
        root = game.new_initial_state()
        self.add_node(self.state_to_str(root), **self._node_decorator(root))
        self._build_tree(root, 0, depth_limit)
        for ((player, info_state), sibblings) in self._infosets.items():
            cluster_name = 'cluster_{}_{}'.format(player, info_state)
            self.add_subgraph(sibblings, cluster_name, **infoset_attrs or {'style': 'dashed'})
        for (pubset, sibblings) in self._pubsets.items():
            if target_pubset == '*' or target_pubset == pubset:
                cluster_name = 'cluster_{}'.format(pubset)
                self.add_subgraph(sibblings, cluster_name, **pubset_attrs or {'style': 'dashed'})
        if group_terminal:
            self.add_subgraph(self._terminal_nodes, rank='same')

    def state_to_str(self, state):
        if False:
            i = 10
            return i + 15
        'Unique string representation of a state.\n\n    Args:\n      state: The state.\n\n    Returns:\n      String representation of state.\n    '
        assert not state.is_simultaneous_node()
        return ' ' + state.history_str()

    def _build_tree(self, state, depth, depth_limit):
        if False:
            for i in range(10):
                print('nop')
        'Recursively builds the game tree.'
        state_str = self.state_to_str(state)
        if state.is_terminal():
            self._terminal_nodes.append(state_str)
            return
        if depth > depth_limit >= 0:
            return
        for action in state.legal_actions():
            child = state.child(action)
            child_str = self.state_to_str(child)
            self.add_node(child_str, **self._node_decorator(child))
            self.add_edge(state_str, child_str, **self._edge_decorator(state, child, action))
            if self._group_infosets and (not child.is_chance_node()) and (not child.is_terminal()):
                player = child.current_player()
                info_state = child.information_state_string()
                self._infosets[player, info_state].append(child_str)
            if self._group_pubsets:
                pub_obs_history = str(pyspiel.PublicObservationHistory(child))
                self._pubsets[pub_obs_history].append(child_str)
            self._build_tree(child, depth + 1, depth_limit)

    def _repr_svg_(self):
        if False:
            while True:
                i = 10
        'Allows to render directly in Jupyter notebooks and Google Colab.'
        if not self.has_layout:
            self.layout(prog='dot')
        return self.draw(format='svg').decode(self.encoding)