"""Export game trees in gambit format.

An exporter for the .efg format used by Gambit:
http://www.gambit-project.org/gambit14/formats.html

See `examples/gambit_example.py` for an example of usage.

"""
import collections
import functools

def quote(x):
    if False:
        for i in range(10):
            print('nop')
    return f'"{x}"'

def export_gambit(game):
    if False:
        i = 10
        return i + 15
    'Builds gambit representation of the game tree.\n\n  Args:\n    game: A `pyspiel.Game` object.\n\n  Returns:\n    string: Gambit tree\n  '
    players = ' '.join([f'"Pl{i}"' for i in range(game.num_players())])
    ret = f'EFG 2 R {quote(game)} {{ {players} }} \n'
    terminal_idx = 1
    chance_idx = 1
    infoset_idx = [0] * game.num_players()

    def infoset_next_id(player):
        if False:
            print('Hello World!')
        nonlocal infoset_idx
        infoset_idx[player] += 1
        return infoset_idx[player]
    infoset_tables = [collections.defaultdict(functools.partial(infoset_next_id, player)) for player in range(game.num_players())]

    def build_tree(state, depth):
        if False:
            for i in range(10):
                print('nop')
        nonlocal ret, terminal_idx, chance_idx, infoset_tables
        ret += ' ' * depth
        state_str = str(state)
        if len(state_str) > 10:
            state_str = ''
        if state.is_terminal():
            utils = ' '.join(map(str, state.returns()))
            ret += f't {quote(state_str)} {terminal_idx} "" {{ {utils} }}\n'
            terminal_idx += 1
            return
        if state.is_chance_node():
            ret += f'c {quote(state_str)} {chance_idx} "" {{ '
            for (action, prob) in state.chance_outcomes():
                action_str = state.action_to_string(state.current_player(), action)
                ret += f'{quote(action_str)} {prob:.16f} '
            ret += ' } 0\n'
            chance_idx += 1
        else:
            player = state.current_player()
            gambit_player = player + 1
            infoset = state.information_state_string()
            infoset_idx = infoset_tables[player][infoset]
            ret += f'p {quote(state_str)} {gambit_player} {infoset_idx} "" {{ '
            for action in state.legal_actions():
                action_str = state.action_to_string(state.current_player(), action)
                ret += f'{quote(action_str)} '
            ret += ' } 0\n'
        for action in state.legal_actions():
            child = state.child(action)
            build_tree(child, depth + 1)
    build_tree(game.new_initial_state(), 0)
    return ret