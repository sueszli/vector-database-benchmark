"""Monte-Carlo Tree Search algorithm for game play."""
import math
import time
import numpy as np
import pyspiel

class Evaluator(object):
    """Abstract class representing an evaluation function for a game.

  The evaluation function takes in an intermediate state in the game and returns
  an evaluation of that state, which should correlate with chances of winning
  the game. It returns the evaluation from all player's perspectives.
  """

    def evaluate(self, state):
        if False:
            print('Hello World!')
        'Returns evaluation on given state.'
        raise NotImplementedError

    def prior(self, state):
        if False:
            print('Hello World!')
        'Returns a probability for each legal action in the given state.'
        raise NotImplementedError

class RandomRolloutEvaluator(Evaluator):
    """A simple evaluator doing random rollouts.

  This evaluator returns the average outcome of playing random actions from the
  given state until the end of the game.  n_rollouts is the number of random
  outcomes to be considered.
  """

    def __init__(self, n_rollouts=1, random_state=None):
        if False:
            return 10
        self.n_rollouts = n_rollouts
        self._random_state = random_state or np.random.RandomState()

    def evaluate(self, state):
        if False:
            i = 10
            return i + 15
        'Returns evaluation on given state.'
        result = None
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            while not working_state.is_terminal():
                if working_state.is_chance_node():
                    outcomes = working_state.chance_outcomes()
                    (action_list, prob_list) = zip(*outcomes)
                    action = self._random_state.choice(action_list, p=prob_list)
                else:
                    action = self._random_state.choice(working_state.legal_actions())
                working_state.apply_action(action)
            returns = np.array(working_state.returns())
            result = returns if result is None else result + returns
        return result / self.n_rollouts

    def prior(self, state):
        if False:
            for i in range(10):
                print('nop')
        'Returns equal probability for all actions.'
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            legal_actions = state.legal_actions(state.current_player())
            return [(action, 1.0 / len(legal_actions)) for action in legal_actions]

class SearchNode(object):
    """A node in the search tree.

  A SearchNode represents a state and possible continuations from it. Each child
  represents a possible action, and the expected result from doing so.

  Attributes:
    action: The action from the parent node's perspective. Not important for the
      root node, as the actions that lead to it are in the past.
    player: Which player made this action.
    prior: A prior probability for how likely this action will be selected.
    explore_count: How many times this node was explored.
    total_reward: The sum of rewards of rollouts through this node, from the
      parent node's perspective. The average reward of this node is
      `total_reward / explore_count`
    outcome: The rewards for all players if this is a terminal node or the
      subtree has been proven, otherwise None.
    children: A list of SearchNodes representing the possible actions from this
      node, along with their expected rewards.
  """
    __slots__ = ['action', 'player', 'prior', 'explore_count', 'total_reward', 'outcome', 'children']

    def __init__(self, action, player, prior):
        if False:
            i = 10
            return i + 15
        self.action = action
        self.player = player
        self.prior = prior
        self.explore_count = 0
        self.total_reward = 0.0
        self.outcome = None
        self.children = []

    def uct_value(self, parent_explore_count, uct_c):
        if False:
            while True:
                i = 10
        'Returns the UCT value of child.'
        if self.outcome is not None:
            return self.outcome[self.player]
        if self.explore_count == 0:
            return float('inf')
        return self.total_reward / self.explore_count + uct_c * math.sqrt(math.log(parent_explore_count) / self.explore_count)

    def puct_value(self, parent_explore_count, uct_c):
        if False:
            for i in range(10):
                print('nop')
        'Returns the PUCT value of child.'
        if self.outcome is not None:
            return self.outcome[self.player]
        return (self.explore_count and self.total_reward / self.explore_count) + uct_c * self.prior * math.sqrt(parent_explore_count) / (self.explore_count + 1)

    def sort_key(self):
        if False:
            while True:
                i = 10
        'Returns the best action from this node, either proven or most visited.\n\n    This ordering leads to choosing:\n    - Highest proven score > 0 over anything else, including a promising but\n      unproven action.\n    - A proven draw only if it has higher exploration than others that are\n      uncertain, or the others are losses.\n    - Uncertain action with most exploration over loss of any difficulty\n    - Hardest loss if everything is a loss\n    - Highest expected reward if explore counts are equal (unlikely).\n    - Longest win, if multiple are proven (unlikely due to early stopping).\n    '
        return (0 if self.outcome is None else self.outcome[self.player], self.explore_count, self.total_reward)

    def best_child(self):
        if False:
            return 10
        'Returns the best child in order of the sort key.'
        return max(self.children, key=SearchNode.sort_key)

    def children_str(self, state=None):
        if False:
            while True:
                i = 10
        "Returns the string representation of this node's children.\n\n    They are ordered based on the sort key, so order of being chosen to play.\n\n    Args:\n      state: A `pyspiel.State` object, to be used to convert the action id into\n        a human readable format. If None, the action integer id is used.\n    "
        return '\n'.join([c.to_str(state) for c in reversed(sorted(self.children, key=SearchNode.sort_key))])

    def to_str(self, state=None):
        if False:
            while True:
                i = 10
        'Returns the string representation of this node.\n\n    Args:\n      state: A `pyspiel.State` object, to be used to convert the action id into\n        a human readable format. If None, the action integer id is used.\n    '
        action = state.action_to_string(state.current_player(), self.action) if state and self.action is not None else str(self.action)
        return '{:>6}: player: {}, prior: {:5.3f}, value: {:6.3f}, sims: {:5d}, outcome: {}, {:3d} children'.format(action, self.player, self.prior, self.explore_count and self.total_reward / self.explore_count, self.explore_count, '{:4.1f}'.format(self.outcome[self.player]) if self.outcome else 'none', len(self.children))

    def __str__(self):
        if False:
            return 10
        return self.to_str(None)

class MCTSBot(pyspiel.Bot):
    """Bot that uses Monte-Carlo Tree Search algorithm."""

    def __init__(self, game, uct_c, max_simulations, evaluator, solve=True, random_state=None, child_selection_fn=SearchNode.uct_value, dirichlet_noise=None, verbose=False, dont_return_chance_node=False):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a MCTS Search algorithm in the form of a bot.\n\n    In multiplayer games, or non-zero-sum games, the players will play the\n    greedy strategy.\n\n    Args:\n      game: A pyspiel.Game to play.\n      uct_c: The exploration constant for UCT.\n      max_simulations: How many iterations of MCTS to perform. Each simulation\n        will result in one call to the evaluator. Memory usage should grow\n        linearly with simulations * branching factor. How many nodes in the\n        search tree should be evaluated. This is correlated with memory size and\n        tree depth.\n      evaluator: A `Evaluator` object to use to evaluate a leaf node.\n      solve: Whether to back up solved states.\n      random_state: An optional numpy RandomState to make it deterministic.\n      child_selection_fn: A function to select the child in the descent phase.\n        The default is UCT.\n      dirichlet_noise: A tuple of (epsilon, alpha) for adding dirichlet noise to\n        the policy at the root. This is from the alpha-zero paper.\n      verbose: Whether to print information about the search tree before\n        returning the action. Useful for confirming the search is working\n        sensibly.\n      dont_return_chance_node: If true, do not stop expanding at chance nodes.\n        Enabled for AlphaZero.\n\n    Raises:\n      ValueError: if the game type isn't supported.\n    "
        pyspiel.Bot.__init__(self)
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError('Game must have terminal rewards.')
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError('Game must have sequential turns.')
        self._game = game
        self.uct_c = uct_c
        self.max_simulations = max_simulations
        self.evaluator = evaluator
        self.verbose = verbose
        self.solve = solve
        self.max_utility = game.max_utility()
        self._dirichlet_noise = dirichlet_noise
        self._random_state = random_state or np.random.RandomState()
        self._child_selection_fn = child_selection_fn
        self.dont_return_chance_node = dont_return_chance_node

    def restart_at(self, state):
        if False:
            return 10
        pass

    def step_with_policy(self, state):
        if False:
            i = 10
            return i + 15
        "Returns bot's policy and action at given state."
        t1 = time.time()
        root = self.mcts_search(state)
        best = root.best_child()
        if self.verbose:
            seconds = time.time() - t1
            print('Finished {} sims in {:.3f} secs, {:.1f} sims/s'.format(root.explore_count, seconds, root.explore_count / seconds))
            print('Root:')
            print(root.to_str(state))
            print('Children:')
            print(root.children_str(state))
            if best.children:
                chosen_state = state.clone()
                chosen_state.apply_action(best.action)
                print('Children of chosen:')
                print(best.children_str(chosen_state))
        mcts_action = best.action
        policy = [(action, 1.0 if action == mcts_action else 0.0) for action in state.legal_actions(state.current_player())]
        return (policy, mcts_action)

    def step(self, state):
        if False:
            i = 10
            return i + 15
        return self.step_with_policy(state)[1]

    def _apply_tree_policy(self, root, state):
        if False:
            while True:
                i = 10
        "Applies the UCT policy to play the game until reaching a leaf node.\n\n    A leaf node is defined as a node that is terminal or has not been evaluated\n    yet. If it reaches a node that has been evaluated before but hasn't been\n    expanded, then expand it's children and continue.\n\n    Args:\n      root: The root node in the search tree.\n      state: The state of the game at the root node.\n\n    Returns:\n      visit_path: A list of nodes descending from the root node to a leaf node.\n      working_state: The state of the game at the leaf node.\n    "
        visit_path = [root]
        working_state = state.clone()
        current_node = root
        while not working_state.is_terminal() and current_node.explore_count > 0 or (working_state.is_chance_node() and self.dont_return_chance_node):
            if not current_node.children:
                legal_actions = self.evaluator.prior(working_state)
                if current_node is root and self._dirichlet_noise:
                    (epsilon, alpha) = self._dirichlet_noise
                    noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                    legal_actions = [(a, (1 - epsilon) * p + epsilon * n) for ((a, p), n) in zip(legal_actions, noise)]
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [SearchNode(action, player, prior) for (action, prior) in legal_actions]
            if working_state.is_chance_node():
                outcomes = working_state.chance_outcomes()
                (action_list, prob_list) = zip(*outcomes)
                action = self._random_state.choice(action_list, p=prob_list)
                chosen_child = next((c for c in current_node.children if c.action == action))
            else:
                chosen_child = max(current_node.children, key=lambda c: self._child_selection_fn(c, current_node.explore_count, self.uct_c))
            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)
        return (visit_path, working_state)

    def mcts_search(self, state):
        if False:
            print('Hello World!')
        'A vanilla Monte-Carlo Tree Search algorithm.\n\n    This algorithm searches the game tree from the given state.\n    At the leaf, the evaluator is called if the game state is not terminal.\n    A total of max_simulations states are explored.\n\n    At every node, the algorithm chooses the action with the highest PUCT value,\n    defined as: `Q/N + c * prior * sqrt(parent_N) / N`, where Q is the total\n    reward after the action, and N is the number of times the action was\n    explored in this position. The input parameter c controls the balance\n    between exploration and exploitation; higher values of c encourage\n    exploration of under-explored nodes. Unseen actions are always explored\n    first.\n\n    At the end of the search, the chosen action is the action that has been\n    explored most often. This is the action that is returned.\n\n    This implementation supports sequential n-player games, with or without\n    chance nodes. All players maximize their own reward and ignore the other\n    players\' rewards. This corresponds to max^n for n-player games. It is the\n    norm for zero-sum games, but doesn\'t have any special handling for\n    non-zero-sum games. It doesn\'t have any special handling for imperfect\n    information games.\n\n    The implementation also supports backing up solved states, i.e. MCTS-Solver.\n    The implementation is general in that it is based on a max^n backup (each\n    player greedily chooses their maximum among proven children values, or there\n    exists one child whose proven value is game.max_utility()), so it will work\n    for multiplayer, general-sum, and arbitrary payoff games (not just win/loss/\n    draw games). Also chance nodes are considered proven only if all children\n    have the same value.\n\n    Some references:\n    - Sturtevant, An Analysis of UCT in Multi-Player Games,  2008,\n      https://web.cs.du.edu/~sturtevant/papers/multi-player_UCT.pdf\n    - Nijssen, Monte-Carlo Tree Search for Multi-Player Games, 2013,\n      https://project.dke.maastrichtuniversity.nl/games/files/phd/Nijssen_thesis.pdf\n    - Silver, AlphaGo Zero: Starting from scratch, 2017\n      https://deepmind.com/blog/article/alphago-zero-starting-scratch\n    - Winands, Bjornsson, and Saito, "Monte-Carlo Tree Search Solver", 2008.\n      https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf\n\n    Arguments:\n      state: pyspiel.State object, state to search from\n\n    Returns:\n      The most visited move from the root node.\n    '
        root = SearchNode(None, state.current_player(), 1)
        for _ in range(self.max_simulations):
            (visit_path, working_state) = self._apply_tree_policy(root, state)
            if working_state.is_terminal():
                returns = working_state.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = self.evaluator.evaluate(working_state)
                solved = False
            while visit_path:
                decision_node_idx = -1
                while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
                    decision_node_idx -= 1
                target_return = returns[visit_path[decision_node_idx].player]
                node = visit_path.pop()
                node.total_reward += target_return
                node.explore_count += 1
                if solved and node.children:
                    player = node.children[0].player
                    if player == pyspiel.PlayerId.CHANCE:
                        outcome = node.children[0].outcome
                        if outcome is not None and all((np.array_equal(c.outcome, outcome) for c in node.children)):
                            node.outcome = outcome
                        else:
                            solved = False
                    else:
                        best = None
                        all_solved = True
                        for child in node.children:
                            if child.outcome is None:
                                all_solved = False
                            elif best is None or child.outcome[player] > best.outcome[player]:
                                best = child
                        if best is not None and (all_solved or best.outcome[player] == self.max_utility):
                            node.outcome = best.outcome
                        else:
                            solved = False
            if root.outcome is not None:
                break
        return root