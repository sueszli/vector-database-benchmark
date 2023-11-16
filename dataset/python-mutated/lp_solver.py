"""LP Solver for two-player zero-sum games."""
import cvxopt
import numpy as np
from open_spiel.python.egt import utils
import pyspiel
OBJ_MAX = 1
OBJ_MIN = 2
CONS_TYPE_LEQ = 3
CONS_TYPE_GEQ = 4
CONS_TYPE_EQ = 5
DOMINANCE_STRICT = 1
DOMINANCE_VERY_WEAK = 2
DOMINANCE_WEAK = 3

class _Variable(object):
    """A variable in an LP."""

    def __init__(self, vid, lb=None, ub=None):
        if False:
            for i in range(10):
                print('nop')
        "Creates a variable in a linear program.\n\n    Args:\n      vid: (integer) the variable id (should be unique for each variable)\n      lb: the lower bound on the variable's value (None means no lower bound)\n      ub: the upper bound on the variable's valie (None means no upper bound)\n    "
        self.vid = vid
        self.lb = lb
        self.ub = ub

class _Constraint(object):
    """A constraint in an LP."""

    def __init__(self, cid, ctype):
        if False:
            print('Hello World!')
        'Creates a constraint in a linear program.\n\n    Args:\n      cid: (integer) the constraint id (should be unique for each constraint)\n      ctype: the constraint type (CONS_TYPE_{LEQ, GEQ, EQ})\n    '
        self.cid = cid
        self.ctype = ctype
        self.coeffs = {}
        self.rhs = None

class LinearProgram(object):
    """A object used to provide a user-friendly API for building LPs."""

    def __init__(self, objective):
        if False:
            for i in range(10):
                print('nop')
        assert objective == OBJ_MIN or objective == OBJ_MAX
        self._valid_constraint_types = [CONS_TYPE_EQ, CONS_TYPE_LEQ, CONS_TYPE_GEQ]
        self._objective = objective
        self._obj_coeffs = {}
        self._vars = {}
        self._cons = {}
        self._var_list = []
        self._leq_cons_list = []
        self._eq_cons_list = []

    def add_or_reuse_variable(self, label, lb=None, ub=None):
        if False:
            return 10
        'Adds a variable to this LP, or reuses one if the label exists.\n\n    If the variable already exists, simply checks that the upper and lower\n    bounds are the same as previously specified.\n\n    Args:\n      label: a label to assign to this constraint\n      lb: a lower-bound value for this variable\n      ub: an upper-bound value for this variable\n    '
        var = self._vars.get(label)
        if var is not None:
            assert var.lb == lb and var.ub == ub
            return
        var = _Variable(len(self._var_list), lb, ub)
        self._vars[label] = var
        self._var_list.append(var)

    def add_or_reuse_constraint(self, label, ctype):
        if False:
            while True:
                i = 10
        "Adds a constraint to this LP, or reuses one if the label exists.\n\n     If the constraint is already present, simply checks it's the same type as\n     previously specified.\n\n    Args:\n      label: a label to assign to this constraint\n      ctype: the constraint type (in CONS_TYPE_{LEQ,GEQ,EQ})\n    "
        assert ctype in self._valid_constraint_types
        cons = self._cons.get(label)
        if cons is not None:
            assert cons.ctype == ctype
            return
        if ctype == CONS_TYPE_LEQ or ctype == CONS_TYPE_GEQ:
            cons = _Constraint(len(self._leq_cons_list), ctype)
            self._cons[label] = cons
            self._leq_cons_list.append(cons)
        elif ctype == CONS_TYPE_EQ:
            cons = _Constraint(len(self._eq_cons_list), ctype)
            self._cons[label] = cons
            self._eq_cons_list.append(cons)
        else:
            assert False, 'Unknown constraint type'

    def set_obj_coeff(self, var_label, coeff):
        if False:
            print('Hello World!')
        'Sets a coefficient of a variable in the objective.'
        self._obj_coeffs[var_label] = coeff

    def set_cons_coeff(self, cons_label, var_label, coeff):
        if False:
            return 10
        'Sets a coefficient of a constraint in the LP.'
        self._cons[cons_label].coeffs[var_label] = coeff

    def add_to_cons_coeff(self, cons_label, var_label, add_coeff):
        if False:
            return 10
        'Sets a coefficient of a constraint in the LP.'
        val = self._cons[cons_label].coeffs.get(var_label)
        if val is None:
            val = 0
        self._cons[cons_label].coeffs[var_label] = val + add_coeff

    def set_cons_rhs(self, cons_label, value):
        if False:
            i = 10
            return i + 15
        'Sets the right-hand side of a constraint.'
        self._cons[cons_label].rhs = value

    def get_var_id(self, label):
        if False:
            for i in range(10):
                print('nop')
        var = self._vars.get(label)
        assert var is not None
        return var.vid

    def get_num_cons(self):
        if False:
            while True:
                i = 10
        return (len(self._leq_cons_list), len(self._eq_cons_list))

    def solve(self, solver=None):
        if False:
            for i in range(10):
                print('nop')
        "Solves the LP.\n\n    Args:\n      solver: the solver to use ('blas', 'lapack', 'glpk'). Defaults to None,\n        which then uses the cvxopt internal default.\n\n    Returns:\n      The solution as a dict of var label -> value, one for each variable.\n    "
        num_vars = len(self._var_list)
        num_eq_cons = len(self._eq_cons_list)
        num_leq_cons = len(self._leq_cons_list)
        for var in self._var_list:
            if var.lb is not None:
                num_leq_cons += 1
            if var.ub is not None:
                num_leq_cons += 1
        c = cvxopt.matrix([0.0] * num_vars)
        h = cvxopt.matrix([0.0] * num_leq_cons)
        g_mat = cvxopt.spmatrix([], [], [], (num_leq_cons, num_vars))
        a_mat = None
        b = None
        if num_eq_cons > 0:
            a_mat = cvxopt.spmatrix([], [], [], (num_eq_cons, num_vars))
            b = cvxopt.matrix([0.0] * num_eq_cons)
        for var_label in self._obj_coeffs:
            value = self._obj_coeffs[var_label]
            vid = self._vars[var_label].vid
            if self._objective == OBJ_MAX:
                c[vid] = -value
            else:
                c[vid] = value
        row = 0
        for cons in self._leq_cons_list:
            if cons.rhs is not None:
                h[row] = cons.rhs if cons.ctype == CONS_TYPE_LEQ else -cons.rhs
            for var_label in cons.coeffs:
                value = cons.coeffs[var_label]
                vid = self._vars[var_label].vid
                g_mat[row, vid] = value if cons.ctype == CONS_TYPE_LEQ else -value
            row += 1
        for var in self._var_list:
            if var.lb is not None:
                g_mat[row, var.vid] = -1.0
                h[row] = -var.lb
                row += 1
            if var.ub is not None:
                g_mat[row, var.vid] = 1.0
                h[row] = var.ub
                row += 1
        if num_eq_cons > 0:
            row = 0
            for cons in self._eq_cons_list:
                b[row] = cons.rhs if cons.rhs is not None else 0.0
                for var_label in cons.coeffs:
                    value = cons.coeffs[var_label]
                    vid = self._vars[var_label].vid
                    a_mat[row, vid] = value
                row += 1
        if num_eq_cons > 0:
            sol = cvxopt.solvers.lp(c, g_mat, h, a_mat, b, solver=solver)
        else:
            sol = cvxopt.solvers.lp(c, g_mat, h, solver=solver)
        return sol['x']

def solve_zero_sum_matrix_game(game):
    if False:
        print('Hello World!')
    'Solves a matrix game by using linear programming.\n\n  Args:\n    game: a pyspiel MatrixGame\n\n  Returns:\n    A 4-tuple containing:\n      - p0_sol (array-like): probability distribution over row actions\n      - p1_sol (array-like): probability distribution over column actions\n      - p0_sol_value, expected value to the first player\n      - p1_sol_value, expected value to the second player\n  '
    assert isinstance(game, pyspiel.MatrixGame)
    assert game.get_type().information == pyspiel.GameType.Information.ONE_SHOT
    assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
    num_rows = game.num_rows()
    num_cols = game.num_cols()
    cvxopt.solvers.options['show_progress'] = False
    lp0 = LinearProgram(OBJ_MAX)
    for r in range(num_rows):
        lp0.add_or_reuse_variable(r, lb=0)
    lp0.add_or_reuse_variable(num_rows)
    lp0.set_obj_coeff(num_rows, 1.0)
    for c in range(num_cols):
        lp0.add_or_reuse_constraint(c, CONS_TYPE_GEQ)
        for r in range(num_rows):
            lp0.set_cons_coeff(c, r, game.player_utility(0, r, c))
        lp0.set_cons_coeff(c, num_rows, -1.0)
    lp0.add_or_reuse_constraint(num_cols + 1, CONS_TYPE_EQ)
    lp0.set_cons_rhs(num_cols + 1, 1.0)
    for r in range(num_rows):
        lp0.set_cons_coeff(num_cols + 1, r, 1.0)
    sol = lp0.solve()
    p0_sol = sol[:-1]
    p0_sol_val = sol[-1]
    lp1 = LinearProgram(OBJ_MAX)
    for c in range(num_cols):
        lp1.add_or_reuse_variable(c, lb=0)
    lp1.add_or_reuse_variable(num_cols)
    lp1.set_obj_coeff(num_cols, 1)
    for r in range(num_rows):
        lp1.add_or_reuse_constraint(r, CONS_TYPE_GEQ)
        for c in range(num_cols):
            lp1.set_cons_coeff(r, c, game.player_utility(1, r, c))
        lp1.set_cons_coeff(r, num_cols, -1.0)
    lp1.add_or_reuse_constraint(num_rows + 1, CONS_TYPE_EQ)
    lp1.set_cons_rhs(num_rows + 1, 1.0)
    for c in range(num_cols):
        lp1.set_cons_coeff(num_rows + 1, c, 1.0)
    sol = lp1.solve()
    p1_sol = sol[:-1]
    p1_sol_val = sol[-1]
    return (p0_sol, p1_sol, p0_sol_val, p1_sol_val)

def is_dominated(action, game_or_payoffs, player, mode=DOMINANCE_STRICT, tol=1e-07, return_mixture=False):
    if False:
        i = 10
        return i + 15
    'Determines whether a pure strategy is dominated by any mixture strategies.\n\n  Args:\n    action: index of an action for `player`\n    game_or_payoffs: either a pyspiel matrix- or normal-form game, or a payoff\n      tensor for `player` with ndim == number of players\n    player: index of the player (an integer)\n    mode: dominance criterion: strict, weak, or very weak\n    tol: tolerance\n    return_mixture: whether to return the dominating strategy if one exists\n\n  Returns:\n    If `return_mixture`:\n      a dominating mixture strategy if one exists, or `None`.\n      the strategy is provided as a 1D numpy array of mixture weights.\n    Otherwise: True if a dominating strategy exists, False otherwise.\n  '
    assert mode in (DOMINANCE_STRICT, DOMINANCE_VERY_WEAK, DOMINANCE_WEAK)
    payoffs = utils.game_payoffs_array(game_or_payoffs)[player] if isinstance(game_or_payoffs, pyspiel.NormalFormGame) else np.asfarray(game_or_payoffs)
    payoffs = np.moveaxis(payoffs, player, 0)
    payoffs = payoffs.reshape((payoffs.shape[0], -1))
    (num_rows, num_cols) = payoffs.shape
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxtol'] = tol
    cvxopt.solvers.options['feastol'] = tol
    lp = LinearProgram(OBJ_MAX)
    for r in range(num_rows):
        if r == action:
            lp.add_or_reuse_variable(r, lb=0, ub=0)
        else:
            lp.add_or_reuse_variable(r, lb=0)
    if mode == DOMINANCE_STRICT:
        to_subtract = payoffs.min() - 1
    else:
        to_subtract = 0
        lp.add_or_reuse_constraint(num_cols, CONS_TYPE_EQ)
        lp.set_cons_rhs(num_cols, 1)
        for r in range(num_rows):
            if r != action:
                lp.set_cons_coeff(num_cols, r, 1)
    for c in range(num_cols):
        lp.add_or_reuse_constraint(c, CONS_TYPE_GEQ)
        lp.set_cons_rhs(c, payoffs[action, c] - to_subtract)
        for r in range(num_rows):
            if r != action:
                lp.set_cons_coeff(c, r, payoffs[r, c] - to_subtract)
    if mode == DOMINANCE_STRICT:
        for r in range(num_rows):
            if r != action:
                lp.set_obj_coeff(r, -1)
        mixture = lp.solve()
        if mixture is not None and np.sum(mixture) < 1 - tol:
            mixture = np.squeeze(mixture, 1) / np.sum(mixture)
        else:
            mixture = None
    if mode == DOMINANCE_VERY_WEAK:
        mixture = lp.solve()
        if mixture is not None:
            mixture = np.squeeze(mixture, 1)
    if mode == DOMINANCE_WEAK:
        for r in range(num_rows):
            lp.set_obj_coeff(r, payoffs[r].sum())
        mixture = lp.solve()
        if mixture is not None:
            mixture = np.squeeze(mixture, 1)
            if (np.dot(mixture, payoffs) - payoffs[action]).sum() <= tol:
                mixture = None
    return mixture if return_mixture else mixture is not None

def _pure_dominated_from_advantages(advantages, mode, tol=1e-07):
    if False:
        while True:
            i = 10
    if mode == DOMINANCE_STRICT:
        return (advantages > tol).all(1)
    if mode == DOMINANCE_WEAK:
        return (advantages >= -tol).all(1) & (advantages.sum(1) > tol)
    if mode == DOMINANCE_VERY_WEAK:
        return (advantages >= -tol).all(1)

def iterated_dominance(game_or_payoffs, mode, tol=1e-07):
    if False:
        print('Hello World!')
    "Reduces a strategy space using iterated dominance.\n\n  See: http://www.smallparty.com/yoram/classes/principles/nash.pdf\n\n  Args:\n    game_or_payoffs: either a pyspiel matrix- or normal-form game, or a payoff\n      tensor of dimension `num_players` + 1. First dimension is the player,\n      followed by the actions of all players, e.g. a 3x3 game (2 players) has\n      dimension [2,3,3].\n    mode: DOMINANCE_STRICT, DOMINANCE_WEAK, or DOMINANCE_VERY_WEAK\n    tol: tolerance\n\n  Returns:\n    A tuple (`reduced_game`, `live_actions`).\n    * if `game_or_payoffs` is an instance of `pyspiel.MatrixGame`, so is\n      `reduced_game`; otherwise `reduced_game` is a payoff tensor.\n    * `live_actions` is a tuple of length `num_players`, where\n      `live_actions[player]` is a boolean vector of shape `num_actions`;\n       `live_actions[player][action]` is `True` if `action` wasn't dominated for\n       `player`.\n  "
    payoffs = utils.game_payoffs_array(game_or_payoffs) if isinstance(game_or_payoffs, pyspiel.NormalFormGame) else np.asfarray(game_or_payoffs)
    live_actions = [np.ones(num_actions, bool) for num_actions in payoffs.shape[1:]]
    progress = True
    while progress:
        progress = False
        for method in ('pure', 'mixed'):
            if progress:
                continue
            for (player, live) in enumerate(live_actions):
                if live.sum() == 1:
                    continue
                payoffs_live = payoffs[player]
                for opponent in range(payoffs.shape[0]):
                    if opponent != player:
                        payoffs_live = payoffs_live.compress(live_actions[opponent], opponent)
                payoffs_live = np.moveaxis(payoffs_live, player, 0)
                payoffs_live = payoffs_live.reshape((payoffs_live.shape[0], -1))
                for action in range(live.size):
                    if not live[action]:
                        continue
                    if method == 'pure':
                        advantage = payoffs_live[action] - payoffs_live
                        dominated = _pure_dominated_from_advantages(advantage, mode, tol)
                        dominated[action] = False
                        dominated &= live
                        if dominated.any():
                            progress = True
                            live &= ~dominated
                            if live.sum() == 1:
                                break
                    if method == 'mixed':
                        mixture = is_dominated(live[:action].sum(), payoffs_live[live], 0, mode, tol, return_mixture=True)
                        if mixture is None:
                            continue
                        progress = True
                        advantage = mixture.dot(payoffs_live[live]) - payoffs_live[live]
                        dominated = _pure_dominated_from_advantages(advantage, mode, tol)
                        dominated[mixture > tol] = False
                        assert dominated[live[:action].sum()]
                        live.put(live.nonzero()[0], ~dominated)
                        if live.sum() == 1:
                            break
    for (player, live) in enumerate(live_actions):
        payoffs = payoffs.compress(live, player + 1)
    if isinstance(game_or_payoffs, pyspiel.MatrixGame):
        return (pyspiel.MatrixGame(game_or_payoffs.get_type(), game_or_payoffs.get_parameters(), [game_or_payoffs.row_action_name(action) for action in live_actions[0].nonzero()[0]], [game_or_payoffs.col_action_name(action) for action in live_actions[1].nonzero()[0]], *payoffs), live_actions)
    else:
        return (payoffs, live_actions)