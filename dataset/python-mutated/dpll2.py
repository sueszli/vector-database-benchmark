"""Implementation of DPLL algorithm

Features:
  - Clause learning
  - Watch literal scheme
  - VSIDS heuristic

References:
  - https://en.wikipedia.org/wiki/DPLL_algorithm
"""
from collections import defaultdict
from heapq import heappush, heappop
from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF
from sympy.logic.algorithms.lra_theory import LRASolver

def dpll_satisfiable(expr, all_models=False, use_lra_theory=False):
    if False:
        return 10
    '\n    Check satisfiability of a propositional sentence.\n    It returns a model rather than True when it succeeds.\n    Returns a generator of all models if all_models is True.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import A, B\n    >>> from sympy.logic.algorithms.dpll2 import dpll_satisfiable\n    >>> dpll_satisfiable(A & ~B)\n    {A: True, B: False}\n    >>> dpll_satisfiable(A & ~A)\n    False\n\n    '
    if not isinstance(expr, EncodedCNF):
        exprs = EncodedCNF()
        exprs.add_prop(expr)
        expr = exprs
    if {0} in expr.data:
        if all_models:
            return (f for f in [False])
        return False
    if use_lra_theory:
        (lra, immediate_conflicts) = LRASolver.from_encoded_cnf(expr)
    else:
        lra = None
        immediate_conflicts = []
    solver = SATSolver(expr.data + immediate_conflicts, expr.variables, set(), expr.symbols, lra_theory=lra)
    models = solver._find_model()
    if all_models:
        return _all_models(models)
    try:
        return next(models)
    except StopIteration:
        return False

def _all_models(models):
    if False:
        print('Hello World!')
    satisfiable = False
    try:
        while True:
            yield next(models)
            satisfiable = True
    except StopIteration:
        if not satisfiable:
            yield False

class SATSolver:
    """
    Class for representing a SAT solver capable of
     finding a model to a boolean theory in conjunctive
     normal form.
    """

    def __init__(self, clauses, variables, var_settings, symbols=None, heuristic='vsids', clause_learning='none', INTERVAL=500, lra_theory=None):
        if False:
            i = 10
            return i + 15
        self.var_settings = var_settings
        self.heuristic = heuristic
        self.is_unsatisfied = False
        self._unit_prop_queue = []
        self.update_functions = []
        self.INTERVAL = INTERVAL
        if symbols is None:
            self.symbols = list(ordered(variables))
        else:
            self.symbols = symbols
        self._initialize_variables(variables)
        self._initialize_clauses(clauses)
        if 'vsids' == heuristic:
            self._vsids_init()
            self.heur_calculate = self._vsids_calculate
            self.heur_lit_assigned = self._vsids_lit_assigned
            self.heur_lit_unset = self._vsids_lit_unset
            self.heur_clause_added = self._vsids_clause_added
        else:
            raise NotImplementedError
        if 'simple' == clause_learning:
            self.add_learned_clause = self._simple_add_learned_clause
            self.compute_conflict = self.simple_compute_conflict
            self.update_functions.append(self.simple_clean_clauses)
        elif 'none' == clause_learning:
            self.add_learned_clause = lambda x: None
            self.compute_conflict = lambda : None
        else:
            raise NotImplementedError
        self.levels = [Level(0)]
        self._current_level.varsettings = var_settings
        self.num_decisions = 0
        self.num_learned_clauses = 0
        self.original_num_clauses = len(self.clauses)
        self.lra = lra_theory

    def _initialize_variables(self, variables):
        if False:
            return 10
        'Set up the variable data structures needed.'
        self.sentinels = defaultdict(set)
        self.occurrence_count = defaultdict(int)
        self.variable_set = [False] * (len(variables) + 1)

    def _initialize_clauses(self, clauses):
        if False:
            while True:
                i = 10
        'Set up the clause data structures needed.\n\n        For each clause, the following changes are made:\n        - Unit clauses are queued for propagation right away.\n        - Non-unit clauses have their first and last literals set as sentinels.\n        - The number of clauses a literal appears in is computed.\n        '
        self.clauses = [list(clause) for clause in clauses]
        for (i, clause) in enumerate(self.clauses):
            if 1 == len(clause):
                self._unit_prop_queue.append(clause[0])
                continue
            self.sentinels[clause[0]].add(i)
            self.sentinels[clause[-1]].add(i)
            for lit in clause:
                self.occurrence_count[lit] += 1

    def _find_model(self):
        if False:
            return 10
        '\n        Main DPLL loop. Returns a generator of models.\n\n        Variables are chosen successively, and assigned to be either\n        True or False. If a solution is not found with this setting,\n        the opposite is chosen and the search continues. The solver\n        halts when every variable has a setting.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n        >>> list(l._find_model())\n        [{1: True, 2: False, 3: False}, {1: True, 2: True, 3: True}]\n\n        >>> from sympy.abc import A, B, C\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set(), [A, B, C])\n        >>> list(l._find_model())\n        [{A: True, B: False, C: False}, {A: True, B: True, C: True}]\n\n        '
        flip_var = False
        self._simplify()
        if self.is_unsatisfied:
            return
        while True:
            if self.num_decisions % self.INTERVAL == 0:
                for func in self.update_functions:
                    func()
            if flip_var:
                flip_var = False
                lit = self._current_level.decision
            else:
                lit = self.heur_calculate()
                self.num_decisions += 1
                if 0 == lit:
                    if self.lra:
                        for enc_var in self.var_settings:
                            res = self.lra.assert_lit(enc_var)
                            if res is not None:
                                break
                        res = self.lra.check()
                        self.lra.reset_bounds()
                    else:
                        res = None
                    if res is None or res[0]:
                        yield {self.symbols[abs(lit) - 1]: lit > 0 for lit in self.var_settings}
                    else:
                        self._simple_add_learned_clause(res[1])
                    while self._current_level.flipped:
                        self._undo()
                    if len(self.levels) == 1:
                        return
                    flip_lit = -self._current_level.decision
                    self._undo()
                    self.levels.append(Level(flip_lit, flipped=True))
                    flip_var = True
                    continue
                self.levels.append(Level(lit))
            self._assign_literal(lit)
            self._simplify()
            if self.is_unsatisfied:
                self.is_unsatisfied = False
                while self._current_level.flipped:
                    self._undo()
                    if 1 == len(self.levels):
                        return
                self.add_learned_clause(self.compute_conflict())
                flip_lit = -self._current_level.decision
                self._undo()
                self.levels.append(Level(flip_lit, flipped=True))
                flip_var = True

    @property
    def _current_level(self):
        if False:
            while True:
                i = 10
        'The current decision level data structure\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{1}, {2}], {1, 2}, set())\n        >>> next(l._find_model())\n        {1: True, 2: True}\n        >>> l._current_level.decision\n        0\n        >>> l._current_level.flipped\n        False\n        >>> l._current_level.var_settings\n        {1, 2}\n\n        '
        return self.levels[-1]

    def _clause_sat(self, cls):
        if False:
            i = 10
            return i + 15
        'Check if a clause is satisfied by the current variable setting.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{1}, {-1}], {1}, set())\n        >>> try:\n        ...     next(l._find_model())\n        ... except StopIteration:\n        ...     pass\n        >>> l._clause_sat(0)\n        False\n        >>> l._clause_sat(1)\n        True\n\n        '
        for lit in self.clauses[cls]:
            if lit in self.var_settings:
                return True
        return False

    def _is_sentinel(self, lit, cls):
        if False:
            return 10
        'Check if a literal is a sentinel of a given clause.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n        >>> next(l._find_model())\n        {1: True, 2: False, 3: False}\n        >>> l._is_sentinel(2, 3)\n        True\n        >>> l._is_sentinel(-3, 1)\n        False\n\n        '
        return cls in self.sentinels[lit]

    def _assign_literal(self, lit):
        if False:
            for i in range(10):
                print('nop')
        'Make a literal assignment.\n\n        The literal assignment must be recorded as part of the current\n        decision level. Additionally, if the literal is marked as a\n        sentinel of any clause, then a new sentinel must be chosen. If\n        this is not possible, then unit propagation is triggered and\n        another literal is added to the queue to be set in the future.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n        >>> next(l._find_model())\n        {1: True, 2: False, 3: False}\n        >>> l.var_settings\n        {-3, -2, 1}\n\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n        >>> l._assign_literal(-1)\n        >>> try:\n        ...     next(l._find_model())\n        ... except StopIteration:\n        ...     pass\n        >>> l.var_settings\n        {-1}\n\n        '
        self.var_settings.add(lit)
        self._current_level.var_settings.add(lit)
        self.variable_set[abs(lit)] = True
        self.heur_lit_assigned(lit)
        sentinel_list = list(self.sentinels[-lit])
        for cls in sentinel_list:
            if not self._clause_sat(cls):
                other_sentinel = None
                for newlit in self.clauses[cls]:
                    if newlit != -lit:
                        if self._is_sentinel(newlit, cls):
                            other_sentinel = newlit
                        elif not self.variable_set[abs(newlit)]:
                            self.sentinels[-lit].remove(cls)
                            self.sentinels[newlit].add(cls)
                            other_sentinel = None
                            break
                if other_sentinel:
                    self._unit_prop_queue.append(other_sentinel)

    def _undo(self):
        if False:
            return 10
        '\n        _undo the changes of the most recent decision level.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n        >>> next(l._find_model())\n        {1: True, 2: False, 3: False}\n        >>> level = l._current_level\n        >>> level.decision, level.var_settings, level.flipped\n        (-3, {-3, -2}, False)\n        >>> l._undo()\n        >>> level = l._current_level\n        >>> level.decision, level.var_settings, level.flipped\n        (0, {1}, False)\n\n        '
        for lit in self._current_level.var_settings:
            self.var_settings.remove(lit)
            self.heur_lit_unset(lit)
            self.variable_set[abs(lit)] = False
        self.levels.pop()
    '\n    Propagation methods should attempt to soundly simplify the boolean\n      theory, and return True if any simplification occurred and False\n      otherwise.\n    '

    def _simplify(self):
        if False:
            i = 10
            return i + 15
        'Iterate over the various forms of propagation to simplify the theory.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n        >>> l.variable_set\n        [False, False, False, False]\n        >>> l.sentinels\n        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4}}\n\n        >>> l._simplify()\n\n        >>> l.variable_set\n        [False, True, False, False]\n        >>> l.sentinels\n        {-3: {0, 2}, -2: {3, 4}, -1: set(), 2: {0, 3},\n        ...3: {2, 4}}\n\n        '
        changed = True
        while changed:
            changed = False
            changed |= self._unit_prop()
            changed |= self._pure_literal()

    def _unit_prop(self):
        if False:
            for i in range(10):
                print('nop')
        'Perform unit propagation on the current theory.'
        result = len(self._unit_prop_queue) > 0
        while self._unit_prop_queue:
            next_lit = self._unit_prop_queue.pop()
            if -next_lit in self.var_settings:
                self.is_unsatisfied = True
                self._unit_prop_queue = []
                return False
            else:
                self._assign_literal(next_lit)
        return result

    def _pure_literal(self):
        if False:
            return 10
        'Look for pure literals and assign them when found.'
        return False

    def _vsids_init(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the data structures needed for the VSIDS heuristic.'
        self.lit_heap = []
        self.lit_scores = {}
        for var in range(1, len(self.variable_set)):
            self.lit_scores[var] = float(-self.occurrence_count[var])
            self.lit_scores[-var] = float(-self.occurrence_count[-var])
            heappush(self.lit_heap, (self.lit_scores[var], var))
            heappush(self.lit_heap, (self.lit_scores[-var], -var))

    def _vsids_decay(self):
        if False:
            i = 10
            return i + 15
        'Decay the VSIDS scores for every literal.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n\n        >>> l.lit_scores\n        {-3: -2.0, -2: -2.0, -1: 0.0, 1: 0.0, 2: -2.0, 3: -2.0}\n\n        >>> l._vsids_decay()\n\n        >>> l.lit_scores\n        {-3: -1.0, -2: -1.0, -1: 0.0, 1: 0.0, 2: -1.0, 3: -1.0}\n\n        '
        for lit in self.lit_scores.keys():
            self.lit_scores[lit] /= 2.0

    def _vsids_calculate(self):
        if False:
            i = 10
            return i + 15
        '\n            VSIDS Heuristic Calculation\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n\n        >>> l.lit_heap\n        [(-2.0, -3), (-2.0, 2), (-2.0, -2), (0.0, 1), (-2.0, 3), (0.0, -1)]\n\n        >>> l._vsids_calculate()\n        -3\n\n        >>> l.lit_heap\n        [(-2.0, -2), (-2.0, 2), (0.0, -1), (0.0, 1), (-2.0, 3)]\n\n        '
        if len(self.lit_heap) == 0:
            return 0
        while self.variable_set[abs(self.lit_heap[0][1])]:
            heappop(self.lit_heap)
            if len(self.lit_heap) == 0:
                return 0
        return heappop(self.lit_heap)[1]

    def _vsids_lit_assigned(self, lit):
        if False:
            return 10
        'Handle the assignment of a literal for the VSIDS heuristic.'
        pass

    def _vsids_lit_unset(self, lit):
        if False:
            return 10
        'Handle the unsetting of a literal for the VSIDS heuristic.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n        >>> l.lit_heap\n        [(-2.0, -3), (-2.0, 2), (-2.0, -2), (0.0, 1), (-2.0, 3), (0.0, -1)]\n\n        >>> l._vsids_lit_unset(2)\n\n        >>> l.lit_heap\n        [(-2.0, -3), (-2.0, -2), (-2.0, -2), (-2.0, 2), (-2.0, 3), (0.0, -1),\n        ...(-2.0, 2), (0.0, 1)]\n\n        '
        var = abs(lit)
        heappush(self.lit_heap, (self.lit_scores[var], var))
        heappush(self.lit_heap, (self.lit_scores[-var], -var))

    def _vsids_clause_added(self, cls):
        if False:
            return 10
        'Handle the addition of a new clause for the VSIDS heuristic.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n\n        >>> l.num_learned_clauses\n        0\n        >>> l.lit_scores\n        {-3: -2.0, -2: -2.0, -1: 0.0, 1: 0.0, 2: -2.0, 3: -2.0}\n\n        >>> l._vsids_clause_added({2, -3})\n\n        >>> l.num_learned_clauses\n        1\n        >>> l.lit_scores\n        {-3: -1.0, -2: -2.0, -1: 0.0, 1: 0.0, 2: -1.0, 3: -2.0}\n\n        '
        self.num_learned_clauses += 1
        for lit in cls:
            self.lit_scores[lit] += 1

    def _simple_add_learned_clause(self, cls):
        if False:
            i = 10
            return i + 15
        'Add a new clause to the theory.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n\n        >>> l.num_learned_clauses\n        0\n        >>> l.clauses\n        [[2, -3], [1], [3, -3], [2, -2], [3, -2]]\n        >>> l.sentinels\n        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4}}\n\n        >>> l._simple_add_learned_clause([3])\n\n        >>> l.clauses\n        [[2, -3], [1], [3, -3], [2, -2], [3, -2], [3]]\n        >>> l.sentinels\n        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4, 5}}\n\n        '
        cls_num = len(self.clauses)
        self.clauses.append(cls)
        for lit in cls:
            self.occurrence_count[lit] += 1
        self.sentinels[cls[0]].add(cls_num)
        self.sentinels[cls[-1]].add(cls_num)
        self.heur_clause_added(cls)

    def _simple_compute_conflict(self):
        if False:
            i = 10
            return i + 15
        ' Build a clause representing the fact that at least one decision made\n        so far is wrong.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.algorithms.dpll2 import SATSolver\n        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},\n        ... {3, -2}], {1, 2, 3}, set())\n        >>> next(l._find_model())\n        {1: True, 2: False, 3: False}\n        >>> l._simple_compute_conflict()\n        [3]\n\n        '
        return [-level.decision for level in self.levels[1:]]

    def _simple_clean_clauses(self):
        if False:
            i = 10
            return i + 15
        'Clean up learned clauses.'
        pass

class Level:
    """
    Represents a single level in the DPLL algorithm, and contains
    enough information for a sound backtracking procedure.
    """

    def __init__(self, decision, flipped=False):
        if False:
            while True:
                i = 10
        self.decision = decision
        self.var_settings = set()
        self.flipped = flipped