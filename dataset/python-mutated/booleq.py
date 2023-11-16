"""Data structures and algorithms for boolean equations."""
import collections
import itertools
from pytype.pytd import pytd_utils
chain = itertools.chain.from_iterable

class BooleanTerm:
    """Base class for boolean terms."""
    __slots__ = ()

    def simplify(self, assignments):
        if False:
            return 10
        'Simplify this term, given a list of possible values for each variable.\n\n    Args:\n      assignments: A list of possible values for each variable. A dictionary\n        mapping strings (variable name) to sets of strings (value names).\n\n    Returns:\n      A new BooleanTerm, potentially simplified.\n    '
        raise NotImplementedError()

    def extract_pivots(self, assignments):
        if False:
            return 10
        'Find values for every variable that appears in this term.\n\n    This finds all variables that appear in this term and limits them to the\n    values they appear together with. For example, consider the equation\n      t = v1 | (t = v2 & (t = v2 | t = v3))\n    Here, t can be limited to [v1, v2]. (v3 is impossible.)\n\n    Args:\n      assignments: The current "upper bound", i.e. all values that are still\n        possible for variables. Used for extracting pivots out of Eq(var, var).\n\n    Returns:\n      A dictionary mapping strings (variable names) to sets of strings (value\n      or variable names).\n    '
        raise NotImplementedError()

    def extract_equalities(self):
        if False:
            while True:
                i = 10
        'Find all equalities that appear in this term.\n\n    Returns:\n      A sequence of tuples of a string (variable name) and a string (value or\n      variable name).\n    '
        raise NotImplementedError()

class TrueValue(BooleanTerm):
    """Class for representing "TRUE"."""

    def simplify(self, assignments):
        if False:
            return 10
        return self

    def __repr__(self):
        if False:
            return 10
        return 'TRUE'

    def __str__(self):
        if False:
            return 10
        return 'TRUE'

    def extract_pivots(self, assignments):
        if False:
            print('Hello World!')
        return {}

    def extract_equalities(self):
        if False:
            i = 10
            return i + 15
        return ()

class FalseValue(BooleanTerm):
    """Class for representing "FALSE"."""

    def simplify(self, assignments):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'FALSE'

    def __str__(self):
        if False:
            print('Hello World!')
        return 'FALSE'

    def extract_pivots(self, assignments):
        if False:
            i = 10
            return i + 15
        return {}

    def extract_equalities(self):
        if False:
            while True:
                i = 10
        return ()
TRUE = TrueValue()
FALSE = FalseValue()

def simplify_exprs(exprs, result_type, stop_term, skip_term):
    if False:
        i = 10
        return i + 15
    'Simplify a set of subexpressions for a conjunction or disjunction.\n\n  Args:\n    exprs: An iterable. The subexpressions.\n    result_type: _And or _Or. The type of result (unless it simplifies\n      down to something simpler).\n    stop_term: FALSE for _And, TRUE for _Or. If this term is encountered,\n      it will be immediately returned.\n    skip_term: TRUE for _And, FALSE for _Or. If this term is encountered,\n      it will be ignored.\n\n  Returns:\n    A BooleanTerm.\n  '
    expr_set = set()
    for e in exprs:
        if e is stop_term:
            return stop_term
        elif e is skip_term:
            continue
        elif isinstance(e, result_type):
            expr_set = expr_set.union(e.exprs)
        else:
            expr_set.add(e)
    if len(expr_set) > 1:
        return result_type(expr_set)
    elif expr_set:
        return expr_set.pop()
    else:
        return skip_term

class _Eq(BooleanTerm):
    """An equality constraint.

  This declares an equality between a variable and a value, or a variable
  and a variable. External code should use Eq rather than creating an _Eq
  instance directly.

  Attributes:
    left: A string; left side of the equality. This is expected to be the
      string with the higher ascii value, so e.g. strings starting with "~"
      (ascii 0x7e) should be on the left.
    right: A string; right side of the equality. This is the lower ascii value.
  """
    __slots__ = ('left', 'right')

    def __init__(self, left, right):
        if False:
            return 10
        'Initialize an equality.\n\n    Args:\n      left: A string. Left side of the equality.\n      right: A string. Right side of the equality.\n    '
        self.left = left
        self.right = right

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'Eq({self.left!r}, {self.right!r})'

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'{self.left} == {self.right}'

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.left, self.right))

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__class__ == other.__class__ and self.left == other.left and (self.right == other.right)

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

    def simplify(self, assignments):
        if False:
            return 10
        "Simplify this equality.\n\n    This will try to look up the values, and return FALSE if they're no longer\n    possible. Also, when comparing two variables, it will compute the\n    intersection, and return a disjunction of variable=value equalities instead.\n\n    Args:\n      assignments: Variable assignments (dict mapping strings to sets of\n      strings). Used to determine whether this equality is still possible, and\n      to compute intersections between two variables.\n\n    Returns:\n      A new BooleanTerm.\n    "
        if self.right in assignments:
            return self
        else:
            return self if self.right in assignments[self.left] else FALSE

    def extract_pivots(self, assignments):
        if False:
            return 10
        'Extract the pivots. See BooleanTerm.extract_pivots().'
        if self.left in assignments and self.right in assignments:
            intersection = assignments[self.left] & assignments[self.right]
            return {self.left: frozenset(intersection), self.right: frozenset(intersection)}
        else:
            return {self.left: frozenset((self.right,)), self.right: frozenset((self.left,))}

    def extract_equalities(self):
        if False:
            print('Hello World!')
        return ((self.left, self.right),)

def _expr_set_hash(expr_set):
    if False:
        print('Hello World!')
    return hash(tuple(sorted((hash(e) for e in expr_set))))

class _And(BooleanTerm):
    """A conjunction of equalities and disjunctions.

  External code should use And rather than creating an _And instance directly.
  """
    __slots__ = ('exprs',)

    def __init__(self, exprs):
        if False:
            while True:
                i = 10
        'Initialize a conjunction.\n\n    Args:\n      exprs: A set. The subterms.\n    '
        self.exprs = exprs

    def __eq__(self, other):
        if False:
            return 10
        return self.__class__ == other.__class__ and self.exprs == other.exprs

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

    def __repr__(self):
        if False:
            return 10
        return f'And({list(self.exprs)!r})'

    def __str__(self):
        if False:
            print('Hello World!')
        return '(' + ' & '.join((str(t) for t in self.exprs)) + ')'

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return _expr_set_hash(self.exprs)

    def simplify(self, assignments):
        if False:
            return 10
        return simplify_exprs((e.simplify(assignments) for e in self.exprs), _And, FALSE, TRUE)

    def extract_pivots(self, assignments):
        if False:
            i = 10
            return i + 15
        'Extract the pivots. See BooleanTerm.extract_pivots().'
        pivots = {}
        for expr in self.exprs:
            expr_pivots = expr.extract_pivots(assignments)
            for (name, values) in expr_pivots.items():
                if name in pivots:
                    pivots[name] = pivots[name] & values
                else:
                    pivots[name] = values
        return {var: values for (var, values) in pivots.items() if values}

    def extract_equalities(self):
        if False:
            return 10
        return tuple(chain((expr.extract_equalities() for expr in self.exprs)))

class _Or(BooleanTerm):
    """A disjunction of equalities and conjunctions.

  External code should use Or rather than creating an _Or instance directly.
  """
    __slots__ = ('exprs',)

    def __init__(self, exprs):
        if False:
            print('Hello World!')
        'Initialize a disjunction.\n\n    Args:\n      exprs: A set. The subterms.\n    '
        self.exprs = exprs

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__ == other.__class__ and self.exprs == other.exprs

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'Or({list(self.exprs)!r})'

    def __str__(self):
        if False:
            return 10
        return '(' + ' | '.join((str(t) for t in self.exprs)) + ')'

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return _expr_set_hash(self.exprs)

    def simplify(self, assignments):
        if False:
            i = 10
            return i + 15
        return simplify_exprs((e.simplify(assignments) for e in self.exprs), _Or, TRUE, FALSE)

    def extract_pivots(self, assignments):
        if False:
            return 10
        'Extract the pivots. See BooleanTerm.extract_pivots().'
        pivots = {}
        for expr in self.exprs:
            expr_pivots = expr.extract_pivots(assignments)
            for (name, values) in expr_pivots.items():
                if name in pivots:
                    pivots[name] = pivots[name] | values
                else:
                    pivots[name] = values
        return pivots

    def extract_equalities(self):
        if False:
            print('Hello World!')
        return tuple(chain((expr.extract_equalities() for expr in self.exprs)))

def Eq(left: str, right: str) -> BooleanTerm:
    if False:
        return 10
    "Create an equality or its simplified equivalent.\n\n  This will ensure that left > right. (For left == right, it'll just return\n  TRUE).\n\n  Args:\n    left: A string. Left side of the equality. This will get sorted, so it\n      might end up on the right.\n    right: A string. Right side of the equality. This will get sorted, so it\n      might end up on the left.\n\n  Returns:\n    A BooleanTerm.\n  "
    if left == right:
        return TRUE
    elif left > right:
        return _Eq(left, right)
    else:
        return _Eq(right, left)

def And(exprs):
    if False:
        while True:
            i = 10
    'Create a conjunction or its simplified equivalent.\n\n  This will ensure that, when an _And is returned, none of its immediate\n  subterms is TRUE, FALSE, or another conjunction.\n\n  Args:\n    exprs: An iterable. The subterms.\n\n  Returns:\n    A BooleanTerm.\n  '
    return simplify_exprs(exprs, _And, FALSE, TRUE)

def Or(exprs):
    if False:
        while True:
            i = 10
    'Create a disjunction or its simplified equivalent.\n\n  This will ensure that, when an _Or is returned, none of its immediate\n  subterms is TRUE, FALSE, or another disjunction.\n\n  Args:\n    exprs: An iterable. The subterms.\n\n  Returns:\n    A BooleanTerm.\n  '
    return simplify_exprs(exprs, _Or, TRUE, FALSE)

class Solver:
    """Solver for boolean equations.

  This solver computes the union of all solutions. I.e. rather than assigning
  exactly one value to each variable, it will create a list of values for each
  variable: All the values this variable has in any of the solutions.

  To accomplish this, we use the following rewriting rules:
    [1]  (t in X && ...) || (t in Y && ...) -->  t in (X | Y)
    [2]  t in X && t in Y                   -->  t in (X & Y)
  Applying these iteratively for each variable in turn ("extracting pivots")
  reduces the system to one where we can "read off" the possible values for each
  variable.

  Attributes:
    ANY_VALUE: A special value assigned to variables with no constraints.
    variables: A list of all variables.
    implications: A nested dictionary mapping variable names to values to
      BooleanTerm instances. This is used to specify rules like "if x is 1,
      then ..."
    ground_truth: An equation that needs to always be TRUE. If this is FALSE,
      or can be reduced to FALSE, the system is unsolvable.
    assignments: The solutions, a mapping of variables to values.
  """
    ANY_VALUE = '?'

    def __init__(self):
        if False:
            print('Hello World!')
        self.variables = set()
        self.implications = collections.defaultdict(dict)
        self.ground_truth = TRUE
        self.assignments = None

    def __str__(self):
        if False:
            while True:
                i = 10
        lines = []
        (count_false, count_true) = (0, 0)
        if self.ground_truth is not TRUE:
            lines.append(f'always: {self.ground_truth}')
        for (var, value, implication) in self._iter_implications():
            if implication is FALSE:
                count_false += 1
            elif implication is TRUE:
                count_true += 1
            else:
                lines.append(f'if {_Eq(var, value)} then {implication}')
        return '%s\n(not shown: %d always FALSE, %d always TRUE)\n' % ('\n'.join(lines), count_false, count_true)

    def __repr__(self):
        if False:
            print('Hello World!')
        lines = []
        for var in self.variables:
            lines.append(f'solver.register_variable({var!r})')
        if self.ground_truth is not TRUE:
            lines.append(f'solver.always_true({self.ground_truth!r})')
        for (var, value, implication) in self._iter_implications():
            lines.append(f'solver.implies({_Eq(var, value)!r}, {implication!r})')
        return '\n' + ''.join((line + '\n' for line in lines))

    def register_variable(self, variable):
        if False:
            print('Hello World!')
        'Register a variable. Call before calling solve().'
        self.variables.add(variable)

    def always_true(self, formula):
        if False:
            return 10
        'Register a ground truth. Call before calling solve().'
        assert formula is not FALSE
        self.ground_truth = And([self.ground_truth, formula])

    def implies(self, e: BooleanTerm, implication: BooleanTerm) -> None:
        if False:
            while True:
                i = 10
        'Register an implication. Call before calling solve().'
        if e is FALSE or e is TRUE:
            raise AssertionError('Illegal equation')
        assert isinstance(e, _Eq)
        assert e.right not in self.implications[e.left]
        self.implications[e.left][e.right] = implication

    def _iter_implications(self):
        if False:
            while True:
                i = 10
        for (var, value_to_implication) in self.implications.items():
            for (value, implication) in value_to_implication.items():
                yield (var, value, implication)

    def _get_nonfalse_values(self, var):
        if False:
            for i in range(10):
                print('nop')
        return {value for (value, implication) in self.implications[var].items() if implication is not FALSE}

    def _get_first_approximation(self):
        if False:
            while True:
                i = 10
        'Get all (variable, value) combinations to consider.\n\n    This gets the (variable, value) combinations that the solver needs to\n    consider based on the equalities that appear in the implications. E.g.,\n    with the following implication:\n      t1 = v1 => t1 = t2 | t3 = v2\n    the combinations to consider are\n      (t1, v1) because t1 = v1 appears,\n      (t2, v1) because t1 = t2 and t1 = v1 appear, and\n      (t3, v2) because t3 = v2 appears.\n\n    Returns:\n      A dictionary D mapping strings (variables) to sets of strings\n      (values). For two variables t1 and t2, if t1 = t2 is a possible\n      assignment (by first approximation), then D[t1] and D[t2] point\n      to the same memory location.\n    '
        equalities = set(chain((implication.extract_equalities() for (_, _, implication) in self._iter_implications()))).union(self.ground_truth.extract_equalities())
        var_assignments = {}
        value_assignments = {}
        for var in self.variables:
            var_assignments[var] = {var}
            value_assignments[var] = self._get_nonfalse_values(var)
        for (var, value) in equalities:
            if value in self.variables:
                other_var = value
                value_assignments[var] |= value_assignments[other_var]
                for var_assignment in var_assignments[other_var]:
                    var_assignments[var].add(var_assignment)
                    var_assignments[var_assignment] = var_assignments[var]
                    value_assignments[var_assignment] = value_assignments[var]
            else:
                value_assignments[var].add(value)
        return value_assignments

    def _complete(self):
        if False:
            while True:
                i = 10
        'Insert missing implications.\n\n    Insert all implications needed to have one implication for every\n    (variable, value) combination returned by _get_first_approximation().\n    '
        for (var, values) in self._get_first_approximation().items():
            for value in values:
                if value not in self.implications[var]:
                    self.implications[var][value] = TRUE
            if not self.implications[var]:
                self.implications[var][Solver.ANY_VALUE] = TRUE

    def solve(self):
        if False:
            i = 10
            return i + 15
        'Solve the system of equations.\n\n    Returns:\n      An assignment, mapping strings (variables) to sets of strings (values).\n    '
        if self.assignments:
            return self.assignments
        self._complete()
        assignments = {var: self._get_nonfalse_values(var) for var in self.variables}
        ground_pivots = self.ground_truth.simplify(assignments).extract_pivots(assignments)
        for (pivot, possible_values) in ground_pivots.items():
            if pivot in assignments:
                assignments[pivot] &= set(possible_values)
        something_changed = True
        while something_changed:
            something_changed = False
            and_terms = []
            for var in self.variables:
                or_terms = []
                for value in assignments[var].copy():
                    implication = self.implications[var][value].simplify(assignments)
                    if implication is FALSE:
                        assignments[var].remove(value)
                        something_changed = True
                    else:
                        or_terms.append(implication)
                    self.implications[var][value] = implication
                and_terms.append(Or(or_terms))
            d = And(and_terms)
            for (pivot, possible_values) in d.extract_pivots(assignments).items():
                if pivot in assignments:
                    length_before = len(assignments[pivot])
                    assignments[pivot] &= set(possible_values)
                    length_after = len(assignments[pivot])
                    something_changed |= length_before != length_after
        self.register_variable = pytd_utils.disabled_function
        self.implies = pytd_utils.disabled_function
        self.assignments = assignments
        return assignments