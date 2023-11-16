"""
An implementation of the Hole Semantics model, following Blackburn and Bos,
Representation and Inference for Natural Language (CSLI, 2005).

The semantic representations are built by the grammar hole.fcfg.
This module contains driver code to read in sentences and parse them
according to a hole semantics grammar.

After parsing, the semantic representation is in the form of an underspecified
representation that is not easy to read.  We use a "plugging" algorithm to
convert that representation into first-order logic formulas.
"""
from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import AllExpression, AndExpression, ApplicationExpression, ExistsExpression, IffExpression, ImpExpression, LambdaExpression, NegatedExpression, OrExpression
from nltk.sem.skolemize import skolemize

class Constants:
    ALL = 'ALL'
    EXISTS = 'EXISTS'
    NOT = 'NOT'
    AND = 'AND'
    OR = 'OR'
    IMP = 'IMP'
    IFF = 'IFF'
    PRED = 'PRED'
    LEQ = 'LEQ'
    HOLE = 'HOLE'
    LABEL = 'LABEL'
    MAP = {ALL: lambda v, e: AllExpression(v.variable, e), EXISTS: lambda v, e: ExistsExpression(v.variable, e), NOT: NegatedExpression, AND: AndExpression, OR: OrExpression, IMP: ImpExpression, IFF: IffExpression, PRED: ApplicationExpression}

class HoleSemantics:
    """
    This class holds the broken-down components of a hole semantics, i.e. it
    extracts the holes, labels, logic formula fragments and constraints out of
    a big conjunction of such as produced by the hole semantics grammar.  It
    then provides some operations on the semantics dealing with holes, labels
    and finding legal ways to plug holes with labels.
    """

    def __init__(self, usr):
        if False:
            i = 10
            return i + 15
        "\n        Constructor.  `usr' is a ``sem.Expression`` representing an\n        Underspecified Representation Structure (USR).  A USR has the following\n        special predicates:\n        ALL(l,v,n),\n        EXISTS(l,v,n),\n        AND(l,n,n),\n        OR(l,n,n),\n        IMP(l,n,n),\n        IFF(l,n,n),\n        PRED(l,v,n,v[,v]*) where the brackets and star indicate zero or more repetitions,\n        LEQ(n,n),\n        HOLE(n),\n        LABEL(n)\n        where l is the label of the node described by the predicate, n is either\n        a label or a hole, and v is a variable.\n        "
        self.holes = set()
        self.labels = set()
        self.fragments = {}
        self.constraints = set()
        self._break_down(usr)
        self.top_most_labels = self._find_top_most_labels()
        self.top_hole = self._find_top_hole()

    def is_node(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return true if x is a node (label or hole) in this semantic\n        representation.\n        '
        return x in self.labels | self.holes

    def _break_down(self, usr):
        if False:
            i = 10
            return i + 15
        '\n        Extract holes, labels, formula fragments and constraints from the hole\n        semantics underspecified representation (USR).\n        '
        if isinstance(usr, AndExpression):
            self._break_down(usr.first)
            self._break_down(usr.second)
        elif isinstance(usr, ApplicationExpression):
            (func, args) = usr.uncurry()
            if func.variable.name == Constants.LEQ:
                self.constraints.add(Constraint(args[0], args[1]))
            elif func.variable.name == Constants.HOLE:
                self.holes.add(args[0])
            elif func.variable.name == Constants.LABEL:
                self.labels.add(args[0])
            else:
                label = args[0]
                assert label not in self.fragments
                self.fragments[label] = (func, args[1:])
        else:
            raise ValueError(usr.label())

    def _find_top_nodes(self, node_list):
        if False:
            while True:
                i = 10
        top_nodes = node_list.copy()
        for f in self.fragments.values():
            args = f[1]
            for arg in args:
                if arg in node_list:
                    top_nodes.discard(arg)
        return top_nodes

    def _find_top_most_labels(self):
        if False:
            print('Hello World!')
        '\n        Return the set of labels which are not referenced directly as part of\n        another formula fragment.  These will be the top-most labels for the\n        subtree that they are part of.\n        '
        return self._find_top_nodes(self.labels)

    def _find_top_hole(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the hole that will be the top of the formula tree.\n        '
        top_holes = self._find_top_nodes(self.holes)
        assert len(top_holes) == 1
        return top_holes.pop()

    def pluggings(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate and return all the legal pluggings (mappings of labels to\n        holes) of this semantics given the constraints.\n        '
        record = []
        self._plug_nodes([(self.top_hole, [])], self.top_most_labels, {}, record)
        return record

    def _plug_nodes(self, queue, potential_labels, plug_acc, record):
        if False:
            print('Hello World!')
        "\n        Plug the nodes in `queue' with the labels in `potential_labels'.\n\n        Each element of `queue' is a tuple of the node to plug and the list of\n        ancestor holes from the root of the graph to that node.\n\n        `potential_labels' is a set of the labels which are still available for\n        plugging.\n\n        `plug_acc' is the incomplete mapping of holes to labels made on the\n        current branch of the search tree so far.\n\n        `record' is a list of all the complete pluggings that we have found in\n        total so far.  It is the only parameter that is destructively updated.\n        "
        if queue != []:
            (node, ancestors) = queue[0]
            if node in self.holes:
                self._plug_hole(node, ancestors, queue[1:], potential_labels, plug_acc, record)
            else:
                assert node in self.labels
                args = self.fragments[node][1]
                head = [(a, ancestors) for a in args if self.is_node(a)]
                self._plug_nodes(head + queue[1:], potential_labels, plug_acc, record)
        else:
            raise Exception('queue empty')

    def _plug_hole(self, hole, ancestors0, queue, potential_labels0, plug_acc0, record):
        if False:
            for i in range(10):
                print('nop')
        '\n        Try all possible ways of plugging a single hole.\n        See _plug_nodes for the meanings of the parameters.\n        '
        assert hole not in ancestors0
        ancestors = [hole] + ancestors0
        for l in potential_labels0:
            if self._violates_constraints(l, ancestors):
                continue
            plug_acc = plug_acc0.copy()
            plug_acc[hole] = l
            potential_labels = potential_labels0.copy()
            potential_labels.remove(l)
            if len(potential_labels) == 0:
                self._sanity_check_plugging(plug_acc, self.top_hole, [])
                record.append(plug_acc)
            else:
                self._plug_nodes(queue + [(l, ancestors)], potential_labels, plug_acc, record)

    def _violates_constraints(self, label, ancestors):
        if False:
            i = 10
            return i + 15
        "\n        Return True if the `label' cannot be placed underneath the holes given\n        by the set `ancestors' because it would violate the constraints imposed\n        on it.\n        "
        for c in self.constraints:
            if c.lhs == label:
                if c.rhs not in ancestors:
                    return True
        return False

    def _sanity_check_plugging(self, plugging, node, ancestors):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure that a given plugging is legal.  We recursively go through\n        each node and make sure that no constraints are violated.\n        We also check that all holes have been filled.\n        '
        if node in self.holes:
            ancestors = [node] + ancestors
            label = plugging[node]
        else:
            label = node
        assert label in self.labels
        for c in self.constraints:
            if c.lhs == label:
                assert c.rhs in ancestors
        args = self.fragments[label][1]
        for arg in args:
            if self.is_node(arg):
                self._sanity_check_plugging(plugging, arg, [label] + ancestors)

    def formula_tree(self, plugging):
        if False:
            i = 10
            return i + 15
        '\n        Return the first-order logic formula tree for this underspecified\n        representation using the plugging given.\n        '
        return self._formula_tree(plugging, self.top_hole)

    def _formula_tree(self, plugging, node):
        if False:
            return 10
        if node in plugging:
            return self._formula_tree(plugging, plugging[node])
        elif node in self.fragments:
            (pred, args) = self.fragments[node]
            children = [self._formula_tree(plugging, arg) for arg in args]
            return reduce(Constants.MAP[pred.variable.name], children)
        else:
            return node

class Constraint:
    """
    This class represents a constraint of the form (L =< N),
    where L is a label and N is a node (a label or a hole).
    """

    def __init__(self, lhs, rhs):
        if False:
            print('Hello World!')
        self.lhs = lhs
        self.rhs = rhs

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self.__class__ == other.__class__:
            return self.lhs == other.lhs and self.rhs == other.rhs
        else:
            return False

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self == other

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(repr(self))

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'({self.lhs} < {self.rhs})'

def hole_readings(sentence, grammar_filename=None, verbose=False):
    if False:
        while True:
            i = 10
    if not grammar_filename:
        grammar_filename = 'grammars/sample_grammars/hole.fcfg'
    if verbose:
        print('Reading grammar file', grammar_filename)
    parser = load_parser(grammar_filename)
    tokens = sentence.split()
    trees = list(parser.parse(tokens))
    if verbose:
        print('Got %d different parses' % len(trees))
    all_readings = []
    for tree in trees:
        sem = tree.label()['SEM'].simplify()
        if verbose:
            print('Raw:       ', sem)
        while isinstance(sem, LambdaExpression):
            sem = sem.term
        skolemized = skolemize(sem)
        if verbose:
            print('Skolemized:', skolemized)
        hole_sem = HoleSemantics(skolemized)
        if verbose:
            print('Holes:       ', hole_sem.holes)
            print('Labels:      ', hole_sem.labels)
            print('Constraints: ', hole_sem.constraints)
            print('Top hole:    ', hole_sem.top_hole)
            print('Top labels:  ', hole_sem.top_most_labels)
            print('Fragments:')
            for (l, f) in hole_sem.fragments.items():
                print(f'\t{l}: {f}')
        pluggings = hole_sem.pluggings()
        readings = list(map(hole_sem.formula_tree, pluggings))
        if verbose:
            for (i, r) in enumerate(readings):
                print()
                print('%d. %s' % (i, r))
            print()
        all_readings.extend(readings)
    return all_readings
if __name__ == '__main__':
    for r in hole_readings('a dog barks'):
        print(r)
    print()
    for r in hole_readings('every girl chases a dog'):
        print(r)