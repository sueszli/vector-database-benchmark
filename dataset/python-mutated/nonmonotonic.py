"""
A module to perform nonmonotonic reasoning.  The ideas and demonstrations in
this module are based on "Logical Foundations of Artificial Intelligence" by
Michael R. Genesereth and Nils J. Nilsson.
"""
from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import AbstractVariableExpression, AllExpression, AndExpression, ApplicationExpression, BooleanExpression, EqualityExpression, ExistsExpression, Expression, ImpExpression, NegatedExpression, Variable, VariableExpression, operator, unique_variable

class ProverParseError(Exception):
    pass

def get_domain(goal, assumptions):
    if False:
        i = 10
        return i + 15
    if goal is None:
        all_expressions = assumptions
    else:
        all_expressions = assumptions + [-goal]
    return reduce(operator.or_, (a.constants() for a in all_expressions), set())

class ClosedDomainProver(ProverCommandDecorator):
    """
    This is a prover decorator that adds domain closure assumptions before
    proving.
    """

    def assumptions(self):
        if False:
            return 10
        assumptions = [a for a in self._command.assumptions()]
        goal = self._command.goal()
        domain = get_domain(goal, assumptions)
        return [self.replace_quants(ex, domain) for ex in assumptions]

    def goal(self):
        if False:
            return 10
        goal = self._command.goal()
        domain = get_domain(goal, self._command.assumptions())
        return self.replace_quants(goal, domain)

    def replace_quants(self, ex, domain):
        if False:
            return 10
        '\n        Apply the closed domain assumption to the expression\n\n        - Domain = union([e.free()|e.constants() for e in all_expressions])\n        - translate "exists x.P" to "(z=d1 | z=d2 | ... ) & P.replace(x,z)" OR\n                    "P.replace(x, d1) | P.replace(x, d2) | ..."\n        - translate "all x.P" to "P.replace(x, d1) & P.replace(x, d2) & ..."\n\n        :param ex: ``Expression``\n        :param domain: set of {Variable}s\n        :return: ``Expression``\n        '
        if isinstance(ex, AllExpression):
            conjuncts = [ex.term.replace(ex.variable, VariableExpression(d)) for d in domain]
            conjuncts = [self.replace_quants(c, domain) for c in conjuncts]
            return reduce(lambda x, y: x & y, conjuncts)
        elif isinstance(ex, BooleanExpression):
            return ex.__class__(self.replace_quants(ex.first, domain), self.replace_quants(ex.second, domain))
        elif isinstance(ex, NegatedExpression):
            return -self.replace_quants(ex.term, domain)
        elif isinstance(ex, ExistsExpression):
            disjuncts = [ex.term.replace(ex.variable, VariableExpression(d)) for d in domain]
            disjuncts = [self.replace_quants(d, domain) for d in disjuncts]
            return reduce(lambda x, y: x | y, disjuncts)
        else:
            return ex

class UniqueNamesProver(ProverCommandDecorator):
    """
    This is a prover decorator that adds unique names assumptions before
    proving.
    """

    def assumptions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        - Domain = union([e.free()|e.constants() for e in all_expressions])\n        - if "d1 = d2" cannot be proven from the premises, then add "d1 != d2"\n        '
        assumptions = self._command.assumptions()
        domain = list(get_domain(self._command.goal(), assumptions))
        eq_sets = SetHolder()
        for a in assumptions:
            if isinstance(a, EqualityExpression):
                av = a.first.variable
                bv = a.second.variable
                eq_sets[av].add(bv)
        new_assumptions = []
        for (i, a) in enumerate(domain):
            for b in domain[i + 1:]:
                if b not in eq_sets[a]:
                    newEqEx = EqualityExpression(VariableExpression(a), VariableExpression(b))
                    if Prover9().prove(newEqEx, assumptions):
                        eq_sets[a].add(b)
                    else:
                        new_assumptions.append(-newEqEx)
        return assumptions + new_assumptions

class SetHolder(list):
    """
    A list of sets of Variables.
    """

    def __getitem__(self, item):
        if False:
            return 10
        "\n        :param item: ``Variable``\n        :return: the set containing 'item'\n        "
        assert isinstance(item, Variable)
        for s in self:
            if item in s:
                return s
        new = {item}
        self.append(new)
        return new

class ClosedWorldProver(ProverCommandDecorator):
    """
    This is a prover decorator that completes predicates before proving.

    If the assumptions contain "P(A)", then "all x.(P(x) -> (x=A))" is the completion of "P".
    If the assumptions contain "all x.(ostrich(x) -> bird(x))", then "all x.(bird(x) -> ostrich(x))" is the completion of "bird".
    If the assumptions don't contain anything that are "P", then "all x.-P(x)" is the completion of "P".

    walk(Socrates)
    Socrates != Bill
    + all x.(walk(x) -> (x=Socrates))
    ----------------
    -walk(Bill)

    see(Socrates, John)
    see(John, Mary)
    Socrates != John
    John != Mary
    + all x.all y.(see(x,y) -> ((x=Socrates & y=John) | (x=John & y=Mary)))
    ----------------
    -see(Socrates, Mary)

    all x.(ostrich(x) -> bird(x))
    bird(Tweety)
    -ostrich(Sam)
    Sam != Tweety
    + all x.(bird(x) -> (ostrich(x) | x=Tweety))
    + all x.-ostrich(x)
    -------------------
    -bird(Sam)
    """

    def assumptions(self):
        if False:
            return 10
        assumptions = self._command.assumptions()
        predicates = self._make_predicate_dict(assumptions)
        new_assumptions = []
        for p in predicates:
            predHolder = predicates[p]
            new_sig = self._make_unique_signature(predHolder)
            new_sig_exs = [VariableExpression(v) for v in new_sig]
            disjuncts = []
            for sig in predHolder.signatures:
                equality_exs = []
                for (v1, v2) in zip(new_sig_exs, sig):
                    equality_exs.append(EqualityExpression(v1, v2))
                disjuncts.append(reduce(lambda x, y: x & y, equality_exs))
            for prop in predHolder.properties:
                bindings = {}
                for (v1, v2) in zip(new_sig_exs, prop[0]):
                    bindings[v2] = v1
                disjuncts.append(prop[1].substitute_bindings(bindings))
            if disjuncts:
                antecedent = self._make_antecedent(p, new_sig)
                consequent = reduce(lambda x, y: x | y, disjuncts)
                accum = ImpExpression(antecedent, consequent)
            else:
                accum = NegatedExpression(self._make_antecedent(p, new_sig))
            for new_sig_var in new_sig[::-1]:
                accum = AllExpression(new_sig_var, accum)
            new_assumptions.append(accum)
        return assumptions + new_assumptions

    def _make_unique_signature(self, predHolder):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method figures out how many arguments the predicate takes and\n        returns a tuple containing that number of unique variables.\n        '
        return tuple((unique_variable() for i in range(predHolder.signature_len)))

    def _make_antecedent(self, predicate, signature):
        if False:
            return 10
        "\n        Return an application expression with 'predicate' as the predicate\n        and 'signature' as the list of arguments.\n        "
        antecedent = predicate
        for v in signature:
            antecedent = antecedent(VariableExpression(v))
        return antecedent

    def _make_predicate_dict(self, assumptions):
        if False:
            i = 10
            return i + 15
        '\n        Create a dictionary of predicates from the assumptions.\n\n        :param assumptions: a list of ``Expression``s\n        :return: dict mapping ``AbstractVariableExpression`` to ``PredHolder``\n        '
        predicates = defaultdict(PredHolder)
        for a in assumptions:
            self._map_predicates(a, predicates)
        return predicates

    def _map_predicates(self, expression, predDict):
        if False:
            print('Hello World!')
        if isinstance(expression, ApplicationExpression):
            (func, args) = expression.uncurry()
            if isinstance(func, AbstractVariableExpression):
                predDict[func].append_sig(tuple(args))
        elif isinstance(expression, AndExpression):
            self._map_predicates(expression.first, predDict)
            self._map_predicates(expression.second, predDict)
        elif isinstance(expression, AllExpression):
            sig = [expression.variable]
            term = expression.term
            while isinstance(term, AllExpression):
                sig.append(term.variable)
                term = term.term
            if isinstance(term, ImpExpression):
                if isinstance(term.first, ApplicationExpression) and isinstance(term.second, ApplicationExpression):
                    (func1, args1) = term.first.uncurry()
                    (func2, args2) = term.second.uncurry()
                    if isinstance(func1, AbstractVariableExpression) and isinstance(func2, AbstractVariableExpression) and (sig == [v.variable for v in args1]) and (sig == [v.variable for v in args2]):
                        predDict[func2].append_prop((tuple(sig), term.first))
                        predDict[func1].validate_sig_len(sig)

class PredHolder:
    """
    This class will be used by a dictionary that will store information
    about predicates to be used by the ``ClosedWorldProver``.

    The 'signatures' property is a list of tuples defining signatures for
    which the predicate is true.  For instance, 'see(john, mary)' would be
    result in the signature '(john,mary)' for 'see'.

    The second element of the pair is a list of pairs such that the first
    element of the pair is a tuple of variables and the second element is an
    expression of those variables that makes the predicate true.  For instance,
    'all x.all y.(see(x,y) -> know(x,y))' would result in "((x,y),('see(x,y)'))"
    for 'know'.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.signatures = []
        self.properties = []
        self.signature_len = None

    def append_sig(self, new_sig):
        if False:
            while True:
                i = 10
        self.validate_sig_len(new_sig)
        self.signatures.append(new_sig)

    def append_prop(self, new_prop):
        if False:
            return 10
        self.validate_sig_len(new_prop[0])
        self.properties.append(new_prop)

    def validate_sig_len(self, new_sig):
        if False:
            while True:
                i = 10
        if self.signature_len is None:
            self.signature_len = len(new_sig)
        elif self.signature_len != len(new_sig):
            raise Exception('Signature lengths do not match')

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'({self.signatures},{self.properties},{self.signature_len})'

    def __repr__(self):
        if False:
            return 10
        return '%s' % self

def closed_domain_demo():
    if False:
        print('Hello World!')
    lexpr = Expression.fromstring
    p1 = lexpr('exists x.walk(x)')
    p2 = lexpr('man(Socrates)')
    c = lexpr('walk(Socrates)')
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print('assumptions:')
    for a in cdp.assumptions():
        print('   ', a)
    print('goal:', cdp.goal())
    print(cdp.prove())
    p1 = lexpr('exists x.walk(x)')
    p2 = lexpr('man(Socrates)')
    p3 = lexpr('-walk(Bill)')
    c = lexpr('walk(Socrates)')
    prover = Prover9Command(c, [p1, p2, p3])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print('assumptions:')
    for a in cdp.assumptions():
        print('   ', a)
    print('goal:', cdp.goal())
    print(cdp.prove())
    p1 = lexpr('exists x.walk(x)')
    p2 = lexpr('man(Socrates)')
    p3 = lexpr('-walk(Bill)')
    c = lexpr('walk(Socrates)')
    prover = Prover9Command(c, [p1, p2, p3])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print('assumptions:')
    for a in cdp.assumptions():
        print('   ', a)
    print('goal:', cdp.goal())
    print(cdp.prove())
    p1 = lexpr('walk(Socrates)')
    p2 = lexpr('walk(Bill)')
    c = lexpr('all x.walk(x)')
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print('assumptions:')
    for a in cdp.assumptions():
        print('   ', a)
    print('goal:', cdp.goal())
    print(cdp.prove())
    p1 = lexpr('girl(mary)')
    p2 = lexpr('dog(rover)')
    p3 = lexpr('all x.(girl(x) -> -dog(x))')
    p4 = lexpr('all x.(dog(x) -> -girl(x))')
    p5 = lexpr('chase(mary, rover)')
    c = lexpr('exists y.(dog(y) & all x.(girl(x) -> chase(x,y)))')
    prover = Prover9Command(c, [p1, p2, p3, p4, p5])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print('assumptions:')
    for a in cdp.assumptions():
        print('   ', a)
    print('goal:', cdp.goal())
    print(cdp.prove())

def unique_names_demo():
    if False:
        print('Hello World!')
    lexpr = Expression.fromstring
    p1 = lexpr('man(Socrates)')
    p2 = lexpr('man(Bill)')
    c = lexpr('exists x.exists y.(x != y)')
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    unp = UniqueNamesProver(prover)
    print('assumptions:')
    for a in unp.assumptions():
        print('   ', a)
    print('goal:', unp.goal())
    print(unp.prove())
    p1 = lexpr('all x.(walk(x) -> (x = Socrates))')
    p2 = lexpr('Bill = William')
    p3 = lexpr('Bill = Billy')
    c = lexpr('-walk(William)')
    prover = Prover9Command(c, [p1, p2, p3])
    print(prover.prove())
    unp = UniqueNamesProver(prover)
    print('assumptions:')
    for a in unp.assumptions():
        print('   ', a)
    print('goal:', unp.goal())
    print(unp.prove())

def closed_world_demo():
    if False:
        print('Hello World!')
    lexpr = Expression.fromstring
    p1 = lexpr('walk(Socrates)')
    p2 = lexpr('(Socrates != Bill)')
    c = lexpr('-walk(Bill)')
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    cwp = ClosedWorldProver(prover)
    print('assumptions:')
    for a in cwp.assumptions():
        print('   ', a)
    print('goal:', cwp.goal())
    print(cwp.prove())
    p1 = lexpr('see(Socrates, John)')
    p2 = lexpr('see(John, Mary)')
    p3 = lexpr('(Socrates != John)')
    p4 = lexpr('(John != Mary)')
    c = lexpr('-see(Socrates, Mary)')
    prover = Prover9Command(c, [p1, p2, p3, p4])
    print(prover.prove())
    cwp = ClosedWorldProver(prover)
    print('assumptions:')
    for a in cwp.assumptions():
        print('   ', a)
    print('goal:', cwp.goal())
    print(cwp.prove())
    p1 = lexpr('all x.(ostrich(x) -> bird(x))')
    p2 = lexpr('bird(Tweety)')
    p3 = lexpr('-ostrich(Sam)')
    p4 = lexpr('Sam != Tweety')
    c = lexpr('-bird(Sam)')
    prover = Prover9Command(c, [p1, p2, p3, p4])
    print(prover.prove())
    cwp = ClosedWorldProver(prover)
    print('assumptions:')
    for a in cwp.assumptions():
        print('   ', a)
    print('goal:', cwp.goal())
    print(cwp.prove())

def combination_prover_demo():
    if False:
        print('Hello World!')
    lexpr = Expression.fromstring
    p1 = lexpr('see(Socrates, John)')
    p2 = lexpr('see(John, Mary)')
    c = lexpr('-see(Socrates, Mary)')
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    command = ClosedDomainProver(UniqueNamesProver(ClosedWorldProver(prover)))
    for a in command.assumptions():
        print(a)
    print(command.prove())

def default_reasoning_demo():
    if False:
        while True:
            i = 10
    lexpr = Expression.fromstring
    premises = []
    premises.append(lexpr('all x.(elephant(x)        -> animal(x))'))
    premises.append(lexpr('all x.(bird(x)            -> animal(x))'))
    premises.append(lexpr('all x.(dove(x)            -> bird(x))'))
    premises.append(lexpr('all x.(ostrich(x)         -> bird(x))'))
    premises.append(lexpr('all x.(flying_ostrich(x)  -> ostrich(x))'))
    premises.append(lexpr('all x.((animal(x)  & -Ab1(x)) -> -fly(x))'))
    premises.append(lexpr('all x.((bird(x)    & -Ab2(x)) -> fly(x))'))
    premises.append(lexpr('all x.((ostrich(x) & -Ab3(x)) -> -fly(x))'))
    premises.append(lexpr('all x.(bird(x)           -> Ab1(x))'))
    premises.append(lexpr('all x.(ostrich(x)        -> Ab2(x))'))
    premises.append(lexpr('all x.(flying_ostrich(x) -> Ab3(x))'))
    premises.append(lexpr('elephant(E)'))
    premises.append(lexpr('dove(D)'))
    premises.append(lexpr('ostrich(O)'))
    prover = Prover9Command(None, premises)
    command = UniqueNamesProver(ClosedWorldProver(prover))
    for a in command.assumptions():
        print(a)
    print_proof('-fly(E)', premises)
    print_proof('fly(D)', premises)
    print_proof('-fly(O)', premises)

def print_proof(goal, premises):
    if False:
        return 10
    lexpr = Expression.fromstring
    prover = Prover9Command(lexpr(goal), premises)
    command = UniqueNamesProver(ClosedWorldProver(prover))
    print(goal, prover.prove(), command.prove())

def demo():
    if False:
        for i in range(10):
            print('nop')
    closed_domain_demo()
    unique_names_demo()
    closed_world_demo()
    combination_prover_demo()
    default_reasoning_demo()
if __name__ == '__main__':
    demo()