"""This is rule-based deduction system for SymPy

The whole thing is split into two parts

 - rules compilation and preparation of tables
 - runtime inference

For rule-based inference engines, the classical work is RETE algorithm [1],
[2] Although we are not implementing it in full (or even significantly)
it's still worth a read to understand the underlying ideas.

In short, every rule in a system of rules is one of two forms:

 - atom                     -> ...      (alpha rule)
 - And(atom1, atom2, ...)   -> ...      (beta rule)


The major complexity is in efficient beta-rules processing and usually for an
expert system a lot of effort goes into code that operates on beta-rules.


Here we take minimalistic approach to get something usable first.

 - (preparation)    of alpha- and beta- networks, everything except
 - (runtime)        FactRules.deduce_all_facts

             _____________________________________
            ( Kirr: I've never thought that doing )
            ( logic stuff is that difficult...    )
             -------------------------------------
                    o   ^__^
                     o  (oo)\\_______
                        (__)\\       )\\/\\
                            ||----w |
                            ||     ||


Some references on the topic
----------------------------

[1] https://en.wikipedia.org/wiki/Rete_algorithm
[2] http://reports-archive.adm.cs.cmu.edu/anon/1995/CMU-CS-95-113.pdf

https://en.wikipedia.org/wiki/Propositional_formula
https://en.wikipedia.org/wiki/Inference_rule
https://en.wikipedia.org/wiki/List_of_rules_of_inference
"""
from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not

def _base_fact(atom):
    if False:
        for i in range(10):
            print('nop')
    'Return the literal fact of an atom.\n\n    Effectively, this merely strips the Not around a fact.\n    '
    if isinstance(atom, Not):
        return atom.arg
    else:
        return atom

def _as_pair(atom):
    if False:
        while True:
            i = 10
    if isinstance(atom, Not):
        return (atom.arg, False)
    else:
        return (atom, True)

def transitive_closure(implications):
    if False:
        return 10
    "\n    Computes the transitive closure of a list of implications\n\n    Uses Warshall's algorithm, as described at\n    http://www.cs.hope.edu/~cusack/Notes/Notes/DiscreteMath/Warshall.pdf.\n    "
    full_implications = set(implications)
    literals = set().union(*map(set, full_implications))
    for k in literals:
        for i in literals:
            if (i, k) in full_implications:
                for j in literals:
                    if (k, j) in full_implications:
                        full_implications.add((i, j))
    return full_implications

def deduce_alpha_implications(implications):
    if False:
        for i in range(10):
            print('nop')
    'deduce all implications\n\n       Description by example\n       ----------------------\n\n       given set of logic rules:\n\n         a -> b\n         b -> c\n\n       we deduce all possible rules:\n\n         a -> b, c\n         b -> c\n\n\n       implications: [] of (a,b)\n       return:       {} of a -> set([b, c, ...])\n    '
    implications = implications + [(Not(j), Not(i)) for (i, j) in implications]
    res = defaultdict(set)
    full_implications = transitive_closure(implications)
    for (a, b) in full_implications:
        if a == b:
            continue
        res[a].add(b)
    for (a, impl) in res.items():
        impl.discard(a)
        na = Not(a)
        if na in impl:
            raise ValueError('implications are inconsistent: %s -> %s %s' % (a, na, impl))
    return res

def apply_beta_to_alpha_route(alpha_implications, beta_rules):
    if False:
        while True:
            i = 10
    "apply additional beta-rules (And conditions) to already-built\n    alpha implication tables\n\n       TODO: write about\n\n       - static extension of alpha-chains\n       - attaching refs to beta-nodes to alpha chains\n\n\n       e.g.\n\n       alpha_implications:\n\n       a  ->  [b, !c, d]\n       b  ->  [d]\n       ...\n\n\n       beta_rules:\n\n       &(b,d) -> e\n\n\n       then we'll extend a's rule to the following\n\n       a  ->  [b, !c, d, e]\n    "
    x_impl = {}
    for x in alpha_implications.keys():
        x_impl[x] = (set(alpha_implications[x]), [])
    for (bcond, bimpl) in beta_rules:
        for bk in bcond.args:
            if bk in x_impl:
                continue
            x_impl[bk] = (set(), [])
    seen_static_extension = True
    while seen_static_extension:
        seen_static_extension = False
        for (bcond, bimpl) in beta_rules:
            if not isinstance(bcond, And):
                raise TypeError('Cond is not And')
            bargs = set(bcond.args)
            for (x, (ximpls, bb)) in x_impl.items():
                x_all = ximpls | {x}
                if bimpl not in x_all and bargs.issubset(x_all):
                    ximpls.add(bimpl)
                    bimpl_impl = x_impl.get(bimpl)
                    if bimpl_impl is not None:
                        ximpls |= bimpl_impl[0]
                    seen_static_extension = True
    for (bidx, (bcond, bimpl)) in enumerate(beta_rules):
        bargs = set(bcond.args)
        for (x, (ximpls, bb)) in x_impl.items():
            x_all = ximpls | {x}
            if bimpl in x_all:
                continue
            if any((Not(xi) in bargs or Not(xi) == bimpl for xi in x_all)):
                continue
            if bargs & x_all:
                bb.append(bidx)
    return x_impl

def rules_2prereq(rules):
    if False:
        for i in range(10):
            print('nop')
    "build prerequisites table from rules\n\n       Description by example\n       ----------------------\n\n       given set of logic rules:\n\n         a -> b, c\n         b -> c\n\n       we build prerequisites (from what points something can be deduced):\n\n         b <- a\n         c <- a, b\n\n       rules:   {} of a -> [b, c, ...]\n       return:  {} of c <- [a, b, ...]\n\n       Note however, that this prerequisites may be *not* enough to prove a\n       fact. An example is 'a -> b' rule, where prereq(a) is b, and prereq(b)\n       is a. That's because a=T -> b=T, and b=F -> a=F, but a=F -> b=?\n    "
    prereq = defaultdict(set)
    for ((a, _), impl) in rules.items():
        if isinstance(a, Not):
            a = a.args[0]
        for (i, _) in impl:
            if isinstance(i, Not):
                i = i.args[0]
            prereq[i].add(a)
    return prereq

class TautologyDetected(Exception):
    """(internal) Prover uses it for reporting detected tautology"""
    pass

class Prover:
    """ai - prover of logic rules

       given a set of initial rules, Prover tries to prove all possible rules
       which follow from given premises.

       As a result proved_rules are always either in one of two forms: alpha or
       beta:

       Alpha rules
       -----------

       This are rules of the form::

         a -> b & c & d & ...


       Beta rules
       ----------

       This are rules of the form::

         &(a,b,...) -> c & d & ...


       i.e. beta rules are join conditions that say that something follows when
       *several* facts are true at the same time.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.proved_rules = []
        self._rules_seen = set()

    def split_alpha_beta(self):
        if False:
            i = 10
            return i + 15
        'split proved rules into alpha and beta chains'
        rules_alpha = []
        rules_beta = []
        for (a, b) in self.proved_rules:
            if isinstance(a, And):
                rules_beta.append((a, b))
            else:
                rules_alpha.append((a, b))
        return (rules_alpha, rules_beta)

    @property
    def rules_alpha(self):
        if False:
            return 10
        return self.split_alpha_beta()[0]

    @property
    def rules_beta(self):
        if False:
            print('Hello World!')
        return self.split_alpha_beta()[1]

    def process_rule(self, a, b):
        if False:
            i = 10
            return i + 15
        'process a -> b rule'
        if not a or isinstance(b, bool):
            return
        if isinstance(a, bool):
            return
        if (a, b) in self._rules_seen:
            return
        else:
            self._rules_seen.add((a, b))
        try:
            self._process_rule(a, b)
        except TautologyDetected:
            pass

    def _process_rule(self, a, b):
        if False:
            return 10
        if isinstance(b, And):
            sorted_bargs = sorted(b.args, key=str)
            for barg in sorted_bargs:
                self.process_rule(a, barg)
        elif isinstance(b, Or):
            sorted_bargs = sorted(b.args, key=str)
            if not isinstance(a, Logic):
                if a in sorted_bargs:
                    raise TautologyDetected(a, b, 'a -> a|c|...')
            self.process_rule(And(*[Not(barg) for barg in b.args]), Not(a))
            for bidx in range(len(sorted_bargs)):
                barg = sorted_bargs[bidx]
                brest = sorted_bargs[:bidx] + sorted_bargs[bidx + 1:]
                self.process_rule(And(a, Not(barg)), Or(*brest))
        elif isinstance(a, And):
            sorted_aargs = sorted(a.args, key=str)
            if b in sorted_aargs:
                raise TautologyDetected(a, b, 'a & b -> a')
            self.proved_rules.append((a, b))
        elif isinstance(a, Or):
            sorted_aargs = sorted(a.args, key=str)
            if b in sorted_aargs:
                raise TautologyDetected(a, b, 'a | b -> a')
            for aarg in sorted_aargs:
                self.process_rule(aarg, b)
        else:
            self.proved_rules.append((a, b))
            self.proved_rules.append((Not(b), Not(a)))

class FactRules:
    """Rules that describe how to deduce facts in logic space

       When defined, these rules allow implications to quickly be determined
       for a set of facts. For this precomputed deduction tables are used.
       see `deduce_all_facts`   (forward-chaining)

       Also it is possible to gather prerequisites for a fact, which is tried
       to be proven.    (backward-chaining)


       Definition Syntax
       -----------------

       a -> b       -- a=T -> b=T  (and automatically b=F -> a=F)
       a -> !b      -- a=T -> b=F
       a == b       -- a -> b & b -> a
       a -> b & c   -- a=T -> b=T & c=T
       # TODO b | c


       Internals
       ---------

       .full_implications[k, v]: all the implications of fact k=v
       .beta_triggers[k, v]: beta rules that might be triggered when k=v
       .prereq  -- {} k <- [] of k's prerequisites

       .defined_facts -- set of defined fact names
    """

    def __init__(self, rules):
        if False:
            for i in range(10):
                print('nop')
        'Compile rules into internal lookup tables'
        if isinstance(rules, str):
            rules = rules.splitlines()
        P = Prover()
        for rule in rules:
            (a, op, b) = rule.split(None, 2)
            a = Logic.fromstring(a)
            b = Logic.fromstring(b)
            if op == '->':
                P.process_rule(a, b)
            elif op == '==':
                P.process_rule(a, b)
                P.process_rule(b, a)
            else:
                raise ValueError('unknown op %r' % op)
        self.beta_rules = []
        for (bcond, bimpl) in P.rules_beta:
            self.beta_rules.append(({_as_pair(a) for a in bcond.args}, _as_pair(bimpl)))
        impl_a = deduce_alpha_implications(P.rules_alpha)
        impl_ab = apply_beta_to_alpha_route(impl_a, P.rules_beta)
        self.defined_facts = {_base_fact(k) for k in impl_ab.keys()}
        full_implications = defaultdict(set)
        beta_triggers = defaultdict(set)
        for (k, (impl, betaidxs)) in impl_ab.items():
            full_implications[_as_pair(k)] = {_as_pair(i) for i in impl}
            beta_triggers[_as_pair(k)] = betaidxs
        self.full_implications = full_implications
        self.beta_triggers = beta_triggers
        prereq = defaultdict(set)
        rel_prereq = rules_2prereq(full_implications)
        for (k, pitems) in rel_prereq.items():
            prereq[k] |= pitems
        self.prereq = prereq

    def _to_python(self) -> str:
        if False:
            print('Hello World!')
        ' Generate a string with plain python representation of the instance '
        return '\n'.join(self.print_rules())

    @classmethod
    def _from_python(cls, data: dict):
        if False:
            print('Hello World!')
        ' Generate an instance from the plain python representation '
        self = cls('')
        for key in ['full_implications', 'beta_triggers', 'prereq']:
            d = defaultdict(set)
            d.update(data[key])
            setattr(self, key, d)
        self.beta_rules = data['beta_rules']
        self.defined_facts = set(data['defined_facts'])
        return self

    def _defined_facts_lines(self):
        if False:
            for i in range(10):
                print('nop')
        yield 'defined_facts = ['
        for fact in sorted(self.defined_facts):
            yield f'    {fact!r},'
        yield '] # defined_facts'

    def _full_implications_lines(self):
        if False:
            i = 10
            return i + 15
        yield 'full_implications = dict( ['
        for fact in sorted(self.defined_facts):
            for value in (True, False):
                yield f'    # Implications of {fact} = {value}:'
                yield f'    (({fact!r}, {value!r}), set( ('
                implications = self.full_implications[fact, value]
                for implied in sorted(implications):
                    yield f'        {implied!r},'
                yield '       ) ),'
                yield '     ),'
        yield ' ] ) # full_implications'

    def _prereq_lines(self):
        if False:
            return 10
        yield 'prereq = {'
        yield ''
        for fact in sorted(self.prereq):
            yield f'    # facts that could determine the value of {fact}'
            yield f'    {fact!r}: {{'
            for pfact in sorted(self.prereq[fact]):
                yield f'        {pfact!r},'
            yield '    },'
            yield ''
        yield '} # prereq'

    def _beta_rules_lines(self):
        if False:
            print('Hello World!')
        reverse_implications = defaultdict(list)
        for (n, (pre, implied)) in enumerate(self.beta_rules):
            reverse_implications[implied].append((pre, n))
        yield '# Note: the order of the beta rules is used in the beta_triggers'
        yield 'beta_rules = ['
        yield ''
        m = 0
        indices = {}
        for implied in sorted(reverse_implications):
            (fact, value) = implied
            yield f'    # Rules implying {fact} = {value}'
            for (pre, n) in reverse_implications[implied]:
                indices[n] = m
                m += 1
                setstr = ', '.join(map(str, sorted(pre)))
                yield f'    ({{{setstr}}},'
                yield f'        {implied!r}),'
            yield ''
        yield '] # beta_rules'
        yield 'beta_triggers = {'
        for query in sorted(self.beta_triggers):
            (fact, value) = query
            triggers = [indices[n] for n in self.beta_triggers[query]]
            yield f'    {query!r}: {triggers!r},'
        yield '} # beta_triggers'

    def print_rules(self) -> Iterator[str]:
        if False:
            return 10
        ' Returns a generator with lines to represent the facts and rules '
        yield from self._defined_facts_lines()
        yield ''
        yield ''
        yield from self._full_implications_lines()
        yield ''
        yield ''
        yield from self._prereq_lines()
        yield ''
        yield ''
        yield from self._beta_rules_lines()
        yield ''
        yield ''
        yield "generated_assumptions = {'defined_facts': defined_facts, 'full_implications': full_implications,"
        yield "               'prereq': prereq, 'beta_rules': beta_rules, 'beta_triggers': beta_triggers}"

class InconsistentAssumptions(ValueError):

    def __str__(self):
        if False:
            i = 10
            return i + 15
        (kb, fact, value) = self.args
        return '%s, %s=%s' % (kb, fact, value)

class FactKB(dict):
    """
    A simple propositional knowledge base relying on compiled inference rules.
    """

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '{\n%s}' % ',\n'.join(['\t%s: %s' % i for i in sorted(self.items())])

    def __init__(self, rules):
        if False:
            i = 10
            return i + 15
        self.rules = rules

    def _tell(self, k, v):
        if False:
            i = 10
            return i + 15
        'Add fact k=v to the knowledge base.\n\n        Returns True if the KB has actually been updated, False otherwise.\n        '
        if k in self and self[k] is not None:
            if self[k] == v:
                return False
            else:
                raise InconsistentAssumptions(self, k, v)
        else:
            self[k] = v
            return True

    def deduce_all_facts(self, facts):
        if False:
            while True:
                i = 10
        '\n        Update the KB with all the implications of a list of facts.\n\n        Facts can be specified as a dictionary or as a list of (key, value)\n        pairs.\n        '
        full_implications = self.rules.full_implications
        beta_triggers = self.rules.beta_triggers
        beta_rules = self.rules.beta_rules
        if isinstance(facts, dict):
            facts = facts.items()
        while facts:
            beta_maytrigger = set()
            for (k, v) in facts:
                if not self._tell(k, v) or v is None:
                    continue
                for (key, value) in full_implications[k, v]:
                    self._tell(key, value)
                beta_maytrigger.update(beta_triggers[k, v])
            facts = []
            for bidx in beta_maytrigger:
                (bcond, bimpl) = beta_rules[bidx]
                if all((self.get(k) is v for (k, v) in bcond)):
                    facts.append(bimpl)