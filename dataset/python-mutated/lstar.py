from bisect import bisect_right, insort
from collections import Counter
import attr
from hypothesis.errors import InvalidState
from hypothesis.internal.conjecture.dfa import DFA, cached
from hypothesis.internal.conjecture.junkdrawer import IntList, NotFound, SelfOrganisingList, find_integer
'\nThis module contains an implementation of the L* algorithm\nfor learning a deterministic finite automaton based on an\nunknown membership function and a series of examples of\nstrings that may or may not satisfy it.\n\nThe two relevant papers for understanding this are:\n\n* Angluin, Dana. "Learning regular sets from queries and counterexamples."\n  Information and computation 75.2 (1987): 87-106.\n* Rivest, Ronald L., and Robert E. Schapire. "Inference of finite automata\n  using homing sequences." Information and Computation 103.2 (1993): 299-347.\n  Note that we only use the material from section 4.5 "Improving Angluin\'s L*\n  algorithm" (page 318), and all of the rest of the material on homing\n  sequences can be skipped.\n\nThe former explains the core algorithm, the latter a modification\nwe use (which we have further modified) which allows it to\nbe implemented more efficiently.\n\nAlthough we continue to call this L*, we in fact depart heavily from it to the\npoint where honestly this is an entirely different algorithm and we should come\nup with a better name.\n\nWe have several major departures from the papers:\n\n1. We learn the automaton lazily as we traverse it. This is particularly\n   valuable because if we make many corrections on the same string we only\n   have to learn the transitions that correspond to the string we are\n   correcting on.\n2. We make use of our ``find_integer`` method rather than a binary search\n   as proposed in the Rivest and Schapire paper, as we expect that\n   usually most strings will be mispredicted near the beginning.\n3. We try to learn a smaller alphabet of "interestingly distinct"\n   values. e.g. if all bytes larger than two result in an invalid\n   string, there is no point in distinguishing those bytes. In aid\n   of this we learn a single canonicalisation table which maps integers\n   to smaller integers that we currently think are equivalent, and learn\n   their inequivalence where necessary. This may require more learning\n   steps, as at each stage in the process we might learn either an\n   inequivalent pair of integers or a new experiment, but it may greatly\n   reduce the number of membership queries we have to make.\n\n\nIn addition, we have a totally different approach for mapping a string to its\ncanonical representative, which will be explained below inline. The general gist\nis that our implementation is much more willing to make mistakes: It will often\ncreate a DFA that is demonstrably wrong, based on information that it already\nhas, but where it is too expensive to discover that before it causes us to\nmake a mistake.\n\nA note on performance: This code is not really fast enough for\nus to ever want to run in production on large strings, and this\nis somewhat intrinsic. We should only use it in testing or for\nlearning languages offline that we can record for later use.\n\n'

@attr.s(slots=True)
class DistinguishedState:
    """Relevant information for a state that we have witnessed as definitely
    distinct from ones we have previously seen so far."""
    index = attr.ib()
    label = attr.ib()
    accepting = attr.ib()
    experiments = attr.ib()
    transitions = attr.ib(default=attr.Factory(dict))

class LStar:
    """This class holds the state for learning a DFA. The current DFA can be
    accessed as the ``dfa`` member of this class. Such a DFA becomes invalid
    as soon as ``learn`` has been called, and should only be used until the
    next call to ``learn``.

    Note that many of the DFA methods are on this class, but it is not itself
    a DFA. The reason for this is that it stores mutable state which can cause
    the structure of the learned DFA to change in potentially arbitrary ways,
    making all cached properties become nonsense.
    """

    def __init__(self, member):
        if False:
            for i in range(10):
                print('nop')
        self.experiments = []
        self.__experiment_set = set()
        self.normalizer = IntegerNormalizer()
        self.__member_cache = {}
        self.__member = member
        self.__generation = 0
        self.__states = [DistinguishedState(index=0, label=b'', accepting=self.member(b''), experiments={b'': self.member(b'')})]
        self.__self_organising_states = SelfOrganisingList(self.__states)
        self.start = 0
        self.__dfa_changed()

    def __dfa_changed(self):
        if False:
            while True:
                i = 10
        'Note that something has changed, updating the generation\n        and resetting any cached state.'
        self.__generation += 1
        self.dfa = LearnedDFA(self)

    def is_accepting(self, i):
        if False:
            i = 10
            return i + 15
        'Equivalent to ``self.dfa.is_accepting(i)``'
        return self.__states[i].accepting

    def label(self, i):
        if False:
            print('Hello World!')
        'Returns the string label for state ``i``.'
        return self.__states[i].label

    def transition(self, i, c):
        if False:
            return 10
        'Equivalent to ``self.dfa.transition(i, c)```'
        c = self.normalizer.normalize(c)
        state = self.__states[i]
        try:
            return state.transitions[c]
        except KeyError:
            pass
        string = state.label + bytes([c])
        accumulated = {}
        counts = Counter()

        def equivalent(t):
            if False:
                i = 10
                return i + 15
            'Checks if ``string`` could possibly lead to state ``t``.'
            for (e, expected) in accumulated.items():
                if self.member(t.label + e) != expected:
                    counts[e] += 1
                    return False
            for (e, expected) in t.experiments.items():
                result = self.member(string + e)
                if result != expected:
                    if result:
                        accumulated[e] = result
                    return False
            return True
        try:
            destination = self.__self_organising_states.find(equivalent)
        except NotFound:
            i = len(self.__states)
            destination = DistinguishedState(index=i, label=string, experiments=accumulated, accepting=self.member(string))
            self.__states.append(destination)
            self.__self_organising_states.add(destination)
        state.transitions[c] = destination.index
        return destination.index

    def member(self, s):
        if False:
            i = 10
            return i + 15
        'Check whether this string is a member of the language\n        to be learned.'
        try:
            return self.__member_cache[s]
        except KeyError:
            result = self.__member(s)
            self.__member_cache[s] = result
            return result

    @property
    def generation(self):
        if False:
            while True:
                i = 10
        'Return an integer value that will be incremented\n        every time the DFA we predict changes.'
        return self.__generation

    def learn(self, string):
        if False:
            return 10
        'Learn to give the correct answer on this string.\n        That is, after this method completes we will have\n        ``self.dfa.matches(s) == self.member(s)``.\n\n        Note that we do not guarantee that this will remain\n        true in the event that learn is called again with\n        a different string. It is in principle possible that\n        future learning will cause us to make a mistake on\n        this string. However, repeatedly calling learn on\n        each of a set of strings until the generation stops\n        changing is guaranteed to terminate.\n        '
        string = bytes(string)
        correct_outcome = self.member(string)
        if self.dfa.matches(string) == correct_outcome:
            return
        while True:
            normalized = bytes((self.normalizer.normalize(c) for c in string))
            if self.member(normalized) == correct_outcome:
                string = normalized
                break
            alphabet = sorted(set(string), reverse=True)
            target = string
            for a in alphabet:

                def replace(b):
                    if False:
                        print('Hello World!')
                    if a == b:
                        return target
                    return bytes((b if c == a else c for c in target))
                self.normalizer.distinguish(a, lambda x: self.member(replace(x)))
                target = replace(self.normalizer.normalize(a))
                assert self.member(target) == correct_outcome
            assert target != normalized
            self.__dfa_changed()
        if self.dfa.matches(string) == correct_outcome:
            return
        while True:
            dfa = self.dfa
            states = [dfa.start]

            def seems_right(n):
                if False:
                    while True:
                        i = 10
                'After reading n characters from s, do we seem to be\n                in the right state?\n\n                We determine this by replacing the first n characters\n                of s with the label of the state we expect to be in.\n                If we are in the right state, that will replace a substring\n                with an equivalent one so must produce the same answer.\n                '
                if n > len(string):
                    return False
                while n >= len(states):
                    states.append(dfa.transition(states[-1], string[len(states) - 1]))
                return self.member(dfa.label(states[n]) + string[n:]) == correct_outcome
            assert seems_right(0)
            n = find_integer(seems_right)
            if n == len(string):
                assert dfa.matches(string) == correct_outcome
                break
            source = states[n]
            character = string[n]
            wrong_destination = states[n + 1]
            assert self.transition(source, character) == wrong_destination
            labels_wrong_destination = self.dfa.label(wrong_destination)
            labels_correct_destination = self.dfa.label(source) + bytes([character])
            ex = string[n + 1:]
            assert self.member(labels_wrong_destination + ex) != self.member(labels_correct_destination + ex)
            self.__states[wrong_destination].experiments[ex] = self.member(labels_wrong_destination + ex)
            del self.__states[source].transitions[character]
            self.__dfa_changed()
            new_destination = self.transition(source, string[n])
            assert new_destination != wrong_destination

class LearnedDFA(DFA):
    """This implements a lazily calculated DFA where states
    are labelled by some string that reaches them, and are
    distinguished by a membership test and a set of experiments."""

    def __init__(self, lstar):
        if False:
            print('Hello World!')
        super().__init__()
        self.__lstar = lstar
        self.__generation = lstar.generation

    def __check_changed(self):
        if False:
            while True:
                i = 10
        if self.__generation != self.__lstar.generation:
            raise InvalidState('The underlying L* model has changed, so this DFA is no longer valid. If you want to preserve a previously learned DFA for posterity, call canonicalise() on it first.')

    def label(self, i):
        if False:
            return 10
        self.__check_changed()
        return self.__lstar.label(i)

    @property
    def start(self):
        if False:
            for i in range(10):
                print('nop')
        self.__check_changed()
        return self.__lstar.start

    def is_accepting(self, i):
        if False:
            print('Hello World!')
        self.__check_changed()
        return self.__lstar.is_accepting(i)

    def transition(self, i, c):
        if False:
            while True:
                i = 10
        self.__check_changed()
        return self.__lstar.transition(i, c)

    @cached
    def successor_states(self, state):
        if False:
            while True:
                i = 10
        'Returns all of the distinct states that can be reached via one\n        transition from ``state``, in the lexicographic order of the\n        smallest character that reaches them.'
        seen = set()
        result = []
        for c in self.__lstar.normalizer.representatives():
            j = self.transition(state, c)
            if j not in seen:
                seen.add(j)
                result.append(j)
        return tuple(result)

class IntegerNormalizer:
    """A class for replacing non-negative integers with a
    "canonical" value that is equivalent for all relevant
    purposes."""

    def __init__(self):
        if False:
            return 10
        self.__values = IntList([0])
        self.__cache = {}

    def __repr__(self):
        if False:
            return 10
        return f'IntegerNormalizer({list(self.__values)!r})'

    def __copy__(self):
        if False:
            return 10
        result = IntegerNormalizer()
        result.__values = IntList(self.__values)
        return result

    def representatives(self):
        if False:
            for i in range(10):
                print('nop')
        yield from self.__values

    def normalize(self, value):
        if False:
            while True:
                i = 10
        'Return the canonical integer considered equivalent\n        to ``value``.'
        try:
            return self.__cache[value]
        except KeyError:
            pass
        i = bisect_right(self.__values, value) - 1
        assert i >= 0
        return self.__cache.setdefault(value, self.__values[i])

    def distinguish(self, value, test):
        if False:
            for i in range(10):
                print('nop')
        'Checks whether ``test`` gives the same answer for\n        ``value`` and ``self.normalize(value)``. If it does\n        not, updates the list of canonical values so that\n        it does.\n\n        Returns True if and only if this makes a change to\n        the underlying canonical values.'
        canonical = self.normalize(value)
        if canonical == value:
            return False
        value_test = test(value)
        if test(canonical) == value_test:
            return False
        self.__cache.clear()

        def can_lower(k):
            if False:
                while True:
                    i = 10
            new_canon = value - k
            if new_canon <= canonical:
                return False
            return test(new_canon) == value_test
        new_canon = value - find_integer(can_lower)
        assert new_canon not in self.__values
        insort(self.__values, new_canon)
        assert self.normalize(value) == new_canon
        return True