import itertools

class Suitor:

    def __init__(self, id, preference_list):
        if False:
            print('Hello World!')
        ' A Suitor consists of an integer id (between 0 and the total number\n        of Suitors), and a preference list implicitly defining a ranking of the\n        set of Suiteds.\n\n        E.g., Suitor(2, [5, 0, 3, 4, 1, 2]) says the third Suitor prefers the\n        Suited with index 5 the most, then the Suited with index 0, etc.\n\n        The Suitor will propose in decreasing order of preference, and maintains\n        the internal state index_to_propose_to to keep track of the next proposal.\n        '
        self.preference_list = preference_list
        self.index_to_propose_to = 0
        self.id = id

    def preference(self):
        if False:
            while True:
                i = 10
        return self.preference_list[self.index_to_propose_to]

    def post_rejection(self):
        if False:
            for i in range(10):
                print('nop')
        self.index_to_propose_to += 1

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, Suitor) and self.id == other.id

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.id)

    def __repr__(self):
        if False:
            return 10
        return 'Suitor({})'.format(self.id)

class Suited:

    def __init__(self, id, preference_list):
        if False:
            for i in range(10):
                print('nop')
        self.preference_list = preference_list
        self.held = None
        self.current_suitors = set()
        self.id = id

    def reject(self):
        if False:
            return 10
        'Return the subset of Suitors in self.current_suitors to reject,\n        leaving only the held Suitor in self.current_suitors.\n        '
        if len(self.current_suitors) == 0:
            return set()
        self.held = min(self.current_suitors, key=lambda suitor: self.preference_list.index(suitor.id))
        rejected = self.current_suitors - set([self.held])
        self.current_suitors = set([self.held])
        return rejected

    def add_suitor(self, suitor):
        if False:
            for i in range(10):
                print('nop')
        self.current_suitors.add(suitor)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, Suited) and self.id == other.id

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.id)

    def __repr__(self):
        if False:
            return 10
        return 'Suited({})'.format(self.id)

def stable_marriage(suitors, suiteds):
    if False:
        print('Hello World!')
    ' Construct a stable marriage between Suitors and Suiteds.\n\n    Arguments:\n        suitors: a list of Suitor\n        suiteds: a list of Suited, which deferred acceptance of Suitors.\n\n    Returns:\n        A dict {Suitor: Suited} matching Suitors to Suiteds.\n    '
    unassigned = set(suitors)
    while len(unassigned) > 0:
        for suitor in unassigned:
            next_to_propose_to = suiteds[suitor.preference()]
            next_to_propose_to.add_suitor(suitor)
        unassigned = set()
        for suited in suiteds:
            unassigned |= suited.reject()
        for suitor in unassigned:
            suitor.post_rejection()
    return dict([(suited.held, suited) for suited in suiteds])

def verify_stable(suitors, suiteds, marriage):
    if False:
        while True:
            i = 10
    ' Check that the assignment of suitors to suited is a stable marriage.\n\n    Arguments:\n        suitors: a list of Suitors\n        suiteds: a list of Suiteds\n        marriage: a matching {Suitor: Suited}\n\n    Returns:\n        True if the marriage is stable, otherwise a tuple (False, (x, y))\n        where x is a Suitor, y is a Suited, and (x, y) are a counterexample\n        to the claim that the marriage is stable.\n    '
    suited_to_suitor = dict(((v, k) for (k, v) in marriage.items()))

    def precedes(L, item1, item2):
        if False:
            for i in range(10):
                print('nop')
        return L.index(item1) < L.index(item2)

    def suitor_prefers(suitor, suited):
        if False:
            return 10
        return precedes(suitor.preference_list, suited.id, marriage[suitor].id)

    def suited_prefers(suited, suitor):
        if False:
            return 10
        return precedes(suited.preference_list, suitor.id, suited_to_suitor[suited].id)
    for (suitor, suited) in itertools.product(suitors, suiteds):
        if suited != marriage[suitor] and suitor_prefers(suitor, suited) and suited_prefers(suited, suitor):
            return (False, (suitor, suited))
    return True