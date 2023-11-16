from hypothesis import strategies as st
from hypothesis.internal.conjecture.junkdrawer import IntList
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
INTEGERS = st.integers(0, 2 ** 68)

@st.composite
def valid_index(draw):
    if False:
        return 10
    machine = draw(st.runner())
    if not machine.model:
        return draw(st.nothing())
    return draw(st.integers(0, len(machine.model) - 1))

@st.composite
def valid_slice(draw):
    if False:
        for i in range(10):
            print('nop')
    machine = draw(st.runner())
    result = [draw(st.integers(0, max(3, len(machine.model) * 2 - 1))) for _ in range(2)]
    result.sort()
    return slice(*result)

class IntListRules(RuleBasedStateMachine):

    @initialize(ls=st.lists(INTEGERS))
    def starting_lists(self, ls):
        if False:
            for i in range(10):
                print('nop')
        self.model = list(ls)
        self.target = IntList(ls)

    @invariant()
    def lists_are_equivalent(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'model'):
            assert isinstance(self.model, list)
            assert isinstance(self.target, IntList)
            assert len(self.model) == len(self.target)
            assert list(self.target) == self.model

    @rule(n=INTEGERS)
    def append(self, n):
        if False:
            while True:
                i = 10
        self.model.append(n)
        self.target.append(n)

    @rule(i=valid_index() | valid_slice())
    def delete(self, i):
        if False:
            i = 10
            return i + 15
        del self.model[i]
        del self.target[i]

    @rule(sl=valid_slice())
    def slice(self, sl):
        if False:
            while True:
                i = 10
        self.model = self.model[sl]
        self.target = self.target[sl]

    @rule(i=valid_index())
    def agree_on_values(self, i):
        if False:
            while True:
                i = 10
        assert self.model[i] == self.target[i]
TestIntList = IntListRules.TestCase