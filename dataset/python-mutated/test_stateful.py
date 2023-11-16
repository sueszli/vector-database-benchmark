from collections import namedtuple
import pytest
from hypothesis import settings as Settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, precondition, rule
from hypothesis.strategies import booleans, integers, lists
Leaf = namedtuple('Leaf', ('label',))
Split = namedtuple('Split', ('left', 'right'))

class BalancedTrees(RuleBasedStateMachine):
    trees = Bundle('BinaryTree')

    @rule(target=trees, x=booleans())
    def leaf(self, x):
        if False:
            while True:
                i = 10
        return Leaf(x)

    @rule(target=trees, left=trees, right=trees)
    def split(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        return Split(left, right)

    @rule(tree=trees)
    def test_is_balanced(self, tree):
        if False:
            while True:
                i = 10
        if isinstance(tree, Leaf):
            return
        else:
            assert abs(self.size(tree.left) - self.size(tree.right)) <= 1
            self.test_is_balanced(tree.left)
            self.test_is_balanced(tree.right)

    def size(self, tree):
        if False:
            while True:
                i = 10
        if isinstance(tree, Leaf):
            return 1
        else:
            return 1 + self.size(tree.left) + self.size(tree.right)

class DepthCharge:

    def __init__(self, value):
        if False:
            return 10
        if value is None:
            self.depth = 0
        else:
            self.depth = value.depth + 1

class DepthMachine(RuleBasedStateMachine):
    charges = Bundle('charges')

    @rule(targets=(charges,), child=charges)
    def charge(self, child):
        if False:
            print('Hello World!')
        return DepthCharge(child)

    @rule(targets=(charges,))
    def none_charge(self):
        if False:
            while True:
                i = 10
        return DepthCharge(None)

    @rule(check=charges)
    def is_not_too_deep(self, check):
        if False:
            i = 10
            return i + 15
        assert check.depth < 3

class RoseTreeStateMachine(RuleBasedStateMachine):
    nodes = Bundle('nodes')

    @rule(target=nodes, source=lists(nodes))
    def bunch(self, source):
        if False:
            print('Hello World!')
        return source

    @rule(source=nodes)
    def shallow(self, source):
        if False:
            for i in range(10):
                print('nop')

        def d(ls):
            if False:
                i = 10
                return i + 15
            if not ls:
                return 0
            else:
                return 1 + max(map(d, ls))
        assert d(source) <= 5

class NotTheLastMachine(RuleBasedStateMachine):
    stuff = Bundle('stuff')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.last = None
        self.bye_called = False

    @rule(target=stuff)
    def hi(self):
        if False:
            for i in range(10):
                print('nop')
        result = object()
        self.last = result
        return result

    @precondition(lambda self: not self.bye_called)
    @rule(v=stuff)
    def bye(self, v):
        if False:
            print('Hello World!')
        assert v == self.last
        self.bye_called = True

class PopulateMultipleTargets(RuleBasedStateMachine):
    b1 = Bundle('b1')
    b2 = Bundle('b2')

    @rule(targets=(b1, b2))
    def populate(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    @rule(x=b1, y=b2)
    def fail(self, x, y):
        if False:
            i = 10
            return i + 15
        raise AssertionError

class CanSwarm(RuleBasedStateMachine):
    """This test will essentially never pass if you choose rules uniformly at
    random, because every time the snake rule fires we return to the beginning,
    so we will tend to undo progress well before we make enough progress for
    the test to fail.

    This tests our swarm testing functionality in stateful testing by ensuring
    that we can sometimes generate long runs of steps which exclude a
    particular rule.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.seen = set()

    @rule(move=integers(0, 255))
    def ladder(self, move):
        if False:
            return 10
        self.seen.add(move)
        assert len(self.seen) <= 15

    @rule()
    def snake(self):
        if False:
            while True:
                i = 10
        self.seen.clear()
bad_machines = (BalancedTrees, DepthMachine, RoseTreeStateMachine, NotTheLastMachine, PopulateMultipleTargets, CanSwarm)
for m in bad_machines:
    m.TestCase.settings = Settings(m.TestCase.settings, max_examples=1000)
cheap_bad_machines = list(bad_machines)
cheap_bad_machines.remove(BalancedTrees)
with_cheap_bad_machines = pytest.mark.parametrize('machine', cheap_bad_machines, ids=[t.__name__ for t in cheap_bad_machines])

@pytest.mark.parametrize('machine', bad_machines, ids=[t.__name__ for t in bad_machines])
def test_bad_machines_fail(machine):
    if False:
        print('Hello World!')
    test_class = machine.TestCase
    try:
        test_class().runTest()
        raise RuntimeError('Expected an assertion error')
    except AssertionError as err:
        notes = err.__notes__
    steps = [l for l in notes if 'Step ' in l or 'state.' in l]
    assert 1 <= len(steps) <= 50