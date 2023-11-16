import attr
from hypothesis.errors import Flaky, HypothesisException, StopTest
from hypothesis.internal.compat import int_to_bytes
from hypothesis.internal.conjecture.data import ConjectureData, DataObserver, Status, bits_to_bytes
from hypothesis.internal.conjecture.junkdrawer import IntList

class PreviouslyUnseenBehaviour(HypothesisException):
    pass

def inconsistent_generation():
    if False:
        print('Hello World!')
    raise Flaky('Inconsistent data generation! Data generation behaved differently between different runs. Is your data generation depending on external state?')
EMPTY: frozenset = frozenset()

@attr.s(slots=True)
class Killed:
    """Represents a transition to part of the tree which has been marked as
    "killed", meaning we want to treat it as not worth exploring, so it will
    be treated as if it were completely explored for the purposes of
    exhaustion."""
    next_node = attr.ib()

@attr.s(slots=True)
class Branch:
    """Represents a transition where multiple choices can be made as to what
    to drawn."""
    bit_length = attr.ib()
    children = attr.ib(repr=False)

    @property
    def max_children(self):
        if False:
            while True:
                i = 10
        return 1 << self.bit_length

@attr.s(slots=True, frozen=True)
class Conclusion:
    """Represents a transition to a finished state."""
    status = attr.ib()
    interesting_origin = attr.ib()

@attr.s(slots=True)
class TreeNode:
    """Node in a tree that corresponds to previous interactions with
    a ``ConjectureData`` object according to some fixed test function.

    This is functionally a variant patricia trie.
    See https://en.wikipedia.org/wiki/Radix_tree for the general idea,
    but what this means in particular here is that we have a very deep
    but very lightly branching tree and rather than store this as a fully
    recursive structure we flatten prefixes and long branches into
    lists. This significantly compacts the storage requirements.

    A single ``TreeNode`` corresponds to a previously seen sequence
    of calls to ``ConjectureData`` which we have never seen branch,
    followed by a ``transition`` which describes what happens next.
    """
    bit_lengths = attr.ib(default=attr.Factory(IntList))
    values = attr.ib(default=attr.Factory(IntList))
    __forced = attr.ib(default=None, init=False)
    transition = attr.ib(default=None)
    is_exhausted = attr.ib(default=False, init=False)

    @property
    def forced(self):
        if False:
            i = 10
            return i + 15
        if not self.__forced:
            return EMPTY
        return self.__forced

    def mark_forced(self, i):
        if False:
            print('Hello World!')
        'Note that the value at index ``i`` was forced.'
        assert 0 <= i < len(self.values)
        if self.__forced is None:
            self.__forced = set()
        self.__forced.add(i)

    def split_at(self, i):
        if False:
            for i in range(10):
                print('nop')
        'Splits the tree so that it can incorporate\n        a decision at the ``draw_bits`` call corresponding\n        to position ``i``, or raises ``Flaky`` if that was\n        meant to be a forced node.'
        if i in self.forced:
            inconsistent_generation()
        assert not self.is_exhausted
        key = self.values[i]
        child = TreeNode(bit_lengths=self.bit_lengths[i + 1:], values=self.values[i + 1:], transition=self.transition)
        self.transition = Branch(bit_length=self.bit_lengths[i], children={key: child})
        if self.__forced is not None:
            child.__forced = {j - i - 1 for j in self.__forced if j > i}
            self.__forced = {j for j in self.__forced if j < i}
        child.check_exhausted()
        del self.values[i:]
        del self.bit_lengths[i:]
        assert len(self.values) == len(self.bit_lengths) == i

    def check_exhausted(self):
        if False:
            i = 10
            return i + 15
        'Recalculates ``self.is_exhausted`` if necessary then returns\n        it.'
        if not self.is_exhausted and len(self.forced) == len(self.values) and (self.transition is not None):
            if isinstance(self.transition, (Conclusion, Killed)):
                self.is_exhausted = True
            elif len(self.transition.children) == self.transition.max_children:
                self.is_exhausted = all((v.is_exhausted for v in self.transition.children.values()))
        return self.is_exhausted

class DataTree:
    """Tracks the tree structure of a collection of ConjectureData
    objects, for use in ConjectureRunner."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.root = TreeNode()

    @property
    def is_exhausted(self):
        if False:
            print('Hello World!')
        'Returns True if every possible node is dead and thus the language\n        described must have been fully explored.'
        return self.root.is_exhausted

    def generate_novel_prefix(self, random):
        if False:
            return 10
        'Generate a short random string that (after rewriting) is not\n        a prefix of any buffer previously added to the tree.\n\n        The resulting prefix is essentially arbitrary - it would be nice\n        for it to be uniform at random, but previous attempts to do that\n        have proven too expensive.\n        '
        assert not self.is_exhausted
        novel_prefix = bytearray()

        def append_int(n_bits, value):
            if False:
                print('Hello World!')
            novel_prefix.extend(int_to_bytes(value, bits_to_bytes(n_bits)))
        current_node = self.root
        while True:
            assert not current_node.is_exhausted
            for (i, (n_bits, value)) in enumerate(zip(current_node.bit_lengths, current_node.values)):
                if i in current_node.forced:
                    append_int(n_bits, value)
                else:
                    while True:
                        k = random.getrandbits(n_bits)
                        if k != value:
                            append_int(n_bits, k)
                            break
                    return bytes(novel_prefix)
            else:
                assert not isinstance(current_node.transition, (Conclusion, Killed))
                if current_node.transition is None:
                    return bytes(novel_prefix)
                branch = current_node.transition
                assert isinstance(branch, Branch)
                n_bits = branch.bit_length
                check_counter = 0
                while True:
                    k = random.getrandbits(n_bits)
                    try:
                        child = branch.children[k]
                    except KeyError:
                        append_int(n_bits, k)
                        return bytes(novel_prefix)
                    if not child.is_exhausted:
                        append_int(n_bits, k)
                        current_node = child
                        break
                    check_counter += 1
                    assert check_counter != 1000 or len(branch.children) < 2 ** n_bits or any((not v.is_exhausted for v in branch.children.values()))

    def rewrite(self, buffer):
        if False:
            print('Hello World!')
        'Use previously seen ConjectureData objects to return a tuple of\n        the rewritten buffer and the status we would get from running that\n        buffer with the test function. If the status cannot be predicted\n        from the existing values it will be None.'
        buffer = bytes(buffer)
        data = ConjectureData.for_buffer(buffer)
        try:
            self.simulate_test_function(data)
            return (data.buffer, data.status)
        except PreviouslyUnseenBehaviour:
            return (buffer, None)

    def simulate_test_function(self, data):
        if False:
            while True:
                i = 10
        'Run a simulated version of the test function recorded by\n        this tree. Note that this does not currently call ``stop_example``\n        or ``start_example`` as these are not currently recorded in the\n        tree. This will likely change in future.'
        node = self.root
        try:
            while True:
                for (i, (n_bits, previous)) in enumerate(zip(node.bit_lengths, node.values)):
                    v = data.draw_bits(n_bits, forced=node.values[i] if i in node.forced else None)
                    if v != previous:
                        raise PreviouslyUnseenBehaviour
                if isinstance(node.transition, Conclusion):
                    t = node.transition
                    data.conclude_test(t.status, t.interesting_origin)
                elif node.transition is None:
                    raise PreviouslyUnseenBehaviour
                elif isinstance(node.transition, Branch):
                    v = data.draw_bits(node.transition.bit_length)
                    try:
                        node = node.transition.children[v]
                    except KeyError as err:
                        raise PreviouslyUnseenBehaviour from err
                else:
                    assert isinstance(node.transition, Killed)
                    data.observer.kill_branch()
                    node = node.transition.next_node
        except StopTest:
            pass

    def new_observer(self):
        if False:
            i = 10
            return i + 15
        return TreeRecordingObserver(self)

class TreeRecordingObserver(DataObserver):

    def __init__(self, tree):
        if False:
            for i in range(10):
                print('nop')
        self.__current_node = tree.root
        self.__index_in_current_node = 0
        self.__trail = [self.__current_node]
        self.killed = False

    def draw_bits(self, n_bits, forced, value):
        if False:
            i = 10
            return i + 15
        i = self.__index_in_current_node
        self.__index_in_current_node += 1
        node = self.__current_node
        assert len(node.bit_lengths) == len(node.values)
        if i < len(node.bit_lengths):
            if n_bits != node.bit_lengths[i]:
                inconsistent_generation()
            if forced and i not in node.forced:
                inconsistent_generation()
            if value != node.values[i]:
                node.split_at(i)
                assert i == len(node.values)
                new_node = TreeNode()
                branch = node.transition
                branch.children[value] = new_node
                self.__current_node = new_node
                self.__index_in_current_node = 0
        else:
            trans = node.transition
            if trans is None:
                node.bit_lengths.append(n_bits)
                node.values.append(value)
                if forced:
                    node.mark_forced(i)
            elif isinstance(trans, Conclusion):
                assert trans.status != Status.OVERRUN
                inconsistent_generation()
            else:
                assert isinstance(trans, Branch), trans
                if n_bits != trans.bit_length:
                    inconsistent_generation()
                try:
                    self.__current_node = trans.children[value]
                except KeyError:
                    self.__current_node = trans.children.setdefault(value, TreeNode())
                self.__index_in_current_node = 0
        if self.__trail[-1] is not self.__current_node:
            self.__trail.append(self.__current_node)

    def kill_branch(self):
        if False:
            while True:
                i = 10
        'Mark this part of the tree as not worth re-exploring.'
        if self.killed:
            return
        self.killed = True
        if self.__index_in_current_node < len(self.__current_node.values) or (self.__current_node.transition is not None and (not isinstance(self.__current_node.transition, Killed))):
            inconsistent_generation()
        if self.__current_node.transition is None:
            self.__current_node.transition = Killed(TreeNode())
            self.__update_exhausted()
        self.__current_node = self.__current_node.transition.next_node
        self.__index_in_current_node = 0
        self.__trail.append(self.__current_node)

    def conclude_test(self, status, interesting_origin):
        if False:
            print('Hello World!')
        'Says that ``status`` occurred at node ``node``. This updates the\n        node if necessary and checks for consistency.'
        if status == Status.OVERRUN:
            return
        i = self.__index_in_current_node
        node = self.__current_node
        if i < len(node.values) or isinstance(node.transition, Branch):
            inconsistent_generation()
        new_transition = Conclusion(status, interesting_origin)
        if node.transition is not None and node.transition != new_transition:
            if isinstance(node.transition, Conclusion) and (node.transition.status != Status.INTERESTING or new_transition.status != Status.VALID):
                raise Flaky(f'Inconsistent test results! Test case was {node.transition!r} on first run but {new_transition!r} on second')
        else:
            node.transition = new_transition
        assert node is self.__trail[-1]
        node.check_exhausted()
        assert len(node.values) > 0 or node.check_exhausted()
        if not self.killed:
            self.__update_exhausted()

    def __update_exhausted(self):
        if False:
            return 10
        for t in reversed(self.__trail):
            if not t.check_exhausted():
                break