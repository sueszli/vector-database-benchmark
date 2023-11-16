from enum import Enum
from sortedcontainers import SortedList
from hypothesis.internal.conjecture.data import ConjectureData, ConjectureResult, Status
from hypothesis.internal.conjecture.junkdrawer import LazySequenceCopy, swap
from hypothesis.internal.conjecture.shrinker import sort_key
NO_SCORE = float('-inf')

class DominanceRelation(Enum):
    NO_DOMINANCE = 0
    EQUAL = 1
    LEFT_DOMINATES = 2
    RIGHT_DOMINATES = 3

def dominance(left, right):
    if False:
        return 10
    'Returns the dominance relation between ``left`` and ``right``, according\n    to the rules that one ConjectureResult dominates another if and only if it\n    is better in every way.\n\n    The things we currently consider to be "better" are:\n\n        * Something that is smaller in shrinking order is better.\n        * Something that has higher status is better.\n        * Each ``interesting_origin`` is treated as its own score, so if two\n          interesting examples have different origins then neither dominates\n          the other.\n        * For each target observation, a higher score is better.\n\n    In "normal" operation where there are no bugs or target observations, the\n    pareto front only has one element (the smallest valid test case), but for\n    more structured or failing tests it can be useful to track, and future work\n    will depend on it more.'
    if left.buffer == right.buffer:
        return DominanceRelation.EQUAL
    if sort_key(right.buffer) < sort_key(left.buffer):
        result = dominance(left=right, right=left)
        if result == DominanceRelation.LEFT_DOMINATES:
            return DominanceRelation.RIGHT_DOMINATES
        else:
            assert result == DominanceRelation.NO_DOMINANCE
            return result
    assert sort_key(left.buffer) < sort_key(right.buffer)
    if left.status < right.status:
        return DominanceRelation.NO_DOMINANCE
    if not right.tags.issubset(left.tags):
        return DominanceRelation.NO_DOMINANCE
    if left.status == Status.INTERESTING and left.interesting_origin != right.interesting_origin:
        return DominanceRelation.NO_DOMINANCE
    for target in set(left.target_observations) | set(right.target_observations):
        left_score = left.target_observations.get(target, NO_SCORE)
        right_score = right.target_observations.get(target, NO_SCORE)
        if right_score > left_score:
            return DominanceRelation.NO_DOMINANCE
    return DominanceRelation.LEFT_DOMINATES

class ParetoFront:
    """Maintains an approximate pareto front of ConjectureData objects. That
    is, we try to maintain a collection of objects such that no element of the
    collection is pareto dominated by any other. In practice we don't quite
    manage that, because doing so is computationally very expensive. Instead
    we maintain a random sample of data objects that are "rarely" dominated by
    any other element of the collection (roughly, no more than about 10%).

    Only valid test cases are considered to belong to the pareto front - any
    test case with a status less than valid is discarded.

    Note that the pareto front is potentially quite large, and currently this
    will store the entire front in memory. This is bounded by the number of
    valid examples we run, which is max_examples in normal execution, and
    currently we do not support workflows with large max_examples which have
    large values of max_examples very well anyway, so this isn't a major issue.
    In future we may weish to implement some sort of paging out to disk so that
    we can work with larger fronts.

    Additionally, because this is only an approximate pareto front, there are
    scenarios where it can be much larger than the actual pareto front. There
    isn't a huge amount we can do about this - checking an exact pareto front
    is intrinsically quadratic.

    "Most" of the time we should be relatively close to the true pareto front,
    say within an order of magnitude, but it's not hard to construct scenarios
    where this is not the case. e.g. suppose we enumerate all valid test cases
    in increasing shortlex order as s_1, ..., s_n, ... and have scores f and
    g such that f(s_i) = min(i, N) and g(s_i) = 1 if i >= N, then the pareto
    front is the set {s_1, ..., S_N}, but the only element of the front that
    will dominate s_i when i > N is S_N, which we select with probability
    1 / N. A better data structure could solve this, but at the cost of more
    expensive operations and higher per element memory use, so we'll wait to
    see how much of a problem this is in practice before we try that.
    """

    def __init__(self, random):
        if False:
            return 10
        self.__random = random
        self.__eviction_listeners = []
        self.front = SortedList(key=lambda d: sort_key(d.buffer))
        self.__pending = None

    def add(self, data):
        if False:
            i = 10
            return i + 15
        'Attempts to add ``data`` to the pareto front. Returns True if\n        ``data`` is now in the front, including if data is already in the\n        collection, and False otherwise'
        data = data.as_result()
        if data.status < Status.VALID:
            return False
        if not self.front:
            self.front.add(data)
            return True
        if data in self.front:
            return True
        self.front.add(data)
        assert self.__pending is None
        try:
            self.__pending = data
            front = LazySequenceCopy(self.front)
            to_remove = []
            i = self.front.index(data)
            failures = 0
            while i + 1 < len(front) and failures < 10:
                j = self.__random.randrange(i + 1, len(front))
                swap(front, j, len(front) - 1)
                candidate = front.pop()
                dom = dominance(data, candidate)
                assert dom != DominanceRelation.RIGHT_DOMINATES
                if dom == DominanceRelation.LEFT_DOMINATES:
                    to_remove.append(candidate)
                    failures = 0
                else:
                    failures += 1
            dominators = [data]
            while i >= 0 and len(dominators) < 10:
                swap(front, i, self.__random.randint(0, i))
                candidate = front[i]
                already_replaced = False
                j = 0
                while j < len(dominators):
                    v = dominators[j]
                    dom = dominance(candidate, v)
                    if dom == DominanceRelation.LEFT_DOMINATES:
                        if not already_replaced:
                            already_replaced = True
                            dominators[j] = candidate
                            j += 1
                        else:
                            (dominators[j], dominators[-1]) = (dominators[-1], dominators[j])
                            dominators.pop()
                        to_remove.append(v)
                    elif dom == DominanceRelation.RIGHT_DOMINATES:
                        to_remove.append(candidate)
                        break
                    elif dom == DominanceRelation.EQUAL:
                        break
                    else:
                        j += 1
                else:
                    dominators.append(candidate)
                i -= 1
            for v in to_remove:
                self.__remove(v)
            return data in self.front
        finally:
            self.__pending = None

    def on_evict(self, f):
        if False:
            print('Hello World!')
        'Register a listener function that will be called with data when it\n        gets removed from the front because something else dominates it.'
        self.__eviction_listeners.append(f)

    def __contains__(self, data):
        if False:
            print('Hello World!')
        return isinstance(data, (ConjectureData, ConjectureResult)) and data.as_result() in self.front

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.front)

    def __getitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        return self.front[i]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.front)

    def __remove(self, data):
        if False:
            return 10
        try:
            self.front.remove(data)
        except ValueError:
            return
        if data is not self.__pending:
            for f in self.__eviction_listeners:
                f(data)

class ParetoOptimiser:
    """Class for managing optimisation of the pareto front. That is, given the
    current best known pareto front, this class runs an optimisation process
    that attempts to bring it closer to the actual pareto front.

    Currently this is fairly basic and only handles pareto optimisation that
    works by reducing the test case in the shortlex order. We expect it will
    grow more powerful over time.
    """

    def __init__(self, engine):
        if False:
            print('Hello World!')
        self.__engine = engine
        self.front = self.__engine.pareto_front

    def run(self):
        if False:
            i = 10
            return i + 15
        seen = set()
        i = len(self.front) - 1
        prev = None
        while i >= 0 and (not self.__engine.interesting_examples):
            assert self.front
            i = min(i, len(self.front) - 1)
            target = self.front[i]
            if target.buffer in seen:
                i -= 1
                continue
            assert target is not prev
            prev = target

            def allow_transition(source, destination):
                if False:
                    i = 10
                    return i + 15
                "Shrink to data that strictly pareto dominates the current\n                best value we've seen, which is the current target of the\n                shrinker.\n\n                Note that during shrinking we may discover other smaller\n                examples that this function will reject and will get added to\n                the front. This is fine, because they will be processed on\n                later iterations of this loop."
                if dominance(destination, source) == DominanceRelation.LEFT_DOMINATES:
                    try:
                        self.front.front.remove(source)
                    except ValueError:
                        pass
                    return True
                return False
            shrunk = self.__engine.shrink(target, allow_transition=allow_transition)
            seen.add(shrunk.buffer)
            i = self.front.front.bisect_left(target)