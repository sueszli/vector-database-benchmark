import itertools

class Solution(object):

    def minimumIncompatibility(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        inf = (len(nums) - 1) * (len(nums) // k) + 1

        def backtracking(nums, d, lookup):
            if False:
                i = 10
                return i + 15
            if not nums:
                return 0
            if nums not in lookup:
                ret = inf
                for new_nums in itertools.combinations(nums, d):
                    new_nums_set = set(new_nums)
                    if len(new_nums_set) < d:
                        continue
                    left = []
                    for num in nums:
                        if num in new_nums_set:
                            new_nums_set.remove(num)
                            continue
                        left.append(num)
                    ret = min(ret, max(new_nums) - min(new_nums) + backtracking(tuple(left), d, lookup))
                lookup[nums] = ret
            return lookup[nums]
        result = backtracking(tuple(nums), len(nums) // k, {})
        return result if result != inf else -1

class Solution_TLE(object):

    def minimumIncompatibility(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        inf = (len(nums) - 1) * (len(nums) // k) + 1
        POW = [1]
        for i in xrange(len(nums)):
            POW.append(POW[-1] << 1)

        def popcount(n):
            if False:
                return 10
            result = 0
            while n:
                n &= n - 1
                result += 1
            return result

        def find_candidates(nums, k):
            if False:
                i = 10
                return i + 15
            total = POW[len(nums)] - 1
            m = len(nums) // k
            result = [inf] * (total + 1)
            for mask in xrange(total + 1):
                if popcount(mask) != m:
                    continue
                lookup = 0
                (mx, mn) = (0, inf)
                for i in xrange(len(nums)):
                    if mask & POW[i] == 0:
                        continue
                    if lookup & POW[nums[i]]:
                        break
                    lookup |= POW[nums[i]]
                    mx = max(mx, nums[i])
                    mn = min(mn, nums[i])
                else:
                    result[mask] = mx - mn
            return result
        candidates = find_candidates(nums, k)
        m = len(nums) // k
        total = POW[len(nums)] - 1
        dp = [inf] * (total + 1)
        dp[0] = 0
        for mask in xrange(total + 1):
            if popcount(mask) % m != 0:
                continue
            submask = mask
            while submask:
                dp[mask] = min(dp[mask], dp[mask - submask] + candidates[submask])
                submask = submask - 1 & mask
        return dp[-1] if dp[-1] != inf else -1
import collections
import sortedcontainers

class Solution_Wrong_Greedy_SortedList(object):

    def minimumIncompatibility(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def greedy(nums, k, is_reversed):
            if False:
                return 10
            count = collections.Counter(nums)
            if max(count.itervalues()) > k:
                return -1
            ordered_set = sortedcontainers.SortedList(count.iterkeys())
            freq_to_nodes = collections.defaultdict(collections.OrderedDict)
            for x in ordered_set:
                freq_to_nodes[count[x]][x] = count[x]
            stks = [[] for _ in xrange(k)]
            curr = 0
            while ordered_set:
                if len(stks) - curr in freq_to_nodes:
                    for x in freq_to_nodes[len(stks) - curr].iterkeys():
                        for i in xrange(curr, len(stks)):
                            stks[i].append(x)
                        count.pop(x)
                        ordered_set.remove(x)
                    freq_to_nodes.pop(len(stks) - curr)
                to_remove = []
                direction = (lambda x: x) if not is_reversed else reversed
                for x in direction(ordered_set):
                    stks[curr].append(x)
                    freq_to_nodes[count[x]].pop(x)
                    if not freq_to_nodes[count[x]]:
                        freq_to_nodes.pop(count[x])
                    count[x] -= 1
                    if not count[x]:
                        count.pop(x)
                        to_remove.append(x)
                    else:
                        freq_to_nodes[count[x]][x] = count[x]
                    if len(stks[curr]) == len(nums) // k:
                        curr += 1
                        break
                for x in to_remove:
                    ordered_set.remove(x)
            return sum([max(stk) - min(stk) for stk in stks])
        return min(greedy(nums, k, False), greedy(nums, k, True))
import collections
from random import randint, seed

class SkipNode(object):

    def __init__(self, level=0, val=None):
        if False:
            print('Hello World!')
        self.val = val
        self.nexts = [None] * level
        self.prevs = [None] * level

class SkipList(object):
    (P_NUMERATOR, P_DENOMINATOR) = (1, 2)
    MAX_LEVEL = 32

    def __init__(self, end=float('inf'), can_duplicated=False, cmp=lambda x, y: x < y):
        if False:
            return 10
        seed(0)
        self.__head = SkipNode()
        self.__len = 0
        self.__can_duplicated = can_duplicated
        self.__cmp = cmp
        self.add(end)
        self.__end = self.find(end)

    def begin(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__head.nexts[0]

    def end(self):
        if False:
            print('Hello World!')
        return self.__end

    def lower_bound(self, target):
        if False:
            i = 10
            return i + 15
        return self.__lower_bound(target, self.__find_prev_nodes(target))

    def find(self, target):
        if False:
            for i in range(10):
                print('nop')
        return self.__find(target, self.__find_prev_nodes(target))

    def add(self, val):
        if False:
            while True:
                i = 10
        if not self.__can_duplicated and self.find(val):
            return (self.find(val), False)
        node = SkipNode(self.__random_level(), val)
        if len(self.__head.nexts) < len(node.nexts):
            self.__head.nexts.extend([None] * (len(node.nexts) - len(self.__head.nexts)))
        prevs = self.__find_prev_nodes(val)
        for i in xrange(len(node.nexts)):
            node.nexts[i] = prevs[i].nexts[i]
            if prevs[i].nexts[i]:
                prevs[i].nexts[i].prevs[i] = node
            prevs[i].nexts[i] = node
            node.prevs[i] = prevs[i]
        self.__len += 1
        return node if self.__can_duplicated else (node, True)

    def remove(self, it):
        if False:
            while True:
                i = 10
        prevs = it.prevs
        curr = self.__find(it.val, prevs)
        if not curr:
            return self.__end
        self.__len -= 1
        for i in reversed(xrange(len(curr.nexts))):
            prevs[i].nexts[i] = curr.nexts[i]
            if curr.nexts[i]:
                curr.nexts[i].prevs[i] = prevs[i]
            if not self.__head.nexts[i]:
                self.__head.nexts.pop()
        return curr.nexts[0]

    def __lower_bound(self, val, prevs):
        if False:
            return 10
        if prevs:
            candidate = prevs[0].nexts[0]
            if candidate:
                return candidate
        return None

    def __find(self, val, prevs):
        if False:
            while True:
                i = 10
        candidate = self.__lower_bound(val, prevs)
        if candidate and candidate.val == val:
            return candidate
        return None

    def __find_prev_nodes(self, val):
        if False:
            while True:
                i = 10
        prevs = [None] * len(self.__head.nexts)
        curr = self.__head
        for i in reversed(xrange(len(self.__head.nexts))):
            while curr.nexts[i] and self.__cmp(curr.nexts[i].val, val):
                curr = curr.nexts[i]
            prevs[i] = curr
        return prevs

    def __random_level(self):
        if False:
            for i in range(10):
                print('nop')
        level = 1
        while randint(1, SkipList.P_DENOMINATOR) <= SkipList.P_NUMERATOR and level < SkipList.MAX_LEVEL:
            level += 1
        return level

    def __iter__(self):
        if False:
            return 10
        it = self.begin()
        while it != self.end():
            yield it.val
            it = it.nexts[0]

    def __len__(self):
        if False:
            print('Hello World!')
        return self.__len - 1

    def __str__(self):
        if False:
            return 10
        result = []
        for i in reversed(xrange(len(self.__head.nexts))):
            result.append([])
            curr = self.__head.nexts[i]
            while curr:
                result[-1].append(str(curr.val))
                curr = curr.nexts[i]
        return '\n'.join(map(lambda x: '->'.join(x), result))

class Solution_Wrong_Greedy_SkipList(object):

    def minimumIncompatibility(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def greedy(nums, k, is_reversed):
            if False:
                while True:
                    i = 10
            count = collections.Counter(nums)
            if max(count.itervalues()) > k:
                return -1
            ordered_set = SkipList() if not is_reversed else SkipList(end=float('-inf'), cmp=lambda x, y: x > y)
            freq_to_nodes = collections.defaultdict(collections.OrderedDict)
            for x in sorted(count.keys(), reverse=is_reversed):
                ordered_set.add(x)
                freq_to_nodes[count[x]][x] = count[x]
            stks = [[] for _ in xrange(k)]
            curr = 0
            while ordered_set:
                if len(stks) - curr in freq_to_nodes:
                    for x in freq_to_nodes[len(stks) - curr].iterkeys():
                        for i in xrange(curr, len(stks)):
                            stks[i].append(x)
                        count.pop(x)
                        ordered_set.remove(ordered_set.find(x))
                    freq_to_nodes.pop(len(stks) - curr)
                it = ordered_set.begin()
                while it != ordered_set.end():
                    x = it.val
                    stks[curr].append(x)
                    freq_to_nodes[count[x]].pop(x)
                    if not freq_to_nodes[count[x]]:
                        freq_to_nodes.pop(count[x])
                    count[x] -= 1
                    if not count[x]:
                        count.pop(x)
                        it = ordered_set.remove(it)
                    else:
                        freq_to_nodes[count[x]][x] = count[x]
                        it = it.nexts[0]
                    if len(stks[curr]) == len(nums) // k:
                        curr += 1
                        break
            return sum([max(stk) - min(stk) for stk in stks])
        return min(greedy(nums, k, False), greedy(nums, k, True))
import collections

class Solution_Wrong_Greedy(object):

    def minimumIncompatibility(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def greedy(nums, k, is_reversed):
            if False:
                for i in range(10):
                    print('nop')
            count = collections.Counter(nums)
            if max(count.itervalues()) > k:
                return -1
            sorted_keys = sorted(count.keys(), reverse=is_reversed)
            stks = [[] for _ in xrange(k)]
            (curr, remain) = (0, len(nums))
            while remain:
                for x in sorted_keys:
                    if count[x] != len(stks) - curr:
                        continue
                    for i in xrange(curr, len(stks)):
                        stks[i].append(x)
                    remain -= count[x]
                    count[x] = 0
                for x in sorted_keys:
                    if not count[x]:
                        continue
                    stks[curr].append(x)
                    remain -= 1
                    count[x] -= 1
                    if len(stks[curr]) == len(nums) // k:
                        curr += 1
                        break
            return sum([max(stk) - min(stk) for stk in stks])
        return min(greedy(nums, k, False), greedy(nums, k, True))