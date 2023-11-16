class BIT(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.__bit = [0] * (n + 1)

    def add(self, i, val):
        if False:
            for i in range(10):
                print('nop')
        i += 1
        while i < len(self.__bit):
            self.__bit[i] += val
            i += i & -i

    def query(self, i):
        if False:
            print('Hello World!')
        i += 1
        ret = 0
        while i > 0:
            ret += self.__bit[i]
            i -= i & -i
        return ret

class Solution(object):

    def createSortedArray(self, instructions):
        if False:
            i = 10
            return i + 15
        '\n        :type instructions: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        bit = BIT(max(instructions))
        result = 0
        for (i, inst) in enumerate(instructions):
            inst -= 1
            result += min(bit.query(inst - 1), i - bit.query(inst))
            bit.add(inst, 1)
        return result % MOD
import itertools

class Solution_TLE(object):

    def createSortedArray(self, instructions):
        if False:
            print('Hello World!')
        '\n        :type instructions: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def smallerMergeSort(idxs, start, end, counts):
            if False:
                i = 10
                return i + 15
            if end - start <= 0:
                return 0
            mid = start + (end - start) // 2
            smallerMergeSort(idxs, start, mid, counts)
            smallerMergeSort(idxs, mid + 1, end, counts)
            r = start
            tmp = []
            for i in xrange(mid + 1, end + 1):
                while r <= mid and idxs[r][0] < idxs[i][0]:
                    tmp.append(idxs[r])
                    r += 1
                tmp.append(idxs[i])
                counts[idxs[i][1]] += r - start
            while r <= mid:
                tmp.append(idxs[r])
                r += 1
            idxs[start:start + len(tmp)] = tmp

        def largerMergeSort(idxs, start, end, counts):
            if False:
                return 10
            if end - start <= 0:
                return 0
            mid = start + (end - start) // 2
            largerMergeSort(idxs, start, mid, counts)
            largerMergeSort(idxs, mid + 1, end, counts)
            r = start
            tmp = []
            for i in xrange(mid + 1, end + 1):
                while r <= mid and idxs[r][0] <= idxs[i][0]:
                    tmp.append(idxs[r])
                    r += 1
                if r <= mid:
                    tmp.append(idxs[i])
                counts[idxs[i][1]] += mid - r + 1
            while r <= mid:
                tmp.append(idxs[r])
                r += 1
            idxs[start:start + len(tmp)] = tmp
        idxs = []
        (smaller_counts, larger_counts) = [[0] * len(instructions) for _ in xrange(2)]
        for (i, inst) in enumerate(instructions):
            idxs.append((inst, i))
        smallerMergeSort(idxs[:], 0, len(idxs) - 1, smaller_counts)
        largerMergeSort(idxs, 0, len(idxs) - 1, larger_counts)
        return sum((min(s, l) for (s, l) in itertools.izip(smaller_counts, larger_counts))) % MOD