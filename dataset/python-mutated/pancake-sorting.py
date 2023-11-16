class BIT(object):

    def __init__(self, n):
        if False:
            return 10
        self.__bit = [0] * (n + 1)

    def add(self, i, val):
        if False:
            return 10
        i += 1
        while i < len(self.__bit):
            self.__bit[i] += val
            i += i & -i

    def query(self, i):
        if False:
            while True:
                i = 10
        i += 1
        ret = 0
        while i > 0:
            ret += self.__bit[i]
            i -= i & -i
        return ret

class Solution(object):

    def pancakeSort(self, arr):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :rtype: List[int]\n        '
        bit = BIT(len(arr))
        result = []
        for i in xrange(len(arr)):
            n = bit.query(arr[i] - 1 - 1)
            bit.add(arr[i] - 1, 1)
            if n == i:
                continue
            if n == 0:
                if i > 1:
                    result.append(i)
                result.append(i + 1)
            else:
                if n > 1:
                    result.append(n)
                result.append(i)
                result.append(i + 1)
                result.append(n + 1)
        return result

class Solution2(object):

    def pancakeSort(self, arr):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :rtype: List[int]\n        '

        def smallerMergeSort(idxs, start, end, counts):
            if False:
                while True:
                    i = 10
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
                if r <= mid:
                    tmp.append(idxs[i])
                counts[idxs[i][1]] += r - start
            while r <= mid:
                tmp.append(idxs[r])
                r += 1
            idxs[start:start + len(tmp)] = tmp
        idxs = []
        smaller_counts = [0] * len(arr)
        for (i, x) in enumerate(arr):
            idxs.append((x, i))
        smallerMergeSort(idxs, 0, len(idxs) - 1, smaller_counts)
        result = []
        for (i, n) in enumerate(smaller_counts):
            if n == i:
                continue
            if n == 0:
                if i > 1:
                    result.append(i)
                result.append(i + 1)
            else:
                if n > 1:
                    result.append(n)
                result.append(i)
                result.append(i + 1)
                result.append(n + 1)
        return result

class Solution3(object):

    def pancakeSort(self, A):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: List[int]\n        :rtype: List[int]\n        '

        def reverse(l, begin, end):
            if False:
                while True:
                    i = 10
            for i in xrange((end - begin) // 2):
                (l[begin + i], l[end - 1 - i]) = (l[end - 1 - i], l[begin + i])
        result = []
        for n in reversed(xrange(1, len(A) + 1)):
            i = A.index(n)
            reverse(A, 0, i + 1)
            result.append(i + 1)
            reverse(A, 0, n)
            result.append(n)
        return result