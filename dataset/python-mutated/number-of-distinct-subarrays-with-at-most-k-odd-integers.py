import collections

class Solution(object):

    def distinctSubarraysWithAtMostKOddIntegers(self, A, K):
        if False:
            for i in range(10):
                print('nop')

        def countDistinct(A, left, right, trie):
            if False:
                print('Hello World!')
            result = 0
            for i in reversed(xrange(left, right + 1)):
                if A[i] not in trie:
                    result += 1
                trie = trie[A[i]]
            return result
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        (result, left, count) = (0, 0, 0)
        for right in xrange(len(A)):
            count += A[right] % 2
            while count > K:
                count -= A[left] % 2
                left += 1
            result += countDistinct(A, left, right, trie)
        return result

class Solution2(object):

    def distinctSubarraysWithAtMostKOddIntegers(self, A, K):
        if False:
            i = 10
            return i + 15

        def countDistinct(A, left, right, trie):
            if False:
                print('Hello World!')
            result = 0
            for i in xrange(left, right + 1):
                if A[i] not in trie:
                    result += 1
                trie = trie[A[i]]
            return result
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        result = 0
        for left in xrange(len(A)):
            count = 0
            for right in xrange(left, len(A)):
                count += A[right] % 2
                if count > K:
                    right -= 1
                    break
            result += countDistinct(A, left, right, trie)
        return result