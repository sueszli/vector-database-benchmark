class Solution(object):

    def maximumBooks(self, books):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type books: List[int]\n        :rtype: int\n        '

        def count(right, l):
            if False:
                return 10
            left = max(right - l + 1, 0)
            return (left + right) * (right - left + 1) // 2
        result = curr = 0
        stk = [-1]
        for i in xrange(len(books)):
            while stk[-1] != -1 and books[stk[-1]] >= books[i] - (i - stk[-1]):
                j = stk.pop()
                curr -= count(books[j], j - stk[-1])
            curr += count(books[i], i - stk[-1])
            stk.append(i)
            result = max(result, curr)
        return result