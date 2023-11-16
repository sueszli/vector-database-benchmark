class Solution(object):

    def lastVisitedIntegers(self, words):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type words: List[str]\n        :rtype: List[int]\n        '
        PREV = 'prev'
        (result, stk) = ([], [])
        i = -1
        for x in words:
            if x == PREV:
                result.append(stk[i] if i >= 0 else -1)
                i -= 1
                continue
            stk.append(int(x))
            i = len(stk) - 1
        return result