class Solution(object):

    def letterCasePermutation(self, S):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type S: str\n        :rtype: List[str]\n        '
        result = [[]]
        for c in S:
            if c.isalpha():
                for i in xrange(len(result)):
                    result.append(result[i][:])
                    result[i].append(c.lower())
                    result[-1].append(c.upper())
            else:
                for s in result:
                    s.append(c)
        return map(''.join, result)