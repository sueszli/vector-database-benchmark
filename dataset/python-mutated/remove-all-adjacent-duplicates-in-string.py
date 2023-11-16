class Solution(object):

    def removeDuplicates(self, S):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :rtype: str\n        '
        result = []
        for c in S:
            if result and result[-1] == c:
                result.pop()
            else:
                result.append(c)
        return ''.join(result)