class Solution(object):

    def subarrayBitwiseORs(self, A):
        if False:
            print('Hello World!')
        '\n        :type A: List[int]\n        :rtype: int\n        '
        (result, curr) = (set(), {0})
        for i in A:
            curr = {i} | {i | j for j in curr}
            result |= curr
        return len(result)