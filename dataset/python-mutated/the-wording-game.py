class Solution(object):

    def canAliceWin(self, a, b):
        if False:
            print('Hello World!')
        '\n        :type a: List[str]\n        :type b: List[str]\n        :rtype: bool\n        '

        def is_closely_greater(a, b):
            if False:
                print('Hello World!')
            return ord(a[0]) - ord(b[0]) <= 1 and a > b
        result = True
        (i, j) = (0, -1)
        for _ in xrange(len({w[0] for w in a}) + len({w[0] for w in b})):
            j = next((j for j in xrange(j + 1, len(b)) if is_closely_greater(b[j], a[i])), len(b))
            if j == len(b):
                break
            while j + 1 < len(b) and b[j + 1][0] == b[j][0]:
                j += 1
            (a, b, i, j, result) = (b, a, j, i, not result)
        return result