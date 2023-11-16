class Solution(object):

    def maximumNumber(self, num, change):
        if False:
            while True:
                i = 10
        '\n        :type num: str\n        :type change: List[int]\n        :rtype: str\n        '
        mutated = False
        result = map(int, list(num))
        for (i, d) in enumerate(result):
            if change[d] < d:
                if mutated:
                    break
            elif change[d] > d:
                result[i] = str(change[d])
                mutated = True
        return ''.join(map(str, result))