class Solution(object):

    def minimumRecolors(self, blocks, k):
        if False:
            i = 10
            return i + 15
        '\n        :type blocks: str\n        :type k: int\n        :rtype: int\n        '
        result = k
        curr = 0
        for (i, x) in enumerate(blocks):
            curr += int(blocks[i] == 'W')
            if i + 1 - k < 0:
                continue
            result = min(result, curr)
            curr -= int(blocks[i + 1 - k] == 'W')
        return result