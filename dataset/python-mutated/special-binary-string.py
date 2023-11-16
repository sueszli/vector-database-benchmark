class Solution(object):

    def makeLargestSpecial(self, S):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type S: str\n        :rtype: str\n        '
        result = []
        anchor = count = 0
        for (i, v) in enumerate(S):
            count += 1 if v == '1' else -1
            if count == 0:
                result.append('1{}0'.format(self.makeLargestSpecial(S[anchor + 1:i])))
                anchor = i + 1
        result.sort(reverse=True)
        return ''.join(result)