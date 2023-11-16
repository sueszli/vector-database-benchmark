class Solution(object):

    def shiftingLetters(self, S, shifts):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :type shifts: List[int]\n        :rtype: str\n        '
        result = []
        times = sum(shifts) % 26
        for (i, c) in enumerate(S):
            index = ord(c) - ord('a')
            result.append(chr(ord('a') + (index + times) % 26))
            times = (times - shifts[i]) % 26
        return ''.join(result)