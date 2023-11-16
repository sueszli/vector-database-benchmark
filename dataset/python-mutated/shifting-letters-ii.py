class Solution(object):

    def shiftingLetters(self, s, shifts):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type shifts: List[List[int]]\n        :rtype: str\n        '
        events = [0] * (len(s) + 1)
        for (b, e, d) in shifts:
            events[b] += 1 if d else -1
            events[e + 1] -= 1 if d else -1
        result = []
        curr = 0
        for i in xrange(len(s)):
            curr += events[i]
            result.append(chr(ord('a') + (ord(s[i]) - ord('a') + curr) % 26))
        return ''.join(result)