import collections

class Solution(object):

    def finalString(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: str\n        '
        dq = collections.deque()
        parity = 0
        for x in s:
            if x == 'i':
                parity ^= 1
            else:
                dq.appendleft(x) if parity else dq.append(x)
        if parity:
            dq.reverse()
        return ''.join(dq)