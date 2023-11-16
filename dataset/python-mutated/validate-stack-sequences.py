class Solution(object):

    def validateStackSequences(self, pushed, popped):
        if False:
            i = 10
            return i + 15
        '\n        :type pushed: List[int]\n        :type popped: List[int]\n        :rtype: bool\n        '
        i = 0
        s = []
        for v in pushed:
            s.append(v)
            while s and i < len(popped) and (s[-1] == popped[i]):
                s.pop()
                i += 1
        return i == len(popped)