class Solution(object):

    def minOperations(self, logs):
        if False:
            i = 10
            return i + 15
        '\n        :type logs: List[str]\n        :rtype: int\n        '
        result = 0
        for log in logs:
            if log == '../':
                if result > 0:
                    result -= 1
            elif log != './':
                result += 1
        return result