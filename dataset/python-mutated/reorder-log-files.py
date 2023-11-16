class Solution(object):

    def reorderLogFiles(self, logs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type logs: List[str]\n        :rtype: List[str]\n        '

        def f(log):
            if False:
                print('Hello World!')
            (i, content) = log.split(' ', 1)
            return (0, content, i) if content[0].isalpha() else (1,)
        logs.sort(key=f)
        return logs