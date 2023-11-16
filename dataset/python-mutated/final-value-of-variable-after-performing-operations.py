class Solution(object):

    def finalValueAfterOperations(self, operations):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type operations: List[str]\n        :rtype: int\n        '
        return sum((1 if '+' == op[1] else -1 for op in operations))