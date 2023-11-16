class Solution(object):

    def numberOfBeams(self, bank):
        if False:
            print('Hello World!')
        '\n        :type bank: List[str]\n        :rtype: int\n        '
        result = prev = 0
        for x in bank:
            cnt = x.count('1')
            if not cnt:
                continue
            result += prev * cnt
            prev = cnt
        return result