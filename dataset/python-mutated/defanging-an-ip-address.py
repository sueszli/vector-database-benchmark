class Solution(object):

    def defangIPaddr(self, address):
        if False:
            i = 10
            return i + 15
        '\n        :type address: str\n        :rtype: str\n        '
        result = []
        for c in address:
            if c == '.':
                result.append('[.]')
            else:
                result.append(c)
        return ''.join(result)