import string

class Solution(object):

    def validIPAddress(self, IP):
        if False:
            i = 10
            return i + 15
        '\n        :type IP: str\n        :rtype: str\n        '
        blocks = IP.split('.')
        if len(blocks) == 4:
            for i in xrange(len(blocks)):
                if not blocks[i].isdigit() or not 0 <= int(blocks[i]) < 256 or (blocks[i][0] == '0' and len(blocks[i]) > 1):
                    return 'Neither'
            return 'IPv4'
        blocks = IP.split(':')
        if len(blocks) == 8:
            for i in xrange(len(blocks)):
                if not 1 <= len(blocks[i]) <= 4 or not all((c in string.hexdigits for c in blocks[i])):
                    return 'Neither'
            return 'IPv6'
        return 'Neither'