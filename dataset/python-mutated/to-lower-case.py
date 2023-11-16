class Solution(object):

    def toLowerCase(self, str):
        if False:
            while True:
                i = 10
        '\n        :type str: str\n        :rtype: str\n        '
        return ''.join([chr(ord('a') + ord(c) - ord('A')) if 'A' <= c <= 'Z' else c for c in str])