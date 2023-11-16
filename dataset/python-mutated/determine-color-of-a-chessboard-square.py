class Solution(object):

    def squareIsWhite(self, coordinates):
        if False:
            print('Hello World!')
        '\n        :type coordinates: str\n        :rtype: bool\n        '
        return (ord(coordinates[0]) - ord('a')) % 2 != (ord(coordinates[1]) - ord('1')) % 2