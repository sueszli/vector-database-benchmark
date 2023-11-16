class Solution(object):

    def categorizeBox(self, length, width, height, mass):
        if False:
            while True:
                i = 10
        '\n        :type length: int\n        :type width: int\n        :type height: int\n        :type mass: int\n        :rtype: str\n        '
        bulky = any((x >= 10 ** 4 for x in (length, width, height))) or length * width * height >= 10 ** 9
        heavy = mass >= 100
        if bulky and heavy:
            return 'Both'
        if bulky:
            return 'Bulky'
        if heavy:
            return 'Heavy'
        return 'Neither'

class Solution2(object):

    def categorizeBox(self, length, width, height, mass):
        if False:
            print('Hello World!')
        '\n        :type length: int\n        :type width: int\n        :type height: int\n        :type mass: int\n        :rtype: str\n        '
        CATEGORIES = ['Neither', 'Heavy', 'Bulky', 'Both']
        i = 2 * (any((x >= 10 ** 4 for x in (length, width, height))) or length * width * height >= 10 ** 9) + int(mass >= 100)
        return CATEGORIES[i]