class Solution(object):

    def capitalizeTitle(self, title):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type title: str\n        :rtype: str\n        '
        title = list(title)
        j = 0
        for i in xrange(len(title) + 1):
            if i < len(title) and title[i] != ' ':
                title[i] = title[i].lower()
                continue
            if i - j > 2:
                title[j] = title[j].upper()
            j = i + 1
        return ''.join(title)