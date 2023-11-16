class Solution(object):

    def arrangeWords(self, text):
        if False:
            while True:
                i = 10
        '\n        :type text: str\n        :rtype: str\n        '
        result = text.split()
        result[0] = result[0].lower()
        result.sort(key=len)
        result[0] = result[0].title()
        return ' '.join(result)