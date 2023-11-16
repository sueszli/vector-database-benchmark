class Solution(object):

    def decodeCiphertext(self, encodedText, rows):
        if False:
            print('Hello World!')
        '\n        :type encodedText: str\n        :type rows: int\n        :rtype: str\n        '
        cols = len(encodedText) // rows
        k = len(encodedText)
        for i in reversed(xrange(cols)):
            for j in reversed(xrange(i, len(encodedText), cols + 1)):
                if encodedText[j] != ' ':
                    k = j
                    break
            else:
                continue
            break
        result = []
        for i in xrange(cols):
            for j in xrange(i, len(encodedText), cols + 1):
                result.append(encodedText[j])
                if j == k:
                    break
            else:
                continue
            break
        return ''.join(result)

class Solution2(object):

    def decodeCiphertext(self, encodedText, rows):
        if False:
            i = 10
            return i + 15
        '\n        :type encodedText: str\n        :type rows: int\n        :rtype: str\n        '
        cols = len(encodedText) // rows
        result = []
        for i in xrange(cols):
            for j in xrange(i, len(encodedText), cols + 1):
                result.append(encodedText[j])
        while result and result[-1] == ' ':
            result.pop()
        return ''.join(result)