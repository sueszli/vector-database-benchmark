class Solution(object):

    def reverseOnlyLetters(self, S):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :rtype: str\n        '

        def getNext(S):
            if False:
                return 10
            for i in reversed(xrange(len(S))):
                if S[i].isalpha():
                    yield S[i]
        result = []
        letter = getNext(S)
        for i in xrange(len(S)):
            if S[i].isalpha():
                result.append(letter.next())
            else:
                result.append(S[i])
        return ''.join(result)