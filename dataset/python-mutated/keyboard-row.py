class Solution(object):

    def findWords(self, words):
        if False:
            print('Hello World!')
        '\n        :type words: List[str]\n        :rtype: List[str]\n        '
        rows = [set(['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']), set(['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']), set(['z', 'x', 'c', 'v', 'b', 'n', 'm'])]
        result = []
        for word in words:
            k = 0
            for i in xrange(len(rows)):
                if word[0].lower() in rows[i]:
                    k = i
                    break
            for c in word:
                if c.lower() not in rows[k]:
                    break
            else:
                result.append(word)
        return result

class Solution2(object):

    def findWords(self, words):
        if False:
            i = 10
            return i + 15
        '\n        :type words: List[str]\n        :rtype: List[str]\n        '
        keyboard_rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        single_row_words = []
        for word in words:
            for row in keyboard_rows:
                if all((letter in row for letter in word.lower())):
                    single_row_words.append(word)
        return single_row_words