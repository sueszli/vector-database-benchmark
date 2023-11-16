import collections

class Solution(object):

    def countCharacters(self, words, chars):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        :type chars: str\n        :rtype: int\n        '

        def check(word, chars, count):
            if False:
                while True:
                    i = 10
            if len(word) > len(chars):
                return False
            curr_count = collections.Counter()
            for c in word:
                curr_count[c] += 1
                if c not in count or count[c] < curr_count[c]:
                    return False
            return True
        count = collections.Counter(chars)
        return sum((len(word) for word in words if check(word, chars, count)))