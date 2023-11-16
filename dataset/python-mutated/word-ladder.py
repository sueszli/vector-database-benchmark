from string import ascii_lowercase

class Solution(object):

    def ladderLength(self, beginWord, endWord, wordList):
        if False:
            while True:
                i = 10
        '\n        :type beginWord: str\n        :type endWord: str\n        :type wordList: List[str]\n        :rtype: int\n        '
        words = set(wordList)
        if endWord not in words:
            return 0
        (left, right) = ({beginWord}, {endWord})
        ladder = 2
        while left:
            words -= left
            new_left = set()
            for word in left:
                for new_word in (word[:i] + c + word[i + 1:] for i in xrange(len(beginWord)) for c in ascii_lowercase):
                    if new_word not in words:
                        continue
                    if new_word in right:
                        return ladder
                    new_left.add(new_word)
            left = new_left
            ladder += 1
            if len(left) > len(right):
                (left, right) = (right, left)
        return 0

class Solution2(object):

    def ladderLength(self, beginWord, endWord, wordList):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type beginWord: str\n        :type endWord: str\n        :type wordList: List[str]\n        :rtype: int\n        '
        lookup = set(wordList)
        if endWord not in lookup:
            return 0
        ladder = 2
        q = [beginWord]
        while q:
            new_q = []
            for word in q:
                for i in xrange(len(word)):
                    for j in ascii_lowercase:
                        new_word = word[:i] + j + word[i + 1:]
                        if new_word == endWord:
                            return ladder
                        if new_word in lookup:
                            lookup.remove(new_word)
                            new_q.append(new_word)
            q = new_q
            ladder += 1
        return 0