class Node:

    def __init__(self):
        if False:
            print('Hello World!')
        self.children = [None] * 26
        self.pcnt = 0
        self.cnt = 0

class Trie(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__trie = Node()

    def insert(self, word):
        if False:
            print('Hello World!')
        '\n        :type word: str\n        :rtype: None\n        '
        curr = self.__trie
        curr.pcnt += 1
        for c in word:
            if curr.children[ord(c) - ord('a')] is None:
                curr.children[ord(c) - ord('a')] = Node()
            curr = curr.children[ord(c) - ord('a')]
            curr.pcnt += 1
        curr.cnt += 1

    def countWordsEqualTo(self, word):
        if False:
            print('Hello World!')
        '\n        :type word: str\n        :rtype: int\n        '
        curr = self.__trie
        for c in word:
            if curr.children[ord(c) - ord('a')] is None:
                return 0
            curr = curr.children[ord(c) - ord('a')]
        return curr.cnt

    def countWordsStartingWith(self, prefix):
        if False:
            return 10
        '\n        :type prefix: str\n        :rtype: int\n        '
        curr = self.__trie
        for c in prefix:
            if curr.children[ord(c) - ord('a')] is None:
                return 0
            curr = curr.children[ord(c) - ord('a')]
        return curr.pcnt

    def erase(self, word):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type word: str\n        :rtype: None\n        '
        cnt = self.countWordsEqualTo(word)
        if not cnt:
            return
        curr = self.__trie
        curr.pcnt -= 1
        for c in word:
            if curr.children[ord(c) - ord('a')].pcnt == 1:
                curr.children[ord(c) - ord('a')] = None
                return
            curr = curr.children[ord(c) - ord('a')]
            curr.pcnt -= 1
        curr.cnt -= 1