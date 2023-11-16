class TrieNode(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.indices = []
        self.children = [None] * 26

    def insert(self, words, i):
        if False:
            while True:
                i = 10
        cur = self
        for c in words[i]:
            if not cur.children[ord(c) - ord('a')]:
                cur.children[ord(c) - ord('a')] = TrieNode()
            cur = cur.children[ord(c) - ord('a')]
            cur.indices.append(i)

class Solution(object):

    def wordSquares(self, words):
        if False:
            return 10
        '\n        :type words: List[str]\n        :rtype: List[List[str]]\n        '
        result = []
        trie = TrieNode()
        for i in xrange(len(words)):
            trie.insert(words, i)
        curr = []
        for s in words:
            curr.append(s)
            self.wordSquaresHelper(words, trie, curr, result)
            curr.pop()
        return result

    def wordSquaresHelper(self, words, trie, curr, result):
        if False:
            for i in range(10):
                print('nop')
        if len(curr) >= len(words[0]):
            return result.append(list(curr))
        node = trie
        for s in curr:
            node = node.children[ord(s[len(curr)]) - ord('a')]
            if not node:
                return
        for i in node.indices:
            curr.append(words[i])
            self.wordSquaresHelper(words, trie, curr, result)
            curr.pop()