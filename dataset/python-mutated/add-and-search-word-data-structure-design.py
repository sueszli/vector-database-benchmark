class TrieNode(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.is_string = False
        self.leaves = {}

class WordDictionary(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.root = TrieNode()

    def addWord(self, word):
        if False:
            while True:
                i = 10
        curr = self.root
        for c in word:
            if c not in curr.leaves:
                curr.leaves[c] = TrieNode()
            curr = curr.leaves[c]
        curr.is_string = True

    def search(self, word):
        if False:
            i = 10
            return i + 15
        return self.searchHelper(word, 0, self.root)

    def searchHelper(self, word, start, curr):
        if False:
            return 10
        if start == len(word):
            return curr.is_string
        if word[start] in curr.leaves:
            return self.searchHelper(word, start + 1, curr.leaves[word[start]])
        elif word[start] == '.':
            for c in curr.leaves:
                if self.searchHelper(word, start + 1, curr.leaves[c]):
                    return True
        return False