class TrieNode(object):

    def __init__(self):
        if False:
            return 10
        self.is_string = False
        self.leaves = {}

class Trie(object):

    def __init__(self):
        if False:
            return 10
        self.root = TrieNode()

    def insert(self, word):
        if False:
            print('Hello World!')
        cur = self.root
        for c in word:
            if not c in cur.leaves:
                cur.leaves[c] = TrieNode()
            cur = cur.leaves[c]
        cur.is_string = True

    def search(self, word):
        if False:
            return 10
        node = self.childSearch(word)
        if node:
            return node.is_string
        return False

    def startsWith(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        return self.childSearch(prefix) is not None

    def childSearch(self, word):
        if False:
            for i in range(10):
                print('nop')
        cur = self.root
        for c in word:
            if c in cur.leaves:
                cur = cur.leaves[c]
            else:
                return None
        return cur