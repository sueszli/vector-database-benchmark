class Node:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.children = dict()
        self.isEnd = False

class Trie:

    def __init__(self):
        if False:
            print('Hello World!')
        self.root = Node()

    def insert(self, word: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Node()
            cur = cur.children[ch]
        cur.isEnd = True

    def search(self, word: str) -> bool:
        if False:
            return 10
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return cur is not None and cur.isEnd

    def startsWith(self, prefix: str) -> bool:
        if False:
            return 10
        cur = self.root
        for ch in prefix:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return cur is not None