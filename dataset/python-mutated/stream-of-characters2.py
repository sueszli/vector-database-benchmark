import collections

class AhoNode(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.children = collections.defaultdict(AhoNode)
        self.suffix = None
        self.outputs = []

class AhoTrie(object):

    def step(self, letter):
        if False:
            for i in range(10):
                print('nop')
        while self.__node and letter not in self.__node.children:
            self.__node = self.__node.suffix
        self.__node = self.__node.children[letter] if self.__node else self.__root
        return self.__node.outputs

    def __init__(self, patterns):
        if False:
            i = 10
            return i + 15
        self.__root = self.__create_ac_trie(patterns)
        self.__node = self.__create_ac_suffix_and_output_links(self.__root)

    def __create_ac_trie(self, patterns):
        if False:
            print('Hello World!')
        root = AhoNode()
        for (i, pattern) in enumerate(patterns):
            node = root
            for c in pattern:
                node = node.children[c]
            node.outputs.append(i)
        return root

    def __create_ac_suffix_and_output_links(self, root):
        if False:
            for i in range(10):
                print('nop')
        queue = collections.deque()
        for node in root.children.itervalues():
            queue.append(node)
            node.suffix = root
        while queue:
            node = queue.popleft()
            for (c, child) in node.children.iteritems():
                queue.append(child)
                suffix = node.suffix
                while suffix and c not in suffix.children:
                    suffix = suffix.suffix
                child.suffix = suffix.children[c] if suffix else root
                child.outputs += child.suffix.outputs
        return root

class StreamChecker(object):

    def __init__(self, words):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        '
        self.__trie = AhoTrie(words)

    def query(self, letter):
        if False:
            print('Hello World!')
        '\n        :type letter: str\n        :rtype: bool\n        '
        return len(self.__trie.step(letter)) > 0