import collections

class AhoNode(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.children = collections.defaultdict(AhoNode)
        self.indices = []
        self.suffix = None
        self.output = None

class AhoTrie(object):

    def step(self, letter):
        if False:
            for i in range(10):
                print('nop')
        while self.__node and letter not in self.__node.children:
            self.__node = self.__node.suffix
        self.__node = self.__node.children[letter] if self.__node else self.__root
        return self.__get_ac_node_outputs(self.__node)

    def __init__(self, patterns):
        if False:
            return 10
        self.__root = self.__create_ac_trie(patterns)
        self.__node = self.__create_ac_suffix_and_output_links(self.__root)

    def __create_ac_trie(self, patterns):
        if False:
            for i in range(10):
                print('nop')
        root = AhoNode()
        for (i, pattern) in enumerate(patterns):
            node = root
            for c in pattern:
                node = node.children[c]
            node.indices.append(i)
        return root

    def __create_ac_suffix_and_output_links(self, root):
        if False:
            while True:
                i = 10
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
                child.output = child.suffix if child.suffix.indices else child.suffix.output
        return root

    def __get_ac_node_outputs(self, node):
        if False:
            i = 10
            return i + 15
        result = []
        for i in node.indices:
            result.append(i)
        output = node.output
        while output:
            for i in output.indices:
                result.append(i)
            output = output.output
        return result

class Solution(object):

    def indexPairs(self, text, words):
        if False:
            print('Hello World!')
        '\n        :type text: str\n        :type words: List[str]\n        :rtype: List[List[int]]\n        '
        result = []
        reversed_words = [w[::-1] for w in words]
        trie = AhoTrie(reversed_words)
        for i in reversed(xrange(len(text))):
            for j in trie.step(text[i]):
                result.append([i, i + len(reversed_words[j]) - 1])
        result.reverse()
        return result