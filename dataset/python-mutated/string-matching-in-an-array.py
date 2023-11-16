import collections

class AhoNode(object):

    def __init__(self):
        if False:
            return 10
        self.children = collections.defaultdict(AhoNode)
        self.indices = []
        self.suffix = None
        self.output = None

class AhoTrie(object):

    def step(self, letter):
        if False:
            while True:
                i = 10
        while self.__node and letter not in self.__node.children:
            self.__node = self.__node.suffix
        self.__node = self.__node.children[letter] if self.__node else self.__root
        return self.__get_ac_node_outputs(self.__node)

    def reset(self):
        if False:
            while True:
                i = 10
        self.__node = self.__root

    def __init__(self, patterns):
        if False:
            i = 10
            return i + 15
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
            return 10
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

    def stringMatching(self, words):
        if False:
            i = 10
            return i + 15
        '\n        :type words: List[str]\n        :rtype: List[str]\n        '
        trie = AhoTrie(words)
        lookup = set()
        for i in xrange(len(words)):
            trie.reset()
            for c in words[i]:
                for j in trie.step(c):
                    if j != i:
                        lookup.add(j)
        return [words[i] for i in lookup]

class Solution2(object):

    def stringMatching(self, words):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        :rtype: List[str]\n        '

        def getPrefix(pattern):
            if False:
                i = 10
                return i + 15
            prefix = [-1] * len(pattern)
            j = -1
            for i in xrange(1, len(pattern)):
                while j != -1 and pattern[j + 1] != pattern[i]:
                    j = prefix[j]
                if pattern[j + 1] == pattern[i]:
                    j += 1
                prefix[i] = j
            return prefix

        def kmp(text, pattern, prefix):
            if False:
                return 10
            if not pattern:
                return 0
            if len(text) < len(pattern):
                return -1
            j = -1
            for i in xrange(len(text)):
                while j != -1 and pattern[j + 1] != text[i]:
                    j = prefix[j]
                if pattern[j + 1] == text[i]:
                    j += 1
                if j + 1 == len(pattern):
                    return i - j
            return -1
        result = []
        for (i, pattern) in enumerate(words):
            prefix = getPrefix(pattern)
            for (j, text) in enumerate(words):
                if i != j and kmp(text, pattern, prefix) != -1:
                    result.append(pattern)
                    break
        return result

class Solution3(object):

    def stringMatching(self, words):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type words: List[str]\n        :rtype: List[str]\n        '
        result = []
        for (i, pattern) in enumerate(words):
            for (j, text) in enumerate(words):
                if i != j and pattern in text:
                    result.append(pattern)
                    break
        return result