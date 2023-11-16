import collections

class TrieNode(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__TOP_COUNT = 3
        self.infos = []
        self.leaves = {}

    def insert(self, s, times):
        if False:
            return 10
        cur = self
        cur.add_info(s, times)
        for c in s:
            if c not in cur.leaves:
                cur.leaves[c] = TrieNode()
            cur = cur.leaves[c]
            cur.add_info(s, times)

    def add_info(self, s, times):
        if False:
            i = 10
            return i + 15
        for p in self.infos:
            if p[1] == s:
                p[0] = -times
                break
        else:
            self.infos.append([-times, s])
        self.infos.sort()
        if len(self.infos) > self.__TOP_COUNT:
            self.infos.pop()

class AutocompleteSystem(object):

    def __init__(self, sentences, times):
        if False:
            print('Hello World!')
        '\n        :type sentences: List[str]\n        :type times: List[int]\n        '
        self.__trie = TrieNode()
        self.__cur_node = self.__trie
        self.__search = []
        self.__sentence_to_count = collections.defaultdict(int)
        for (sentence, count) in zip(sentences, times):
            self.__sentence_to_count[sentence] = count
            self.__trie.insert(sentence, count)

    def input(self, c):
        if False:
            print('Hello World!')
        '\n        :type c: str\n        :rtype: List[str]\n        '
        result = []
        if c == '#':
            self.__sentence_to_count[''.join(self.__search)] += 1
            self.__trie.insert(''.join(self.__search), self.__sentence_to_count[''.join(self.__search)])
            self.__cur_node = self.__trie
            self.__search = []
        else:
            self.__search.append(c)
            if self.__cur_node:
                if c not in self.__cur_node.leaves:
                    self.__cur_node = None
                    return []
                self.__cur_node = self.__cur_node.leaves[c]
                result = [p[1] for p in self.__cur_node.infos]
        return result