from modelscope.utils.chinese_utils import normalize_chinese_number

class TrieNode(object):

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Initialize your data structure here.\n        '
        self.data = {}
        self.is_word = False

class Trie(object):
    """
    trie-tree
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize your data structure here.\n        '
        self.root = TrieNode()

    def insert(self, word):
        if False:
            print('Hello World!')
        '\n        Inserts a word into the trie.\n        :type word: str\n        :rtype: void\n        '
        node = self.root
        for chars in word:
            child = node.data.get(chars)
            if not child:
                node.data[chars] = TrieNode()
            node = node.data[chars]
        node.is_word = True

    def search(self, word):
        if False:
            while True:
                i = 10
        '\n        Returns if the word is in the trie.\n        :type word: str\n        :rtype: bool\n        '
        node = self.root
        for chars in word:
            node = node.data.get(chars)
            if not node:
                return False
        return node.is_word

    def startsWith(self, prefix):
        if False:
            while True:
                i = 10
        '\n        Returns if there is any word in the trie that starts with the given prefix.\n        :type prefix: str\n        :rtype: bool\n        '
        node = self.root
        for chars in prefix:
            node = node.data.get(chars)
            if not node:
                return False
        return True

    def get_start(self, prefix):
        if False:
            print('Hello World!')
        '\n          Returns words started with prefix\n          :param prefix:\n          :return: words (list)\n        '

        def get_key(pre, pre_node):
            if False:
                print('Hello World!')
            word_list = []
            if pre_node.is_word:
                word_list.append(pre)
            for x in pre_node.data.keys():
                word_list.extend(get_key(pre + str(x), pre_node.data.get(x)))
            return word_list
        words = []
        if not self.startsWith(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for chars in prefix:
            node = node.data.get(chars)
        return get_key(prefix, node)

class TrieTokenizer(Trie):
    """
    word_split based on trie-tree
    """

    def __init__(self, dict_path):
        if False:
            print('Hello World!')
        super(TrieTokenizer, self).__init__()
        self.dict_path = dict_path
        self.create_trie_tree()

    def load_dict(self):
        if False:
            return 10
        words = []
        with open(self.dict_path, mode='r', encoding='utf-8') as file:
            for line in file:
                words.append(line.strip().split('\t')[0].encode('utf-8').decode('utf-8-sig'))
        return words

    def create_trie_tree(self):
        if False:
            return 10
        words = self.load_dict()
        for word in words:
            self.insert(word)

    def mine_tree(self, tree, sentence, trace_index):
        if False:
            while True:
                i = 10
        if trace_index <= len(sentence) - 1:
            if sentence[trace_index] in tree.data:
                trace_index = trace_index + 1
                trace_index = self.mine_tree(tree.data[sentence[trace_index - 1]], sentence, trace_index)
        return trace_index

    def tokenize(self, sentence):
        if False:
            return 10
        tokens = []
        sentence_len = len(sentence)
        while sentence_len != 0:
            trace_index = 0
            trace_index = self.mine_tree(self.root, sentence, trace_index)
            if trace_index == 0:
                tokens.append(sentence[0:1])
                sentence = sentence[1:len(sentence)]
                sentence_len = len(sentence)
            else:
                tokens.append(sentence[0:trace_index])
                sentence = sentence[trace_index:len(sentence)]
                sentence_len = len(sentence)
        return tokens

    def combine(self, token_list):
        if False:
            while True:
                i = 10
        flag = 0
        output = []
        temp = []
        for i in token_list:
            if len(i) != 1:
                if flag == 0:
                    output.append(i[:])
                else:
                    output.append(''.join(temp))
                    output.append(i[:])
                    temp = []
                    flag = 0
            elif flag == 0:
                temp.append(i)
                flag = 1
            else:
                temp.append(i)
        return output

class Text2Phone:

    def __init__(self, phone_dict_path):
        if False:
            i = 10
            return i + 15
        self.trie_cws = TrieTokenizer(phone_dict_path)
        self.phone_map = self.get_phone_map(phone_dict_path)

    def get_phone_map(self, phone_dict_path):
        if False:
            return 10
        phone_map = dict()
        with open(phone_dict_path, 'r') as phone_map_file_reader:
            for line in phone_map_file_reader:
                (key, phone_series) = line.strip().split('\t')
                if key not in phone_map:
                    phone_map[key] = phone_series
        return phone_map

    def trans(self, text):
        if False:
            return 10
        text = normalize_chinese_number(text)
        tokens = self.trie_cws.tokenize(text)
        phones = []
        for word in tokens:
            if word in self.phone_map:
                phones.append(self.phone_map[word])
            elif len(word) > 1:
                for char in word:
                    if char in self.phone_map:
                        phones.append(self.phone_map[char])
        return ' '.join(phones)