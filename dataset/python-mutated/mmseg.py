"""最大正向匹配分词"""
from pypinyin.constants import PHRASES_DICT

class Seg(object):
    """正向最大匹配分词

    :type prefix_set: PrefixSet
    :param no_non_phrases: 是否严格按照词语分词，不允许把非词语的词当做词语进行分词
    :type no_non_phrases: bool
    """

    def __init__(self, prefix_set, no_non_phrases=False):
        if False:
            i = 10
            return i + 15
        self._prefix_set = prefix_set
        self._no_non_phrases = no_non_phrases

    def cut(self, text):
        if False:
            return 10
        '分词\n\n        :param text: 待分词的文本\n        :yield: 单个词语\n        '
        remain = text
        while remain:
            matched = ''
            for index in range(len(remain)):
                word = remain[:index + 1]
                if word in self._prefix_set:
                    matched = word
                else:
                    if matched and (not self._no_non_phrases or matched in PHRASES_DICT):
                        yield matched
                        matched = ''
                        remain = remain[index:]
                    elif self._no_non_phrases:
                        yield word[0]
                        remain = remain[index + 2 - len(word):]
                    else:
                        yield word
                        remain = remain[index + 1:]
                    matched = ''
                    break
            else:
                if self._no_non_phrases and remain not in PHRASES_DICT:
                    for x in remain:
                        yield x
                else:
                    yield remain
                break

    def train(self, words):
        if False:
            print('Hello World!')
        '训练分词器\n\n        :param words: 词语列表\n        '
        self._prefix_set.train(words)

class PrefixSet(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._set = set()

    def train(self, word_s):
        if False:
            return 10
        '更新 prefix set\n\n        :param word_s: 词语库列表\n        :type word_s: iterable\n        :return: None\n        '
        for word in word_s:
            for index in range(len(word)):
                self._set.add(word[:index + 1])

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        return key in self._set
p_set = PrefixSet()
p_set.train(PHRASES_DICT.keys())
seg = Seg(p_set, no_non_phrases=True)

def retrain(seg_instance):
    if False:
        for i in range(10):
            print('nop')
    '重新使用内置词典训练 seg_instance。\n\n    比如在增加自定义词语信息后需要调用这个模块重新训练分词器\n\n    :type seg_instance: Seg\n    '
    seg_instance.train(PHRASES_DICT.keys())