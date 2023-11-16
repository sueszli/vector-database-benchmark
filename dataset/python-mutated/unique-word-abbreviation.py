import collections

class ValidWordAbbr(object):

    def __init__(self, dictionary):
        if False:
            print('Hello World!')
        '\n        initialize your data structure here.\n        :type dictionary: List[str]\n        '
        self.lookup_ = collections.defaultdict(set)
        for word in dictionary:
            abbr = self.abbreviation(word)
            self.lookup_[abbr].add(word)

    def isUnique(self, word):
        if False:
            for i in range(10):
                print('nop')
        '\n        check if a word is unique.\n        :type word: str\n        :rtype: bool\n        '
        abbr = self.abbreviation(word)
        return self.lookup_[abbr] <= {word}

    def abbreviation(self, word):
        if False:
            while True:
                i = 10
        if len(word) <= 2:
            return word
        return word[0] + str(len(word) - 2) + word[-1]