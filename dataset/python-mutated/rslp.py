from nltk.data import load
from nltk.stem.api import StemmerI

class RSLPStemmer(StemmerI):
    """
    A stemmer for Portuguese.

        >>> from nltk.stem import RSLPStemmer
        >>> st = RSLPStemmer()
        >>> # opening lines of Erico Verissimo's "Música ao Longe"
        >>> text = '''
        ... Clarissa risca com giz no quadro-negro a paisagem que os alunos
        ... devem copiar . Uma casinha de porta e janela , em cima duma
        ... coxilha .'''
        >>> for token in text.split(): # doctest: +NORMALIZE_WHITESPACE
        ...     print(st.stem(token))
        clariss risc com giz no quadro-negr a pais que os alun dev copi .
        uma cas de port e janel , em cim dum coxilh .
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._model = []
        self._model.append(self.read_rule('step0.pt'))
        self._model.append(self.read_rule('step1.pt'))
        self._model.append(self.read_rule('step2.pt'))
        self._model.append(self.read_rule('step3.pt'))
        self._model.append(self.read_rule('step4.pt'))
        self._model.append(self.read_rule('step5.pt'))
        self._model.append(self.read_rule('step6.pt'))

    def read_rule(self, filename):
        if False:
            i = 10
            return i + 15
        rules = load('nltk:stemmers/rslp/' + filename, format='raw').decode('utf8')
        lines = rules.split('\n')
        lines = [line for line in lines if line != '']
        lines = [line for line in lines if line[0] != '#']
        lines = [line.replace('\t\t', '\t') for line in lines]
        rules = []
        for line in lines:
            rule = []
            tokens = line.split('\t')
            rule.append(tokens[0][1:-1])
            rule.append(int(tokens[1]))
            rule.append(tokens[2][1:-1])
            rule.append([token[1:-1] for token in tokens[3].split(',')])
            rules.append(rule)
        return rules

    def stem(self, word):
        if False:
            print('Hello World!')
        word = word.lower()
        if word[-1] == 's':
            word = self.apply_rule(word, 0)
        if word[-1] == 'a':
            word = self.apply_rule(word, 1)
        word = self.apply_rule(word, 3)
        word = self.apply_rule(word, 2)
        prev_word = word
        word = self.apply_rule(word, 4)
        if word == prev_word:
            prev_word = word
            word = self.apply_rule(word, 5)
            if word == prev_word:
                word = self.apply_rule(word, 6)
        return word

    def apply_rule(self, word, rule_index):
        if False:
            i = 10
            return i + 15
        rules = self._model[rule_index]
        for rule in rules:
            suffix_length = len(rule[0])
            if word[-suffix_length:] == rule[0]:
                if len(word) >= suffix_length + rule[1]:
                    if word not in rule[3]:
                        word = word[:-suffix_length] + rule[2]
                        break
        return word