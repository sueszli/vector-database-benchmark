"""
A word stemmer based on the Lancaster (Paice/Husk) stemming algorithm.
Paice, Chris D. "Another Stemmer." ACM SIGIR Forum 24.3 (1990): 56-61.
"""
import re
from nltk.stem.api import StemmerI

class LancasterStemmer(StemmerI):
    """
    Lancaster Stemmer

        >>> from nltk.stem.lancaster import LancasterStemmer
        >>> st = LancasterStemmer()
        >>> st.stem('maximum')     # Remove "-um" when word is intact
        'maxim'
        >>> st.stem('presumably')  # Don't remove "-um" when word is not intact
        'presum'
        >>> st.stem('multiply')    # No action taken if word ends with "-ply"
        'multiply'
        >>> st.stem('provision')   # Replace "-sion" with "-j" to trigger "j" set of rules
        'provid'
        >>> st.stem('owed')        # Word starting with vowel must contain at least 2 letters
        'ow'
        >>> st.stem('ear')         # ditto
        'ear'
        >>> st.stem('saying')      # Words starting with consonant must contain at least 3
        'say'
        >>> st.stem('crying')      #     letters and one of those letters must be a vowel
        'cry'
        >>> st.stem('string')      # ditto
        'string'
        >>> st.stem('meant')       # ditto
        'meant'
        >>> st.stem('cement')      # ditto
        'cem'
        >>> st_pre = LancasterStemmer(strip_prefix_flag=True)
        >>> st_pre.stem('kilometer') # Test Prefix
        'met'
        >>> st_custom = LancasterStemmer(rule_tuple=("ssen4>", "s1t."))
        >>> st_custom.stem("ness") # Change s to t
        'nest'
    """
    default_rule_tuple = ('ai*2.', 'a*1.', 'bb1.', 'city3s.', 'ci2>', 'cn1t>', 'dd1.', 'dei3y>', 'deec2ss.', 'dee1.', 'de2>', 'dooh4>', 'e1>', 'feil1v.', 'fi2>', 'gni3>', 'gai3y.', 'ga2>', 'gg1.', 'ht*2.', 'hsiug5ct.', 'hsi3>', 'i*1.', 'i1y>', 'ji1d.', 'juf1s.', 'ju1d.', 'jo1d.', 'jeh1r.', 'jrev1t.', 'jsim2t.', 'jn1d.', 'j1s.', 'lbaifi6.', 'lbai4y.', 'lba3>', 'lbi3.', 'lib2l>', 'lc1.', 'lufi4y.', 'luf3>', 'lu2.', 'lai3>', 'lau3>', 'la2>', 'll1.', 'mui3.', 'mu*2.', 'msi3>', 'mm1.', 'nois4j>', 'noix4ct.', 'noi3>', 'nai3>', 'na2>', 'nee0.', 'ne2>', 'nn1.', 'pihs4>', 'pp1.', 're2>', 'rae0.', 'ra2.', 'ro2>', 'ru2>', 'rr1.', 'rt1>', 'rei3y>', 'sei3y>', 'sis2.', 'si2>', 'ssen4>', 'ss0.', 'suo3>', 'su*2.', 's*1>', 's0.', 'tacilp4y.', 'ta2>', 'tnem4>', 'tne3>', 'tna3>', 'tpir2b.', 'tpro2b.', 'tcud1.', 'tpmus2.', 'tpec2iv.', 'tulo2v.', 'tsis0.', 'tsi3>', 'tt1.', 'uqi3.', 'ugo1.', 'vis3j>', 'vie0.', 'vi2>', 'ylb1>', 'yli3y>', 'ylp0.', 'yl2>', 'ygo1.', 'yhp1.', 'ymo1.', 'ypo1.', 'yti3>', 'yte3>', 'ytl2.', 'yrtsi5.', 'yra3>', 'yro3>', 'yfi3.', 'ycn2t>', 'yca3>', 'zi2>', 'zy1s.')

    def __init__(self, rule_tuple=None, strip_prefix_flag=False):
        if False:
            print('Hello World!')
        'Create an instance of the Lancaster stemmer.'
        self.rule_dictionary = {}
        self._strip_prefix = strip_prefix_flag
        self._rule_tuple = rule_tuple if rule_tuple else self.default_rule_tuple

    def parseRules(self, rule_tuple=None):
        if False:
            print('Hello World!')
        'Validate the set of rules used in this stemmer.\n\n        If this function is called as an individual method, without using stem\n        method, rule_tuple argument will be compiled into self.rule_dictionary.\n        If this function is called within stem, self._rule_tuple will be used.\n\n        '
        rule_tuple = rule_tuple if rule_tuple else self._rule_tuple
        valid_rule = re.compile('^[a-z]+\\*?\\d[a-z]*[>\\.]?$')
        self.rule_dictionary = {}
        for rule in rule_tuple:
            if not valid_rule.match(rule):
                raise ValueError(f'The rule {rule} is invalid')
            first_letter = rule[0:1]
            if first_letter in self.rule_dictionary:
                self.rule_dictionary[first_letter].append(rule)
            else:
                self.rule_dictionary[first_letter] = [rule]

    def stem(self, word):
        if False:
            print('Hello World!')
        'Stem a word using the Lancaster stemmer.'
        word = word.lower()
        word = self.__stripPrefix(word) if self._strip_prefix else word
        intact_word = word
        if not self.rule_dictionary:
            self.parseRules()
        return self.__doStemming(word, intact_word)

    def __doStemming(self, word, intact_word):
        if False:
            for i in range(10):
                print('nop')
        'Perform the actual word stemming'
        valid_rule = re.compile('^([a-z]+)(\\*?)(\\d)([a-z]*)([>\\.]?)$')
        proceed = True
        while proceed:
            last_letter_position = self.__getLastLetter(word)
            if last_letter_position < 0 or word[last_letter_position] not in self.rule_dictionary:
                proceed = False
            else:
                rule_was_applied = False
                for rule in self.rule_dictionary[word[last_letter_position]]:
                    rule_match = valid_rule.match(rule)
                    if rule_match:
                        (ending_string, intact_flag, remove_total, append_string, cont_flag) = rule_match.groups()
                        remove_total = int(remove_total)
                        if word.endswith(ending_string[::-1]):
                            if intact_flag:
                                if word == intact_word and self.__isAcceptable(word, remove_total):
                                    word = self.__applyRule(word, remove_total, append_string)
                                    rule_was_applied = True
                                    if cont_flag == '.':
                                        proceed = False
                                    break
                            elif self.__isAcceptable(word, remove_total):
                                word = self.__applyRule(word, remove_total, append_string)
                                rule_was_applied = True
                                if cont_flag == '.':
                                    proceed = False
                                break
                if rule_was_applied == False:
                    proceed = False
        return word

    def __getLastLetter(self, word):
        if False:
            print('Hello World!')
        'Get the zero-based index of the last alphabetic character in this string'
        last_letter = -1
        for position in range(len(word)):
            if word[position].isalpha():
                last_letter = position
            else:
                break
        return last_letter

    def __isAcceptable(self, word, remove_total):
        if False:
            return 10
        'Determine if the word is acceptable for stemming.'
        word_is_acceptable = False
        if word[0] in 'aeiouy':
            if len(word) - remove_total >= 2:
                word_is_acceptable = True
        elif len(word) - remove_total >= 3:
            if word[1] in 'aeiouy':
                word_is_acceptable = True
            elif word[2] in 'aeiouy':
                word_is_acceptable = True
        return word_is_acceptable

    def __applyRule(self, word, remove_total, append_string):
        if False:
            i = 10
            return i + 15
        'Apply the stemming rule to the word'
        new_word_length = len(word) - remove_total
        word = word[0:new_word_length]
        if append_string:
            word += append_string
        return word

    def __stripPrefix(self, word):
        if False:
            while True:
                i = 10
        'Remove prefix from a word.\n\n        This function originally taken from Whoosh.\n\n        '
        for prefix in ('kilo', 'micro', 'milli', 'intra', 'ultra', 'mega', 'nano', 'pico', 'pseudo'):
            if word.startswith(prefix):
                return word[len(prefix):]
        return word

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<LancasterStemmer>'