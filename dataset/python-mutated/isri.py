"""
ISRI Arabic Stemmer

The algorithm for this stemmer is described in:

Taghva, K., Elkoury, R., and Coombs, J. 2005. Arabic Stemming without a root dictionary.
Information Science Research Institute. University of Nevada, Las Vegas, USA.

The Information Science Research Institute’s (ISRI) Arabic stemmer shares many features
with the Khoja stemmer. However, the main difference is that ISRI stemmer does not use root
dictionary. Also, if a root is not found, ISRI stemmer returned normalized form, rather than
returning the original unmodified word.

Additional adjustments were made to improve the algorithm:

1- Adding 60 stop words.
2- Adding the pattern (تفاعيل) to ISRI pattern set.
3- The step 2 in the original algorithm was normalizing all hamza. This step is discarded because it
increases the word ambiguities and changes the original root.

"""
import re
from nltk.stem.api import StemmerI

class ISRIStemmer(StemmerI):
    """
    ISRI Arabic stemmer based on algorithm: Arabic Stemming without a root dictionary.
    Information Science Research Institute. University of Nevada, Las Vegas, USA.

    A few minor modifications have been made to ISRI basic algorithm.
    See the source code of this module for more information.

    isri.stem(token) returns Arabic root for the given token.

    The ISRI Stemmer requires that all tokens have Unicode string types.
    If you use Python IDLE on Arabic Windows you have to decode text first
    using Arabic '1256' coding.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.p3 = ['كال', 'بال', 'ولل', 'وال']
        self.p2 = ['ال', 'لل']
        self.p1 = ['ل', 'ب', 'ف', 'س', 'و', 'ي', 'ت', 'ن', 'ا']
        self.s3 = ['تمل', 'همل', 'تان', 'تين', 'كمل']
        self.s2 = ['ون', 'ات', 'ان', 'ين', 'تن', 'كم', 'هن', 'نا', 'يا', 'ها', 'تم', 'كن', 'ني', 'وا', 'ما', 'هم']
        self.s1 = ['ة', 'ه', 'ي', 'ك', 'ت', 'ا', 'ن']
        self.pr4 = {0: ['م'], 1: ['ا'], 2: ['ا', 'و', 'ي'], 3: ['ة']}
        self.pr53 = {0: ['ا', 'ت'], 1: ['ا', 'ي', 'و'], 2: ['ا', 'ت', 'م'], 3: ['م', 'ي', 'ت'], 4: ['م', 'ت'], 5: ['ا', 'و'], 6: ['ا', 'م']}
        self.re_short_vowels = re.compile('[\\u064B-\\u0652]')
        self.re_hamza = re.compile('[\\u0621\\u0624\\u0626]')
        self.re_initial_hamza = re.compile('^[\\u0622\\u0623\\u0625]')
        self.stop_words = ['يكون', 'وليس', 'وكان', 'كذلك', 'التي', 'وبين', 'عليها', 'مساء', 'الذي', 'وكانت', 'ولكن', 'والتي', 'تكون', 'اليوم', 'اللذين', 'عليه', 'كانت', 'لذلك', 'أمام', 'هناك', 'منها', 'مازال', 'لازال', 'لايزال', 'مايزال', 'اصبح', 'أصبح', 'أمسى', 'امسى', 'أضحى', 'اضحى', 'مابرح', 'مافتئ', 'ماانفك', 'لاسيما', 'ولايزال', 'الحالي', 'اليها', 'الذين', 'فانه', 'والذي', 'وهذا', 'لهذا', 'فكان', 'ستكون', 'اليه', 'يمكن', 'بهذا', 'الذى']

    def stem(self, token):
        if False:
            return 10
        '\n        Stemming a word token using the ISRI stemmer.\n        '
        token = self.norm(token, 1)
        if token in self.stop_words:
            return token
        token = self.pre32(token)
        token = self.suf32(token)
        token = self.waw(token)
        token = self.norm(token, 2)
        if len(token) == 4:
            token = self.pro_w4(token)
        elif len(token) == 5:
            token = self.pro_w53(token)
            token = self.end_w5(token)
        elif len(token) == 6:
            token = self.pro_w6(token)
            token = self.end_w6(token)
        elif len(token) == 7:
            token = self.suf1(token)
            if len(token) == 7:
                token = self.pre1(token)
            if len(token) == 6:
                token = self.pro_w6(token)
                token = self.end_w6(token)
        return token

    def norm(self, word, num=3):
        if False:
            i = 10
            return i + 15
        '\n        normalization:\n        num=1  normalize diacritics\n        num=2  normalize initial hamza\n        num=3  both 1&2\n        '
        if num == 1:
            word = self.re_short_vowels.sub('', word)
        elif num == 2:
            word = self.re_initial_hamza.sub('ا', word)
        elif num == 3:
            word = self.re_short_vowels.sub('', word)
            word = self.re_initial_hamza.sub('ا', word)
        return word

    def pre32(self, word):
        if False:
            while True:
                i = 10
        'remove length three and length two prefixes in this order'
        if len(word) >= 6:
            for pre3 in self.p3:
                if word.startswith(pre3):
                    return word[3:]
        if len(word) >= 5:
            for pre2 in self.p2:
                if word.startswith(pre2):
                    return word[2:]
        return word

    def suf32(self, word):
        if False:
            for i in range(10):
                print('nop')
        'remove length three and length two suffixes in this order'
        if len(word) >= 6:
            for suf3 in self.s3:
                if word.endswith(suf3):
                    return word[:-3]
        if len(word) >= 5:
            for suf2 in self.s2:
                if word.endswith(suf2):
                    return word[:-2]
        return word

    def waw(self, word):
        if False:
            i = 10
            return i + 15
        'remove connective ‘و’ if it precedes a word beginning with ‘و’'
        if len(word) >= 4 and word[:2] == 'وو':
            word = word[1:]
        return word

    def pro_w4(self, word):
        if False:
            return 10
        'process length four patterns and extract length three roots'
        if word[0] in self.pr4[0]:
            word = word[1:]
        elif word[1] in self.pr4[1]:
            word = word[:1] + word[2:]
        elif word[2] in self.pr4[2]:
            word = word[:2] + word[3]
        elif word[3] in self.pr4[3]:
            word = word[:-1]
        else:
            word = self.suf1(word)
            if len(word) == 4:
                word = self.pre1(word)
        return word

    def pro_w53(self, word):
        if False:
            i = 10
            return i + 15
        'process length five patterns and extract length three roots'
        if word[2] in self.pr53[0] and word[0] == 'ا':
            word = word[1] + word[3:]
        elif word[3] in self.pr53[1] and word[0] == 'م':
            word = word[1:3] + word[4]
        elif word[0] in self.pr53[2] and word[4] == 'ة':
            word = word[1:4]
        elif word[0] in self.pr53[3] and word[2] == 'ت':
            word = word[1] + word[3:]
        elif word[0] in self.pr53[4] and word[2] == 'ا':
            word = word[1] + word[3:]
        elif word[2] in self.pr53[5] and word[4] == 'ة':
            word = word[:2] + word[3]
        elif word[0] in self.pr53[6] and word[1] == 'ن':
            word = word[2:]
        elif word[3] == 'ا' and word[0] == 'ا':
            word = word[1:3] + word[4]
        elif word[4] == 'ن' and word[3] == 'ا':
            word = word[:3]
        elif word[3] == 'ي' and word[0] == 'ت':
            word = word[1:3] + word[4]
        elif word[3] == 'و' and word[1] == 'ا':
            word = word[0] + word[2] + word[4]
        elif word[2] == 'ا' and word[1] == 'و':
            word = word[0] + word[3:]
        elif word[3] == 'ئ' and word[2] == 'ا':
            word = word[:2] + word[4]
        elif word[4] == 'ة' and word[1] == 'ا':
            word = word[0] + word[2:4]
        elif word[4] == 'ي' and word[2] == 'ا':
            word = word[:2] + word[3]
        else:
            word = self.suf1(word)
            if len(word) == 5:
                word = self.pre1(word)
        return word

    def pro_w54(self, word):
        if False:
            print('Hello World!')
        'process length five patterns and extract length four roots'
        if word[0] in self.pr53[2]:
            word = word[1:]
        elif word[4] == 'ة':
            word = word[:4]
        elif word[2] == 'ا':
            word = word[:2] + word[3:]
        return word

    def end_w5(self, word):
        if False:
            return 10
        'ending step (word of length five)'
        if len(word) == 4:
            word = self.pro_w4(word)
        elif len(word) == 5:
            word = self.pro_w54(word)
        return word

    def pro_w6(self, word):
        if False:
            while True:
                i = 10
        'process length six patterns and extract length three roots'
        if word.startswith('است') or word.startswith('مست'):
            word = word[3:]
        elif word[0] == 'م' and word[3] == 'ا' and (word[5] == 'ة'):
            word = word[1:3] + word[4]
        elif word[0] == 'ا' and word[2] == 'ت' and (word[4] == 'ا'):
            word = word[1] + word[3] + word[5]
        elif word[0] == 'ا' and word[3] == 'و' and (word[2] == word[4]):
            word = word[1] + word[4:]
        elif word[0] == 'ت' and word[2] == 'ا' and (word[4] == 'ي'):
            word = word[1] + word[3] + word[5]
        else:
            word = self.suf1(word)
            if len(word) == 6:
                word = self.pre1(word)
        return word

    def pro_w64(self, word):
        if False:
            return 10
        'process length six patterns and extract length four roots'
        if word[0] == 'ا' and word[4] == 'ا':
            word = word[1:4] + word[5]
        elif word.startswith('مت'):
            word = word[2:]
        return word

    def end_w6(self, word):
        if False:
            return 10
        'ending step (word of length six)'
        if len(word) == 5:
            word = self.pro_w53(word)
            word = self.end_w5(word)
        elif len(word) == 6:
            word = self.pro_w64(word)
        return word

    def suf1(self, word):
        if False:
            while True:
                i = 10
        'normalize short sufix'
        for sf1 in self.s1:
            if word.endswith(sf1):
                return word[:-1]
        return word

    def pre1(self, word):
        if False:
            while True:
                i = 10
        'normalize short prefix'
        for sp1 in self.p1:
            if word.startswith(sp1):
                return word[1:]
        return word