"""
ARLSTem Arabic Stemmer
The details about the implementation of this algorithm are described in:
K. Abainia, S. Ouamour and H. Sayoud, A Novel Robust Arabic Light Stemmer ,
Journal of Experimental & Theoretical Artificial Intelligence (JETAI'17),
Vol. 29, No. 3, 2017, pp. 557-573.
The ARLSTem is a light Arabic stemmer that is based on removing the affixes
from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
compared to several other stemmers using Paice's parameters (under-stemming
index, over-stemming index and stemming weight), and the results showed that
ARLSTem is promising and producing high performances. This stemmer is not
based on any dictionary and can be used on-line effectively.
"""
import re
from nltk.stem.api import StemmerI

class ARLSTem(StemmerI):
    """
    ARLSTem stemmer : a light Arabic Stemming algorithm without any dictionary.
    Department of Telecommunication & Information Processing. USTHB University,
    Algiers, Algeria.
    ARLSTem.stem(token) returns the Arabic stem for the input token.
    The ARLSTem Stemmer requires that all tokens are encoded using Unicode
    encoding.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.re_hamzated_alif = re.compile('[\\u0622\\u0623\\u0625]')
        self.re_alifMaqsura = re.compile('[\\u0649]')
        self.re_diacritics = re.compile('[\\u064B-\\u065F]')
        self.pr2 = ['ال', 'لل', 'فل', 'فب']
        self.pr3 = ['بال', 'كال', 'وال']
        self.pr32 = ['فلل', 'ولل']
        self.pr4 = ['فبال', 'وبال', 'فكال']
        self.su2 = ['كي', 'كم']
        self.su22 = ['ها', 'هم']
        self.su3 = ['كما', 'كنّ']
        self.su32 = ['هما', 'هنّ']
        self.pl_si2 = ['ان', 'ين', 'ون']
        self.pl_si3 = ['تان', 'تين']
        self.verb_su2 = ['ان', 'ون']
        self.verb_pr2 = ['ست', 'سي']
        self.verb_pr22 = ['سا', 'سن']
        self.verb_pr33 = ['لن', 'لت', 'لي', 'لأ']
        self.verb_suf3 = ['تما', 'تنّ']
        self.verb_suf2 = ['نا', 'تم', 'تا', 'وا']
        self.verb_suf1 = ['ت', 'ا', 'ن']

    def stem(self, token):
        if False:
            return 10
        "\n        call this function to get the word's stem based on ARLSTem .\n        "
        try:
            if token is None:
                raise ValueError('The word could not be stemmed, because                                  it is empty !')
            token = self.norm(token)
            pre = self.pref(token)
            if pre is not None:
                token = pre
            token = self.suff(token)
            ps = self.plur2sing(token)
            if ps is None:
                fm = self.fem2masc(token)
                if fm is not None:
                    return fm
                elif pre is None:
                    return self.verb(token)
            else:
                return ps
            return token
        except ValueError as e:
            print(e)

    def norm(self, token):
        if False:
            print('Hello World!')
        '\n        normalize the word by removing diacritics, replacing hamzated Alif\n        with Alif replacing AlifMaqsura with Yaa and removing Waaw at the\n        beginning.\n        '
        token = self.re_diacritics.sub('', token)
        token = self.re_hamzated_alif.sub('ا', token)
        token = self.re_alifMaqsura.sub('ي', token)
        if token.startswith('و') and len(token) > 3:
            token = token[1:]
        return token

    def pref(self, token):
        if False:
            i = 10
            return i + 15
        "\n        remove prefixes from the words' beginning.\n        "
        if len(token) > 5:
            for p3 in self.pr3:
                if token.startswith(p3):
                    return token[3:]
        if len(token) > 6:
            for p4 in self.pr4:
                if token.startswith(p4):
                    return token[4:]
        if len(token) > 5:
            for p3 in self.pr32:
                if token.startswith(p3):
                    return token[3:]
        if len(token) > 4:
            for p2 in self.pr2:
                if token.startswith(p2):
                    return token[2:]

    def suff(self, token):
        if False:
            for i in range(10):
                print('nop')
        "\n        remove suffixes from the word's end.\n        "
        if token.endswith('ك') and len(token) > 3:
            return token[:-1]
        if len(token) > 4:
            for s2 in self.su2:
                if token.endswith(s2):
                    return token[:-2]
        if len(token) > 5:
            for s3 in self.su3:
                if token.endswith(s3):
                    return token[:-3]
        if token.endswith('ه') and len(token) > 3:
            token = token[:-1]
            return token
        if len(token) > 4:
            for s2 in self.su22:
                if token.endswith(s2):
                    return token[:-2]
        if len(token) > 5:
            for s3 in self.su32:
                if token.endswith(s3):
                    return token[:-3]
        if token.endswith('نا') and len(token) > 4:
            return token[:-2]
        return token

    def fem2masc(self, token):
        if False:
            while True:
                i = 10
        '\n        transform the word from the feminine form to the masculine form.\n        '
        if token.endswith('ة') and len(token) > 3:
            return token[:-1]

    def plur2sing(self, token):
        if False:
            i = 10
            return i + 15
        '\n        transform the word from the plural form to the singular form.\n        '
        if len(token) > 4:
            for ps2 in self.pl_si2:
                if token.endswith(ps2):
                    return token[:-2]
        if len(token) > 5:
            for ps3 in self.pl_si3:
                if token.endswith(ps3):
                    return token[:-3]
        if len(token) > 3 and token.endswith('ات'):
            return token[:-2]
        if len(token) > 3 and token.startswith('ا') and (token[2] == 'ا'):
            return token[:2] + token[3:]
        if len(token) > 4 and token.startswith('ا') and (token[-2] == 'ا'):
            return token[1:-2] + token[-1]

    def verb(self, token):
        if False:
            return 10
        '\n        stem the verb prefixes and suffixes or both\n        '
        vb = self.verb_t1(token)
        if vb is not None:
            return vb
        vb = self.verb_t2(token)
        if vb is not None:
            return vb
        vb = self.verb_t3(token)
        if vb is not None:
            return vb
        vb = self.verb_t4(token)
        if vb is not None:
            return vb
        vb = self.verb_t5(token)
        if vb is not None:
            return vb
        return self.verb_t6(token)

    def verb_t1(self, token):
        if False:
            while True:
                i = 10
        '\n        stem the present prefixes and suffixes\n        '
        if len(token) > 5 and token.startswith('ت'):
            for s2 in self.pl_si2:
                if token.endswith(s2):
                    return token[1:-2]
        if len(token) > 5 and token.startswith('ي'):
            for s2 in self.verb_su2:
                if token.endswith(s2):
                    return token[1:-2]
        if len(token) > 4 and token.startswith('ا'):
            if len(token) > 5 and token.endswith('وا'):
                return token[1:-2]
            if token.endswith('ي'):
                return token[1:-1]
            if token.endswith('ا'):
                return token[1:-1]
            if token.endswith('ن'):
                return token[1:-1]
        if len(token) > 4 and token.startswith('ي') and token.endswith('ن'):
            return token[1:-1]
        if len(token) > 4 and token.startswith('ت') and token.endswith('ن'):
            return token[1:-1]

    def verb_t2(self, token):
        if False:
            return 10
        '\n        stem the future prefixes and suffixes\n        '
        if len(token) > 6:
            for s2 in self.pl_si2:
                if token.startswith(self.verb_pr2[0]) and token.endswith(s2):
                    return token[2:-2]
            if token.startswith(self.verb_pr2[1]) and token.endswith(self.pl_si2[0]):
                return token[2:-2]
            if token.startswith(self.verb_pr2[1]) and token.endswith(self.pl_si2[2]):
                return token[2:-2]
        if len(token) > 5 and token.startswith(self.verb_pr2[0]) and token.endswith('ن'):
            return token[2:-1]
        if len(token) > 5 and token.startswith(self.verb_pr2[1]) and token.endswith('ن'):
            return token[2:-1]

    def verb_t3(self, token):
        if False:
            while True:
                i = 10
        '\n        stem the present suffixes\n        '
        if len(token) > 5:
            for su3 in self.verb_suf3:
                if token.endswith(su3):
                    return token[:-3]
        if len(token) > 4:
            for su2 in self.verb_suf2:
                if token.endswith(su2):
                    return token[:-2]
        if len(token) > 3:
            for su1 in self.verb_suf1:
                if token.endswith(su1):
                    return token[:-1]

    def verb_t4(self, token):
        if False:
            return 10
        '\n        stem the present prefixes\n        '
        if len(token) > 3:
            for pr1 in self.verb_suf1:
                if token.startswith(pr1):
                    return token[1:]
            if token.startswith('ي'):
                return token[1:]

    def verb_t5(self, token):
        if False:
            i = 10
            return i + 15
        '\n        stem the future prefixes\n        '
        if len(token) > 4:
            for pr2 in self.verb_pr22:
                if token.startswith(pr2):
                    return token[2:]
            for pr2 in self.verb_pr2:
                if token.startswith(pr2):
                    return token[2:]
        return token

    def verb_t6(self, token):
        if False:
            print('Hello World!')
        '\n        stem the order prefixes\n        '
        if len(token) > 4:
            for pr3 in self.verb_pr33:
                if token.startswith(pr3):
                    return token[2:]
        return token