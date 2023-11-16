"""
ARLSTem2 Arabic Light Stemmer
The details about the implementation of this algorithm are described in:
K. Abainia and H. Rebbani, Comparing the Effectiveness of the Improved ARLSTem
Algorithm with Existing Arabic Light Stemmers, International Conference on
Theoretical and Applicative Aspects of Computer Science (ICTAACS'19), Skikda,
Algeria, December 15-16, 2019.
ARLSTem2 is an Arabic light stemmer based on removing the affixes from
the words (i.e. prefixes, suffixes and infixes). It is an improvement
of the previous Arabic light stemmer (ARLSTem). The new version was compared to
the original algorithm and several existing Arabic light stemmers, where the
results showed that the new version considerably improves the under-stemming
errors that are common to light stemmers. Both ARLSTem and ARLSTem2 can be run
online and do not use any dictionary.
"""
import re
from nltk.stem.api import StemmerI

class ARLSTem2(StemmerI):
    """
    Return a stemmed Arabic word after removing affixes. This an improved
    version of the previous algorithm, which reduces under-stemming errors.
    Typically used in Arabic search engine, information retrieval and NLP.

        >>> from nltk.stem import arlstem2
        >>> stemmer = ARLSTem2()
        >>> word = stemmer.stem('يعمل')
        >>> print(word)
        عمل

    :param token: The input Arabic word (unicode) to be stemmed
    :type token: unicode
    :return: A unicode Arabic word
    """

    def __init__(self):
        if False:
            print('Hello World!')
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

    def stem1(self, token):
        if False:
            print('Hello World!')
        '\n        call this function to get the first stem\n        '
        try:
            if token is None:
                raise ValueError('The word could not be stemmed, because                                  it is empty !')
            self.is_verb = False
            token = self.norm(token)
            pre = self.pref(token)
            if pre is not None:
                token = pre
            fm = self.fem2masc(token)
            if fm is not None:
                return fm
            adj = self.adjective(token)
            if adj is not None:
                return adj
            token = self.suff(token)
            ps = self.plur2sing(token)
            if ps is None:
                if pre is None:
                    verb = self.verb(token)
                    if verb is not None:
                        self.is_verb = True
                        return verb
            else:
                return ps
            return token
        except ValueError as e:
            print(e)

    def stem(self, token):
        if False:
            while True:
                i = 10
        try:
            if token is None:
                raise ValueError('The word could not be stemmed, because                                  it is empty !')
            token = self.stem1(token)
            if len(token) > 4:
                if token.startswith('ت') and token[-2] == 'ي':
                    token = token[1:-2] + token[-1]
                    return token
                if token.startswith('م') and token[-2] == 'و':
                    token = token[1:-2] + token[-1]
                    return token
            if len(token) > 3:
                if not token.startswith('ا') and token.endswith('ي'):
                    token = token[:-1]
                    return token
                if token.startswith('ل'):
                    return token[1:]
            return token
        except ValueError as e:
            print(e)

    def norm(self, token):
        if False:
            return 10
        '\n        normalize the word by removing diacritics, replace hamzated Alif\n        with Alif bare, replace AlifMaqsura with Yaa and remove Waaw at the\n        beginning.\n        '
        token = self.re_diacritics.sub('', token)
        token = self.re_hamzated_alif.sub('ا', token)
        token = self.re_alifMaqsura.sub('ي', token)
        if token.startswith('و') and len(token) > 3:
            token = token[1:]
        return token

    def pref(self, token):
        if False:
            return 10
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

    def adjective(self, token):
        if False:
            return 10
        '\n        remove the infixes from adjectives\n        '
        if len(token) > 5:
            if token.startswith('ا') and token[-3] == 'ا' and token.endswith('ي'):
                return token[:-3] + token[-2]

    def suff(self, token):
        if False:
            print('Hello World!')
        "\n        remove the suffixes from the word's ending.\n        "
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
            for i in range(10):
                print('nop')
        '\n        transform the word from the feminine form to the masculine form.\n        '
        if len(token) > 6:
            if token.startswith('ت') and token[-4] == 'ي' and token.endswith('ية'):
                return token[1:-4] + token[-3]
            if token.startswith('ا') and token[-4] == 'ا' and token.endswith('ية'):
                return token[:-4] + token[-3]
        if token.endswith('اية') and len(token) > 5:
            return token[:-2]
        if len(token) > 4:
            if token[1] == 'ا' and token.endswith('ة'):
                return token[0] + token[2:-1]
            if token.endswith('ية'):
                return token[:-2]
        if token.endswith('ة') and len(token) > 3:
            return token[:-1]

    def plur2sing(self, token):
        if False:
            for i in range(10):
                print('nop')
        '\n        transform the word from the plural form to the singular form.\n        '
        if len(token) > 5:
            if token.startswith('م') and token.endswith('ون'):
                return token[1:-2]
        if len(token) > 4:
            for ps2 in self.pl_si2:
                if token.endswith(ps2):
                    return token[:-2]
        if len(token) > 5:
            for ps3 in self.pl_si3:
                if token.endswith(ps3):
                    return token[:-3]
        if len(token) > 4:
            if token.endswith('ات'):
                return token[:-2]
            if token.startswith('ا') and token[2] == 'ا':
                return token[:2] + token[3:]
            if token.startswith('ا') and token[-2] == 'ا':
                return token[1:-2] + token[-1]

    def verb(self, token):
        if False:
            print('Hello World!')
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
        vb = self.verb_t6(token)
        return vb

    def verb_t1(self, token):
        if False:
            return 10
        '\n        stem the present tense co-occurred prefixes and suffixes\n        '
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
            print('Hello World!')
        '\n        stem the future tense co-occurred prefixes and suffixes\n        '
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
            for i in range(10):
                print('nop')
        '\n        stem the present tense suffixes\n        '
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
            i = 10
            return i + 15
        '\n        stem the present tense prefixes\n        '
        if len(token) > 3:
            for pr1 in self.verb_suf1:
                if token.startswith(pr1):
                    return token[1:]
            if token.startswith('ي'):
                return token[1:]

    def verb_t5(self, token):
        if False:
            return 10
        '\n        stem the future tense prefixes\n        '
        if len(token) > 4:
            for pr2 in self.verb_pr22:
                if token.startswith(pr2):
                    return token[2:]
            for pr2 in self.verb_pr2:
                if token.startswith(pr2):
                    return token[2:]

    def verb_t6(self, token):
        if False:
            i = 10
            return i + 15
        '\n        stem the imperative tense prefixes\n        '
        if len(token) > 4:
            for pr3 in self.verb_pr33:
                if token.startswith(pr3):
                    return token[2:]
        return token