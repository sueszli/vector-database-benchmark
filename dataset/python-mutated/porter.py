"""Porter Stemming Algorithm
This is the Porter stemming algorithm, ported to Python from the
version coded up in ANSI C by the author. It may be be regarded
as canonical, in that it follows the algorithm presented in [1]_, see also [2]_

Author - Vivake Gupta (v@nano.com), optimizations and cleanup of the code by Lars Buitinck.

Examples
--------

.. sourcecode:: pycon

    >>> from gensim.parsing.porter import PorterStemmer
    >>>
    >>> p = PorterStemmer()
    >>> p.stem("apple")
    'appl'
    >>>
    >>> p.stem_sentence("Cats and ponies have meeting")
    'cat and poni have meet'
    >>>
    >>> p.stem_documents(["Cats and ponies", "have meeting"])
    ['cat and poni', 'have meet']

.. [1] Porter, 1980, An algorithm for suffix stripping, http://www.cs.odu.edu/~jbollen/IR04/readings/readings5.pdf
.. [2] http://www.tartarus.org/~martin/PorterStemmer

"""

class PorterStemmer:
    """Class contains implementation of Porter stemming algorithm.

    Attributes
    --------
    b : str
        Buffer holding a word to be stemmed. The letters are in b[0], b[1] ... ending at b[`k`].
    k : int
        Readjusted downwards as the stemming progresses.
    j : int
        Word length.

    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.b = ''
        self.k = 0
        self.j = 0

    def _cons(self, i):
        if False:
            print('Hello World!')
        'Check if b[i] is a consonant letter.\n\n        Parameters\n        ----------\n        i : int\n            Index for `b`.\n\n        Returns\n        -------\n        bool\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.parsing.porter import PorterStemmer\n            >>> p = PorterStemmer()\n            >>> p.b = "hi"\n            >>> p._cons(1)\n            False\n            >>> p.b = "meow"\n            >>> p._cons(3)\n            True\n\n        '
        ch = self.b[i]
        if ch in 'aeiou':
            return False
        if ch == 'y':
            return i == 0 or not self._cons(i - 1)
        return True

    def _m(self):
        if False:
            i = 10
            return i + 15
        'Calculate the number of consonant sequences between 0 and j.\n\n        If c is a consonant sequence and v a vowel sequence, and <..>\n        indicates arbitrary presence,\n\n           <c><v>       gives 0\n           <c>vc<v>     gives 1\n           <c>vcvc<v>   gives 2\n           <c>vcvcvc<v> gives 3\n\n        Returns\n        -------\n        int\n            The number of consonant sequences between 0 and j.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.parsing.porter import PorterStemmer\n            >>> p = PorterStemmer()\n            >>> p.b = "<bm>aobm<ao>"\n            >>> p.j = 11\n            >>> p._m()\n            2\n\n        '
        i = 0
        while True:
            if i > self.j:
                return 0
            if not self._cons(i):
                break
            i += 1
        i += 1
        n = 0
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while 1:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if b[0: j + 1] contains a vowel letter.\n\n        Returns\n        -------\n        bool\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.parsing.porter import PorterStemmer\n            >>> p = PorterStemmer()\n            >>> p.b = "gnsm"\n            >>> p.j = 3\n            >>> p._vowelinstem()\n            False\n            >>> p.b = "gensim"\n            >>> p.j = 5\n            >>> p._vowelinstem()\n            True\n\n        '
        return not all((self._cons(i) for i in range(self.j + 1)))

    def _doublec(self, j):
        if False:
            for i in range(10):
                print('nop')
        'Check if b[j - 1: j + 1] contain a double consonant letter.\n\n        Parameters\n        ----------\n        j : int\n            Index for `b`\n\n        Returns\n        -------\n        bool\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.parsing.porter import PorterStemmer\n            >>> p = PorterStemmer()\n            >>> p.b = "real"\n            >>> p.j = 3\n            >>> p._doublec(3)\n            False\n            >>> p.b = "really"\n            >>> p.j = 5\n            >>> p._doublec(4)\n            True\n\n        '
        return j > 0 and self.b[j] == self.b[j - 1] and self._cons(j)

    def _cvc(self, i):
        if False:
            return 10
        'Check if b[j - 2: j + 1] makes the (consonant, vowel, consonant) pattern and also\n        if the second \'c\' is not \'w\', \'x\' or \'y\'. This is used when trying to restore an \'e\' at the end of a short word,\n        e.g. cav(e), lov(e), hop(e), crim(e), but snow, box, tray.\n\n        Parameters\n        ----------\n        i : int\n            Index for `b`\n\n        Returns\n        -------\n        bool\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.parsing.porter import PorterStemmer\n            >>> p = PorterStemmer()\n            >>> p.b = "lib"\n            >>> p.j = 2\n            >>> p._cvc(2)\n            True\n            >>> p.b = "dll"\n            >>> p.j = 2\n            >>> p._cvc(2)\n            False\n            >>> p.b = "wow"\n            >>> p.j = 2\n            >>> p._cvc(2)\n            False\n\n        '
        if i < 2 or not self._cons(i) or self._cons(i - 1) or (not self._cons(i - 2)):
            return False
        return self.b[i] not in 'wxy'

    def _ends(self, s):
        if False:
            print('Hello World!')
        'Check if b[: k + 1] ends with `s`.\n\n        Parameters\n        ----------\n        s : str\n\n        Returns\n        -------\n        bool\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.parsing.porter import PorterStemmer\n            >>> p = PorterStemmer()\n            >>> p.b = "cowboy"\n            >>> p.j = 5\n            >>> p.k = 2\n            >>> p._ends("cow")\n            True\n\n        '
        if s[-1] != self.b[self.k]:
            return False
        length = len(s)
        if length > self.k + 1:
            return False
        if self.b[self.k - length + 1:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        if False:
            i = 10
            return i + 15
        'Append `s` to `b`, adjusting `k`.\n\n        Parameters\n        ----------\n        s : str\n\n        '
        self.b = self.b[:self.j + 1] + s
        self.k = len(self.b) - 1

    def _r(self, s):
        if False:
            i = 10
            return i + 15
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self):
        if False:
            print('Hello World!')
        'Get rid of plurals and -ed or -ing.\n\n           caresses  ->  caress\n           ponies    ->  poni\n           ties      ->  ti\n           caress    ->  caress\n           cats      ->  cat\n\n           feed      ->  feed\n           agreed    ->  agree\n           disabled  ->  disable\n\n           matting   ->  mat\n           mating    ->  mate\n           meeting   ->  meet\n           milling   ->  mill\n           messing   ->  mess\n\n           meetings  ->  meet\n\n        '
        if self.b[self.k] == 's':
            if self._ends('sses'):
                self.k -= 2
            elif self._ends('ies'):
                self._setto('i')
            elif self.b[self.k - 1] != 's':
                self.k -= 1
        if self._ends('eed'):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends('ed') or self._ends('ing')) and self._vowelinstem():
            self.k = self.j
            if self._ends('at'):
                self._setto('ate')
            elif self._ends('bl'):
                self._setto('ble')
            elif self._ends('iz'):
                self._setto('ize')
            elif self._doublec(self.k):
                if self.b[self.k - 1] not in 'lsz':
                    self.k -= 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto('e')

    def _step1c(self):
        if False:
            while True:
                i = 10
        "Turn terminal 'y' to 'i' when there is another vowel in the stem."
        if self._ends('y') and self._vowelinstem():
            self.b = self.b[:self.k] + 'i'

    def _step2(self):
        if False:
            i = 10
            return i + 15
        'Map double suffices to single ones.\n\n        So, -ization ( = -ize plus -ation) maps to -ize etc. Note that the\n        string before the suffix must give _m() > 0.\n\n        '
        ch = self.b[self.k - 1]
        if ch == 'a':
            if self._ends('ational'):
                self._r('ate')
            elif self._ends('tional'):
                self._r('tion')
        elif ch == 'c':
            if self._ends('enci'):
                self._r('ence')
            elif self._ends('anci'):
                self._r('ance')
        elif ch == 'e':
            if self._ends('izer'):
                self._r('ize')
        elif ch == 'l':
            if self._ends('bli'):
                self._r('ble')
            elif self._ends('alli'):
                self._r('al')
            elif self._ends('entli'):
                self._r('ent')
            elif self._ends('eli'):
                self._r('e')
            elif self._ends('ousli'):
                self._r('ous')
        elif ch == 'o':
            if self._ends('ization'):
                self._r('ize')
            elif self._ends('ation'):
                self._r('ate')
            elif self._ends('ator'):
                self._r('ate')
        elif ch == 's':
            if self._ends('alism'):
                self._r('al')
            elif self._ends('iveness'):
                self._r('ive')
            elif self._ends('fulness'):
                self._r('ful')
            elif self._ends('ousness'):
                self._r('ous')
        elif ch == 't':
            if self._ends('aliti'):
                self._r('al')
            elif self._ends('iviti'):
                self._r('ive')
            elif self._ends('biliti'):
                self._r('ble')
        elif ch == 'g':
            if self._ends('logi'):
                self._r('log')

    def _step3(self):
        if False:
            for i in range(10):
                print('nop')
        'Deal with -ic-, -full, -ness etc. Similar strategy to _step2.'
        ch = self.b[self.k]
        if ch == 'e':
            if self._ends('icate'):
                self._r('ic')
            elif self._ends('ative'):
                self._r('')
            elif self._ends('alize'):
                self._r('al')
        elif ch == 'i':
            if self._ends('iciti'):
                self._r('ic')
        elif ch == 'l':
            if self._ends('ical'):
                self._r('ic')
            elif self._ends('ful'):
                self._r('')
        elif ch == 's':
            if self._ends('ness'):
                self._r('')

    def _step4(self):
        if False:
            print('Hello World!')
        'Takes off -ant, -ence etc., in context <c>vcvc<v>.'
        ch = self.b[self.k - 1]
        if ch == 'a':
            if not self._ends('al'):
                return
        elif ch == 'c':
            if not self._ends('ance') and (not self._ends('ence')):
                return
        elif ch == 'e':
            if not self._ends('er'):
                return
        elif ch == 'i':
            if not self._ends('ic'):
                return
        elif ch == 'l':
            if not self._ends('able') and (not self._ends('ible')):
                return
        elif ch == 'n':
            if self._ends('ant'):
                pass
            elif self._ends('ement'):
                pass
            elif self._ends('ment'):
                pass
            elif self._ends('ent'):
                pass
            else:
                return
        elif ch == 'o':
            if self._ends('ion') and self.b[self.j] in 'st':
                pass
            elif self._ends('ou'):
                pass
            else:
                return
        elif ch == 's':
            if not self._ends('ism'):
                return
        elif ch == 't':
            if not self._ends('ate') and (not self._ends('iti')):
                return
        elif ch == 'u':
            if not self._ends('ous'):
                return
        elif ch == 'v':
            if not self._ends('ive'):
                return
        elif ch == 'z':
            if not self._ends('ize'):
                return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _step5(self):
        if False:
            i = 10
            return i + 15
        'Remove a final -e if _m() > 1, and change -ll to -l if m() > 1.'
        k = self.j = self.k
        if self.b[k] == 'e':
            a = self._m()
            if a > 1 or (a == 1 and (not self._cvc(k - 1))):
                self.k -= 1
        if self.b[self.k] == 'l' and self._doublec(self.k) and (self._m() > 1):
            self.k -= 1

    def stem(self, w):
        if False:
            print('Hello World!')
        'Stem the word `w`.\n\n        Parameters\n        ----------\n        w : str\n\n        Returns\n        -------\n        str\n            Stemmed version of `w`.\n\n        Examples\n        --------\n\n        .. sourcecode:: pycon\n\n            >>> from gensim.parsing.porter import PorterStemmer\n            >>> p = PorterStemmer()\n            >>> p.stem("ponies")\n            \'poni\'\n\n        '
        w = w.lower()
        k = len(w) - 1
        if k <= 1:
            return w
        self.b = w
        self.k = k
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[:self.k + 1]

    def stem_sentence(self, txt):
        if False:
            for i in range(10):
                print('nop')
        'Stem the sentence `txt`.\n\n        Parameters\n        ----------\n        txt : str\n            Input sentence.\n\n        Returns\n        -------\n        str\n            Stemmed sentence.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.parsing.porter import PorterStemmer\n            >>> p = PorterStemmer()\n            >>> p.stem_sentence("Wow very nice woman with apple")\n            \'wow veri nice woman with appl\'\n\n        '
        return ' '.join((self.stem(x) for x in txt.split()))

    def stem_documents(self, docs):
        if False:
            print('Hello World!')
        'Stem documents.\n\n        Parameters\n        ----------\n        docs : list of str\n            Input documents\n\n        Returns\n        -------\n        list of str\n            Stemmed documents.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.parsing.porter import PorterStemmer\n            >>> p = PorterStemmer()\n            >>> p.stem_documents(["Have a very nice weekend", "Have a very nice weekend"])\n            [\'have a veri nice weekend\', \'have a veri nice weekend\']\n\n        '
        return [self.stem_sentence(x) for x in docs]
if __name__ == '__main__':
    import sys
    p = PorterStemmer()
    for f in sys.argv[1:]:
        with open(f) as infile:
            for line in infile:
                print(p.stem_sentence(line))