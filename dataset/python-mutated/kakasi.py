from calibre.ebooks.unihandecode.pykakasi.j2h import J2H
from calibre.ebooks.unihandecode.pykakasi.h2a import H2a
from calibre.ebooks.unihandecode.pykakasi.k2a import K2a

class kakasi:
    j2h = None
    h2a = None
    k2a = None

    def __init__(self):
        if False:
            print('Hello World!')
        self.j2h = J2H()
        self.h2a = H2a()
        self.k2a = K2a()

    def do(self, text):
        if False:
            print('Hello World!')
        otext = ''
        i = 0
        while True:
            if i >= len(text):
                break
            if self.j2h.isKanji(text[i]):
                (t, l) = self.j2h.convert(text[i:])
                if l <= 0:
                    otext = otext + text[i]
                    i = i + 1
                    continue
                i = i + l
                m = 0
                tmptext = ''
                while True:
                    if m >= len(t):
                        break
                    (s, n) = self.h2a.convert(t[m:])
                    if n <= 0:
                        break
                    m = m + n
                    tmptext = tmptext + s
                if i >= len(text):
                    otext = otext + tmptext.capitalize()
                else:
                    otext = otext + tmptext.capitalize() + ' '
            elif self.h2a.isHiragana(text[i]):
                tmptext = ''
                while True:
                    (t, l) = self.h2a.convert(text[i:])
                    tmptext = tmptext + t
                    i = i + l
                    if i >= len(text):
                        otext = otext + tmptext
                        break
                    elif not self.h2a.isHiragana(text[i]):
                        otext = otext + tmptext + ' '
                        break
            elif self.k2a.isKatakana(text[i]):
                tmptext = ''
                while True:
                    (t, l) = self.k2a.convert(text[i:])
                    tmptext = tmptext + t
                    i = i + l
                    if i >= len(text):
                        otext = otext + tmptext
                        break
                    elif not self.k2a.isKatakana(text[i]):
                        otext = otext + tmptext + ' '
                        break
            else:
                otext = otext + text[i]
                i += 1
        return otext