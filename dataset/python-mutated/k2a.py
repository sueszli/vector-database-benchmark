from calibre.ebooks.unihandecode.pykakasi.jisyo import jisyo

class K2a:
    kanwa = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.kanwa = jisyo()

    def isKatakana(self, char):
        if False:
            while True:
                i = 10
        return 12448 < ord(char) and ord(char) < 12535

    def convert(self, text):
        if False:
            i = 10
            return i + 15
        Hstr = ''
        max_len = -1
        r = min(10, len(text) + 1)
        for x in range(r):
            if text[:x] in self.kanwa.kanadict:
                if max_len < x:
                    max_len = x
                    Hstr = self.kanwa.kanadict[text[:x]]
        return (Hstr, max_len)