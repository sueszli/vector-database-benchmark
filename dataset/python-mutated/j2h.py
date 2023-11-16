import re
from calibre.ebooks.unihandecode.pykakasi.jisyo import jisyo
from polyglot.builtins import iteritems

class J2H:
    kanwa = None
    cl_table = ['', 'aiueow', 'aiueow', 'aiueow', 'aiueow', 'aiueow', 'aiueow', 'aiueow', 'aiueow', 'aiueow', 'aiueow', 'k', 'g', 'k', 'g', 'k', 'g', 'k', 'g', 'k', 'g', 's', 'zj', 's', 'zj', 's', 'zj', 's', 'zj', 's', 'zj', 't', 'd', 'tc', 'd', 'aiueokstchgzjfdbpw', 't', 'd', 't', 'd', 't', 'd', 'n', 'n', 'n', 'n', 'n', 'h', 'b', 'p', 'h', 'b', 'p', 'hf', 'b', 'p', 'h', 'b', 'p', 'h', 'b', 'p', 'm', 'm', 'm', 'm', 'm', 'y', 'y', 'y', 'y', 'y', 'y', 'rl', 'rl', 'rl', 'rl', 'rl', 'wiueo', 'wiueo', 'wiueo', 'wiueo', 'w', 'n', 'v', 'k', 'k', '', '', '', '', '', '', '', '', '']

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.kanwa = jisyo()

    def isKanji(self, c):
        if False:
            while True:
                i = 10
        return 13312 <= ord(c) and ord(c) < 64046

    def isCletter(self, l, c):
        if False:
            print('Hello World!')
        if (ord('ぁ') <= ord(c) and ord(c) <= 12447) and l in self.cl_table[ord(c) - ord('ぁ') - 1]:
            return True
        return False

    def itaiji_conv(self, text):
        if False:
            while True:
                i = 10
        r = []
        for c in text:
            if c in self.kanwa.itaijidict:
                r.append(c)
        for c in r:
            text = re.sub(c, self.kanwa.itaijidict[c], text)
        return text

    def convert(self, text):
        if False:
            while True:
                i = 10
        max_len = 0
        Hstr = ''
        table = self.kanwa.load_jisyo(text[0])
        if table is None:
            return ('', 0)
        for (k, v) in iteritems(table):
            length = len(k)
            if len(text) >= length:
                if text.startswith(k):
                    for (yomi, tail) in v:
                        if tail == '':
                            if max_len < length:
                                Hstr = yomi
                                max_len = length
                        elif max_len < length + 1 and len(text) > length and self.isCletter(tail, text[length]):
                            Hstr = ''.join([yomi, text[length]])
                            max_len = length + 1
        return (Hstr, max_len)