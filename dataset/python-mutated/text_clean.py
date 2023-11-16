import codecs
import re
import sys

class TextClean(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        spu = [160, 5760, 8239, 8287, 12288, 65279, 8203, 8206, 8207, 8298, 8300, 65279]
        spu.extend(range(57344, 63743 + 1))
        spu.extend(range(8192, 8202 + 1))
        spu.extend(range(127, 160 + 1))
        self.spaces = set([chr(i) for i in spu])
        self.space_pat = re.compile('\\s+', re.UNICODE)
        self.replace_char = {u'`': u"'", u'’': u"'", u'´': u"'", u'‘': u"'", u'º': u'°', u'–': u'-', u'—': u'-'}

    def sbc2dbc(self, ch):
        if False:
            while True:
                i = 10
        n = ord(ch)
        if 65280 < n < 65375:
            n -= 65248
        elif n == 12288:
            n = 32
        else:
            return ch
        return chr(n)

    def clean(self, s):
        if False:
            i = 10
            return i + 15
        try:
            line = list(s.strip())
            size = len(line)
            i = 0
            while i < size:
                if line[i] < u' ' or line[i] in self.spaces:
                    line[i] = u' '
                else:
                    line[i] = self.replace_char.get(line[i], line[i])
                    line[i] = self.sbc2dbc(line[i])
                i += 1
            line = ''.join(line)
            line = self.space_pat.sub(' ', line).strip()
            return line
        except Exception:
            return ''
if __name__ == '__main__':
    tc = TextClean()
    for line in sys.stdin:
        res = tc.clean(line)
        print(res)