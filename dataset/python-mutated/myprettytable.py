import sys
PY3 = sys.version_info[0] == 3

class IlegalUtf8(Exception):
    pass

def getfirst(c):
    if False:
        return 10
    if c >> 7 == 0:
        x = c
    elif c >> 5 == 6:
        x = c & 31
    elif c >> 4 == 14:
        x = c & 15
    elif c >> 3 == 30:
        x = c & 7
    elif c >> 2 == 62:
        x = c & 3
    elif c >> 1 == 126:
        x = c & 1
    else:
        raise IlegalUtf8
    return x
if not PY3:

    def UniIter(s):
        if False:
            for i in range(10):
                print('nop')
        if not s:
            return
        (x, uchar) = (getfirst(ord(s[0])), s[0])
        for ch in s[1:]:
            c = ord(ch)
            if c >> 6 == 2:
                x = x << 6 | c & 63
                uchar += ch
            else:
                yield (x, uchar)
                (x, uchar) = (getfirst(c), ch)
        else:
            yield (x, uchar)
else:

    def UniIter(s):
        if False:
            i = 10
            return i + 15
        for ch in s:
            yield (ord(ch), ch)

def calWidth(s, maxWidth=20):
    if False:
        while True:
            i = 10
    (rst, w) = ([], 0)
    try:
        for (x, uchar) in UniIter(s):
            if 32 <= x <= 126:
                w += 1
            elif 19968 <= x <= 40959:
                w += 2
            else:
                w += 1
                uchar = '*'
            rst.append(uchar)
            if w >= maxWidth:
                break
    except IlegalUtf8:
        w += 1
        rst.append('*')
    return (w, ''.join(rst))

class PrettyTable:

    def __init__(self, heads, maxWidth=20):
        if False:
            i = 10
            return i + 15
        self.numColumn = len(heads)
        self.maxWidth = maxWidth
        self.W = [-1] * self.numColumn
        self.M = []
        self.addRow(heads)

    def addRow(self, row):
        if False:
            print('Hello World!')
        r = []
        for j in range(self.numColumn):
            (w, s) = calWidth(row[j], self.maxWidth)
            if w > self.W[j]:
                self.W[j] = w
            r.append((w, s))
        self.M.append(r)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        level = ['+']
        for w in self.W:
            level.append('-' * (w + 2))
            level.append('+')
        level = ''.join(level)
        out = [level]
        for row in self.M:
            line = ['|']
            for j in range(self.numColumn):
                (w, s) = row[j]
                line.append(' ' + s + ' ' * (self.W[j] - w) + ' ')
                line.append('|')
            out.append(''.join(line))
            out.append(level)
        return '\n'.join(out)
if __name__ == '__main__':
    pt = PrettyTable(['city', 'name'])
    pt.addRow(['普通', '防静电啦发'])
    if not PY3:
        print(str(pt).decode('utf8'))
    else:
        print(pt)