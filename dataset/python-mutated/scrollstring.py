from time import time

class scrollstring(object):

    def __init__(self, content, START):
        if False:
            for i in range(10):
                print('nop')
        self.content = content
        self.display = content
        self.START = START // 1
        self.update()

    def update(self):
        if False:
            print('Hello World!')
        self.display = self.content
        curTime = time() // 1
        offset = max(int((curTime - self.START) % len(self.content)) - 1, 0)
        while offset > 0:
            if self.display[0] > chr(127):
                offset -= 1
                self.display = self.display[1:] + self.display[:1]
            else:
                offset -= 1
                self.display = self.display[2:] + self.display[:2]

    def __repr__(self):
        if False:
            return 10
        return self.display

def truelen(string):
    if False:
        i = 10
        return i + 15
    "\n    It appears one Asian character takes two spots, but __len__\n    counts it as three, so this function counts the dispalyed\n    length of the string.\n\n    >>> truelen('abc')\n    3\n    >>> truelen('你好')\n    4\n    >>> truelen('1二3')\n    4\n    >>> truelen('')\n    0\n    "
    return len(string) + sum((1 for c in string if c > chr(127)))

def truelen_cut(string, length):
    if False:
        for i in range(10):
            print('nop')
    current_length = 0
    current_pos = 0
    for c in string:
        current_length += 2 if c > chr(127) else 1
        if current_length > length:
            return string[:current_pos]
        current_pos += 1
    return string