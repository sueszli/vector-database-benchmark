class Chunk(object):
    pass

class TagChunk(Chunk):
    __slots__ = ('tag', 'label')

    def __init__(self, tag: str, label: str=None):
        if False:
            return 10
        self.tag = tag
        self.label = label

    def __str__(self):
        if False:
            return 10
        if self.label is None:
            return self.tag
        else:
            return self.label + ':' + self.tag

class TextChunk(Chunk):
    __slots__ = 'text'

    def __init__(self, text: str):
        if False:
            for i in range(10):
                print('nop')
        self.text = text

    def __str__(self):
        if False:
            while True:
                i = 10
        return "'" + self.text + "'"