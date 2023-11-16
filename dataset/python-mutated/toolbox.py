"""
Module for reading, writing and manipulating
Toolbox databases and settings fileids.
"""
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.toolbox import ToolboxData

class ToolboxCorpusReader(CorpusReader):

    def xml(self, fileids, key=None):
        if False:
            for i in range(10):
                print('nop')
        return concat([ToolboxData(path, enc).parse(key=key) for (path, enc) in self.abspaths(fileids, True)])

    def fields(self, fileids, strip=True, unwrap=True, encoding='utf8', errors='strict', unicode_fields=None):
        if False:
            print('Hello World!')
        return concat([list(ToolboxData(fileid, enc).fields(strip, unwrap, encoding, errors, unicode_fields)) for (fileid, enc) in self.abspaths(fileids, include_encoding=True)])

    def entries(self, fileids, **kwargs):
        if False:
            return 10
        if 'key' in kwargs:
            key = kwargs['key']
            del kwargs['key']
        else:
            key = 'lx'
        entries = []
        for (marker, contents) in self.fields(fileids, **kwargs):
            if marker == key:
                entries.append((contents, []))
            else:
                try:
                    entries[-1][-1].append((marker, contents))
                except IndexError:
                    pass
        return entries

    def words(self, fileids, key='lx'):
        if False:
            while True:
                i = 10
        return [contents for (marker, contents) in self.fields(fileids) if marker == key]

def demo():
    if False:
        i = 10
        return i + 15
    pass
if __name__ == '__main__':
    demo()