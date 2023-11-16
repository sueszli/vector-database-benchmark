"""
Read tuples from a corpus consisting of categorized strings.
For example, from the question classification corpus:

NUM:dist How far is it from Denver to Aspen ?
LOC:city What county is Modesto , California in ?
HUM:desc Who was Galileo ?
DESC:def What is an atom ?
NUM:date When did Hawaii become a state ?
"""
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *

class StringCategoryCorpusReader(CorpusReader):

    def __init__(self, root, fileids, delimiter=' ', encoding='utf8'):
        if False:
            while True:
                i = 10
        '\n        :param root: The root directory for this corpus.\n        :param fileids: A list or regexp specifying the fileids in this corpus.\n        :param delimiter: Field delimiter\n        '
        CorpusReader.__init__(self, root, fileids, encoding)
        self._delimiter = delimiter

    def tuples(self, fileids=None):
        if False:
            for i in range(10):
                print('nop')
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat([StreamBackedCorpusView(fileid, self._read_tuple_block, encoding=enc) for (fileid, enc) in self.abspaths(fileids, True)])

    def _read_tuple_block(self, stream):
        if False:
            while True:
                i = 10
        line = stream.readline().strip()
        if line:
            return [tuple(line.split(self._delimiter, 1))]
        else:
            return []