"""
Indian Language POS-Tagged Corpus
Collected by A Kumaran, Microsoft Research, India
Distributed with permission

Contents:
  - Bangla: IIT Kharagpur
  - Hindi: Microsoft Research India
  - Marathi: IIT Bombay
  - Telugu: IIIT Hyderabad
"""
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple

class IndianCorpusReader(CorpusReader):
    """
    List of words, one per line.  Blank lines are ignored.
    """

    def words(self, fileids=None):
        if False:
            i = 10
            return i + 15
        return concat([IndianCorpusView(fileid, enc, False, False) for (fileid, enc) in self.abspaths(fileids, True)])

    def tagged_words(self, fileids=None, tagset=None):
        if False:
            print('Hello World!')
        if tagset and tagset != self._tagset:
            tag_mapping_function = lambda t: map_tag(self._tagset, tagset, t)
        else:
            tag_mapping_function = None
        return concat([IndianCorpusView(fileid, enc, True, False, tag_mapping_function) for (fileid, enc) in self.abspaths(fileids, True)])

    def sents(self, fileids=None):
        if False:
            while True:
                i = 10
        return concat([IndianCorpusView(fileid, enc, False, True) for (fileid, enc) in self.abspaths(fileids, True)])

    def tagged_sents(self, fileids=None, tagset=None):
        if False:
            return 10
        if tagset and tagset != self._tagset:
            tag_mapping_function = lambda t: map_tag(self._tagset, tagset, t)
        else:
            tag_mapping_function = None
        return concat([IndianCorpusView(fileid, enc, True, True, tag_mapping_function) for (fileid, enc) in self.abspaths(fileids, True)])

class IndianCorpusView(StreamBackedCorpusView):

    def __init__(self, corpus_file, encoding, tagged, group_by_sent, tag_mapping_function=None):
        if False:
            return 10
        self._tagged = tagged
        self._group_by_sent = group_by_sent
        self._tag_mapping_function = tag_mapping_function
        StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

    def read_block(self, stream):
        if False:
            i = 10
            return i + 15
        line = stream.readline()
        if line.startswith('<'):
            return []
        sent = [str2tuple(word, sep='_') for word in line.split()]
        if self._tag_mapping_function:
            sent = [(w, self._tag_mapping_function(t)) for (w, t) in sent]
        if not self._tagged:
            sent = [w for (w, t) in sent]
        if self._group_by_sent:
            return [sent]
        else:
            return sent