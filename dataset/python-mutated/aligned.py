from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, concat, read_alignedsent_block
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.translate import AlignedSent, Alignment

class AlignedCorpusReader(CorpusReader):
    """
    Reader for corpora of word-aligned sentences.  Tokens are assumed
    to be separated by whitespace.  Sentences begin on separate lines.
    """

    def __init__(self, root, fileids, sep='/', word_tokenizer=WhitespaceTokenizer(), sent_tokenizer=RegexpTokenizer('\n', gaps=True), alignedsent_block_reader=read_alignedsent_block, encoding='latin1'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a new Aligned Corpus reader for a set of documents\n        located at the given root directory.  Example usage:\n\n            >>> root = '/...path to corpus.../'\n            >>> reader = AlignedCorpusReader(root, '.*', '.txt') # doctest: +SKIP\n\n        :param root: The root directory for this corpus.\n        :param fileids: A list or regexp specifying the fileids in this corpus.\n        "
        CorpusReader.__init__(self, root, fileids, encoding)
        self._sep = sep
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._alignedsent_block_reader = alignedsent_block_reader

    def words(self, fileids=None):
        if False:
            return 10
        '\n        :return: the given file(s) as a list of words\n            and punctuation symbols.\n        :rtype: list(str)\n        '
        return concat([AlignedSentCorpusView(fileid, enc, False, False, self._word_tokenizer, self._sent_tokenizer, self._alignedsent_block_reader) for (fileid, enc) in self.abspaths(fileids, True)])

    def sents(self, fileids=None):
        if False:
            return 10
        '\n        :return: the given file(s) as a list of\n            sentences or utterances, each encoded as a list of word\n            strings.\n        :rtype: list(list(str))\n        '
        return concat([AlignedSentCorpusView(fileid, enc, False, True, self._word_tokenizer, self._sent_tokenizer, self._alignedsent_block_reader) for (fileid, enc) in self.abspaths(fileids, True)])

    def aligned_sents(self, fileids=None):
        if False:
            print('Hello World!')
        '\n        :return: the given file(s) as a list of AlignedSent objects.\n        :rtype: list(AlignedSent)\n        '
        return concat([AlignedSentCorpusView(fileid, enc, True, True, self._word_tokenizer, self._sent_tokenizer, self._alignedsent_block_reader) for (fileid, enc) in self.abspaths(fileids, True)])

class AlignedSentCorpusView(StreamBackedCorpusView):
    """
    A specialized corpus view for aligned sentences.
    ``AlignedSentCorpusView`` objects are typically created by
    ``AlignedCorpusReader`` (not directly by nltk users).
    """

    def __init__(self, corpus_file, encoding, aligned, group_by_sent, word_tokenizer, sent_tokenizer, alignedsent_block_reader):
        if False:
            print('Hello World!')
        self._aligned = aligned
        self._group_by_sent = group_by_sent
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._alignedsent_block_reader = alignedsent_block_reader
        StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

    def read_block(self, stream):
        if False:
            while True:
                i = 10
        block = [self._word_tokenizer.tokenize(sent_str) for alignedsent_str in self._alignedsent_block_reader(stream) for sent_str in self._sent_tokenizer.tokenize(alignedsent_str)]
        if self._aligned:
            block[2] = Alignment.fromstring(' '.join(block[2]))
            block = [AlignedSent(*block)]
        elif self._group_by_sent:
            block = [block[0]]
        else:
            block = block[0]
        return block