"""Сorpus in Blei's LDA-C format."""
from __future__ import with_statement
from os import path
import logging
from gensim import utils
from gensim.corpora import IndexedCorpus
logger = logging.getLogger(__name__)

class BleiCorpus(IndexedCorpus):
    """Corpus in Blei's LDA-C format.

    The corpus is represented as two files: one describing the documents, and another
    describing the mapping between words and their ids.

    Each document is one line::

        N fieldId1:fieldValue1 fieldId2:fieldValue2 ... fieldIdN:fieldValueN


    The vocabulary is a file with words, one word per line; word at line K has an implicit `id=K`.

    """

    def __init__(self, fname, fname_vocab=None):
        if False:
            print('Hello World!')
        "\n\n        Parameters\n        ----------\n        fname : str\n            Path to corpus.\n        fname_vocab : str, optional\n            Vocabulary file. If `fname_vocab` is None, searching one of variants:\n\n            * `fname`.vocab\n            * `fname`/vocab.txt\n            * `fname_without_ext`.vocab\n            * `fname_folder`/vocab.txt\n\n        Raises\n        ------\n        IOError\n            If vocabulary file doesn't exist.\n\n        "
        IndexedCorpus.__init__(self, fname)
        logger.info('loading corpus from %s', fname)
        if fname_vocab is None:
            (fname_base, _) = path.splitext(fname)
            fname_dir = path.dirname(fname)
            for fname_vocab in [utils.smart_extension(fname, '.vocab'), utils.smart_extension(fname, '/vocab.txt'), utils.smart_extension(fname_base, '.vocab'), utils.smart_extension(fname_dir, '/vocab.txt')]:
                if path.exists(fname_vocab):
                    break
            else:
                raise IOError('BleiCorpus: could not find vocabulary file')
        self.fname = fname
        with utils.open(fname_vocab, 'rb') as fin:
            words = [utils.to_unicode(word).rstrip() for word in fin]
        self.id2word = dict(enumerate(words))

    def __iter__(self):
        if False:
            return 10
        "Iterate over the corpus, returning one sparse (BoW) vector at a time.\n\n        Yields\n        ------\n        list of (int, float)\n            Document's BoW representation.\n\n        "
        lineno = -1
        with utils.open(self.fname, 'rb') as fin:
            for (lineno, line) in enumerate(fin):
                yield self.line2doc(line)
        self.length = lineno + 1

    def line2doc(self, line):
        if False:
            while True:
                i = 10
        "Convert line in Blei LDA-C format to document (BoW representation).\n\n        Parameters\n        ----------\n        line : str\n            Line in Blei's LDA-C format.\n\n        Returns\n        -------\n        list of (int, float)\n            Document's BoW representation.\n\n        "
        parts = utils.to_unicode(line).split()
        if int(parts[0]) != len(parts) - 1:
            raise ValueError('invalid format in %s: %s' % (self.fname, repr(line)))
        doc = [part.rsplit(':', 1) for part in parts[1:]]
        doc = [(int(p1), float(p2)) for (p1, p2) in doc]
        return doc

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        if False:
            return 10
        'Save a corpus in the LDA-C format.\n\n        Notes\n        -----\n        There are actually two files saved: `fname` and `fname.vocab`, where `fname.vocab` is the vocabulary file.\n\n        Parameters\n        ----------\n        fname : str\n            Path to output file.\n        corpus : iterable of iterable of (int, float)\n            Input corpus in BoW format.\n        id2word : dict of (str, str), optional\n            Mapping id -> word for `corpus`.\n        metadata : bool, optional\n            THIS PARAMETER WILL BE IGNORED.\n\n        Returns\n        -------\n        list of int\n            Offsets for each line in file (in bytes).\n\n        '
        if id2word is None:
            logger.info('no word id mapping provided; initializing from corpus')
            id2word = utils.dict_from_corpus(corpus)
            num_terms = len(id2word)
        elif id2word:
            num_terms = 1 + max(id2word)
        else:
            num_terms = 0
        logger.info("storing corpus in Blei's LDA-C format into %s", fname)
        with utils.open(fname, 'wb') as fout:
            offsets = []
            for doc in corpus:
                doc = list(doc)
                offsets.append(fout.tell())
                parts = ['%i:%g' % p for p in doc if abs(p[1]) > 1e-07]
                fout.write(utils.to_utf8('%i %s\n' % (len(doc), ' '.join(parts))))
        fname_vocab = utils.smart_extension(fname, '.vocab')
        logger.info('saving vocabulary of %i words to %s', num_terms, fname_vocab)
        with utils.open(fname_vocab, 'wb') as fout:
            for featureid in range(num_terms):
                fout.write(utils.to_utf8('%s\n' % id2word.get(featureid, '---')))
        return offsets

    def docbyoffset(self, offset):
        if False:
            for i in range(10):
                print('nop')
        'Get document corresponding to `offset`.\n        Offset can be given from :meth:`~gensim.corpora.bleicorpus.BleiCorpus.save_corpus`.\n\n        Parameters\n        ----------\n        offset : int\n            Position of the document in the file (in bytes).\n\n        Returns\n        -------\n        list of (int, float)\n            Document in BoW format.\n\n        '
        with utils.open(self.fname, 'rb') as f:
            f.seek(offset)
            return self.line2doc(f.readline())