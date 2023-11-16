"""Corpus in SVMlight format."""
from __future__ import with_statement
import logging
from gensim import utils
from gensim.corpora import IndexedCorpus
logger = logging.getLogger(__name__)

class SvmLightCorpus(IndexedCorpus):
    """Corpus in SVMlight format.

    Quoting http://svmlight.joachims.org/:
    The input file contains the training examples. The first lines  may contain comments and are ignored
    if they start with #. Each of the following lines represents one training example
    and is of the following format::

        <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
        <target> .=. +1 | -1 | 0 | <float>
        <feature> .=. <integer> | "qid"
        <value> .=. <float>
        <info> .=. <string>

    The "qid" feature (used for SVMlight ranking), if present, is ignored.

    Notes
    -----
    Although not mentioned in the specification above, SVMlight also expect its feature ids to be 1-based
    (counting starts at 1). We convert features to 0-base internally by decrementing all ids when loading a SVMlight
    input file, and increment them again when saving as SVMlight.

    """

    def __init__(self, fname, store_labels=True):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Parameters\n        ----------\n        fname: str\n            Path to corpus.\n        store_labels : bool, optional\n            Whether to store labels (~SVM target class). They currently have no application but stored\n            in `self.labels` for convenience by default.\n\n        '
        IndexedCorpus.__init__(self, fname)
        logger.info('loading corpus from %s', fname)
        self.fname = fname
        self.length = None
        self.store_labels = store_labels
        self.labels = []

    def __iter__(self):
        if False:
            print('Hello World!')
        ' Iterate over the corpus, returning one sparse (BoW) vector at a time.\n\n        Yields\n        ------\n        list of (int, float)\n            Document in BoW format.\n\n        '
        lineno = -1
        self.labels = []
        with utils.open(self.fname, 'rb') as fin:
            for (lineno, line) in enumerate(fin):
                doc = self.line2doc(line)
                if doc is not None:
                    if self.store_labels:
                        self.labels.append(doc[1])
                    yield doc[0]
        self.length = lineno + 1

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, labels=False, metadata=False):
        if False:
            for i in range(10):
                print('nop')
        'Save a corpus in the SVMlight format.\n\n        The SVMlight `<target>` class tag is taken from the `labels` array, or set to 0 for all documents\n        if `labels` is not supplied.\n\n        Parameters\n        ----------\n        fname : str\n            Path to output file.\n        corpus : iterable of iterable of (int, float)\n            Corpus in BoW format.\n        id2word : dict of (str, str), optional\n            Mapping id -> word.\n        labels : list or False\n            An SVMlight `<target>` class tags or False if not present.\n        metadata : bool\n            ARGUMENT WILL BE IGNORED.\n\n        Returns\n        -------\n        list of int\n            Offsets for each line in file (in bytes).\n\n        '
        logger.info('converting corpus to SVMlight format: %s', fname)
        if labels is not False:
            labels = list(labels)
        offsets = []
        with utils.open(fname, 'wb') as fout:
            for (docno, doc) in enumerate(corpus):
                label = labels[docno] if labels else 0
                offsets.append(fout.tell())
                fout.write(utils.to_utf8(SvmLightCorpus.doc2line(doc, label)))
        return offsets

    def docbyoffset(self, offset):
        if False:
            while True:
                i = 10
        "Get the document stored at file position `offset`.\n\n        Parameters\n        ----------\n        offset : int\n            Document's position.\n\n        Returns\n        -------\n        tuple of (int, float)\n\n        "
        with utils.open(self.fname, 'rb') as f:
            f.seek(offset)
            return self.line2doc(f.readline())[0]

    def line2doc(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Get a document from a single line in SVMlight format.\n        This method inverse of :meth:`~gensim.corpora.svmlightcorpus.SvmLightCorpus.doc2line`.\n\n        Parameters\n        ----------\n        line : str\n            Line in SVMLight format.\n\n        Returns\n        -------\n        (list of (int, float), str)\n            Document in BoW format and target class label.\n\n        '
        line = utils.to_unicode(line)
        line = line[:line.find('#')].strip()
        if not line:
            return None
        parts = line.split()
        if not parts:
            raise ValueError('invalid line format in %s' % self.fname)
        (target, fields) = (parts[0], [part.rsplit(':', 1) for part in parts[1:]])
        doc = [(int(p1) - 1, float(p2)) for (p1, p2) in fields if p1 != 'qid']
        return (doc, target)

    @staticmethod
    def doc2line(doc, label=0):
        if False:
            while True:
                i = 10
        'Convert BoW representation of document in SVMlight format.\n        This method inverse of :meth:`~gensim.corpora.svmlightcorpus.SvmLightCorpus.line2doc`.\n\n        Parameters\n        ----------\n        doc : list of (int, float)\n            Document in BoW format.\n        label : int, optional\n            Document label (if provided).\n\n        Returns\n        -------\n        str\n            `doc` in SVMlight format.\n\n        '
        pairs = ' '.join(('%i:%s' % (termid + 1, termval) for (termid, termval) in doc))
        return '%s %s\n' % (label, pairs)