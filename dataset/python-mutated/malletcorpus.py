"""Corpus in `Mallet format <http://mallet.cs.umass.edu/import.php>`_."""
from __future__ import with_statement
import logging
from gensim import utils
from gensim.corpora import LowCorpus
logger = logging.getLogger(__name__)

class MalletCorpus(LowCorpus):
    """Corpus handles input in `Mallet format <http://mallet.cs.umass.edu/import.php>`_.

    **Format description**

    One file, one instance per line, assume the data is in the following format ::

        [URL] [language] [text of the page...]

    Or, more generally, ::

        [document #1 id] [label] [text of the document...]
        [document #2 id] [label] [text of the document...]
        ...
        [document #N id] [label] [text of the document...]

    Note that language/label is *not* considered in Gensim, used `__unknown__` as default value.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import get_tmpfile, common_texts
        >>> from gensim.corpora import MalletCorpus
        >>> from gensim.corpora import Dictionary
        >>>
        >>> # Prepare needed data
        >>> dictionary = Dictionary(common_texts)
        >>> corpus = [dictionary.doc2bow(doc) for doc in common_texts]
        >>>
        >>> # Write corpus in Mallet format to disk
        >>> output_fname = get_tmpfile("corpus.mallet")
        >>> MalletCorpus.serialize(output_fname, corpus, dictionary)
        >>>
        >>> # Read corpus
        >>> loaded_corpus = MalletCorpus(output_fname)

    """

    def __init__(self, fname, id2word=None, metadata=False):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Parameters\n        ----------\n        fname : str\n            Path to file in Mallet format.\n        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional\n            Mapping between word_ids (integers) and words (strings).\n            If not provided, the mapping is constructed directly from `fname`.\n        metadata : bool, optional\n            If True, return additional information ("document id" and "lang" when you call\n            :meth:`~gensim.corpora.malletcorpus.MalletCorpus.line2doc`,\n            :meth:`~gensim.corpora.malletcorpus.MalletCorpus.__iter__` or\n            :meth:`~gensim.corpora.malletcorpus.MalletCorpus.docbyoffset`\n\n       '
        self.metadata = metadata
        LowCorpus.__init__(self, fname, id2word)

    def _calculate_num_docs(self):
        if False:
            return 10
        'Get number of documents.\n\n        Returns\n        -------\n        int\n            Number of documents in file.\n\n        '
        with utils.open(self.fname, 'rb') as fin:
            result = sum((1 for _ in fin))
        return result

    def __iter__(self):
        if False:
            return 10
        'Iterate over the corpus.\n\n        Yields\n        ------\n        list of (int, int)\n            Document in BoW format (+"document_id" and "lang" if metadata=True).\n\n        '
        with utils.open(self.fname, 'rb') as f:
            for line in f:
                yield self.line2doc(line)

    def line2doc(self, line):
        if False:
            print('Hello World!')
        'Covert line into document in BoW format.\n\n        Parameters\n        ----------\n        line : str\n            Line from input file.\n\n        Returns\n        -------\n        list of (int, int)\n            Document in BoW format (+"document_id" and "lang" if metadata=True).\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.corpora import MalletCorpus\n            >>>\n            >>> corpus = MalletCorpus(datapath("testcorpus.mallet"))\n            >>> corpus.line2doc("en computer human interface")\n            [(3, 1), (4, 1)]\n\n        '
        split_line = utils.to_unicode(line).strip().split(None, 2)
        (docid, doclang) = (split_line[0], split_line[1])
        words = split_line[2] if len(split_line) >= 3 else ''
        doc = super(MalletCorpus, self).line2doc(words)
        if self.metadata:
            return (doc, (docid, doclang))
        else:
            return doc

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        if False:
            for i in range(10):
                print('nop')
        "Save a corpus in the Mallet format.\n\n        Warnings\n        --------\n        This function is automatically called by :meth:`gensim.corpora.malletcorpus.MalletCorpus.serialize`,\n        don't call it directly, call :meth:`gensim.corpora.lowcorpus.malletcorpus.MalletCorpus.serialize` instead.\n\n        Parameters\n        ----------\n        fname : str\n            Path to output file.\n        corpus : iterable of iterable of (int, int)\n            Corpus in BoW format.\n        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional\n            Mapping between word_ids (integers) and words (strings).\n            If not provided, the mapping is constructed directly from `corpus`.\n        metadata : bool, optional\n            If True - ????\n\n        Return\n        ------\n        list of int\n            List of offsets in resulting file for each document (in bytes),\n            can be used for :meth:`~gensim.corpora.malletcorpus.Malletcorpus.docbyoffset`.\n\n        Notes\n        -----\n        The document id will be generated by enumerating the corpus.\n        That is, it will range between 0 and number of documents in the corpus.\n\n        Since Mallet has a language field in the format, this defaults to the string '__unknown__'.\n        If the language needs to be saved, post-processing will be required.\n\n        "
        if id2word is None:
            logger.info('no word id mapping provided; initializing from corpus')
            id2word = utils.dict_from_corpus(corpus)
        logger.info('storing corpus in Mallet format into %s', fname)
        truncated = 0
        offsets = []
        with utils.open(fname, 'wb') as fout:
            for (doc_id, doc) in enumerate(corpus):
                if metadata:
                    (doc_id, doc_lang) = doc[1]
                    doc = doc[0]
                else:
                    doc_lang = '__unknown__'
                words = []
                for (wordid, value) in doc:
                    if abs(int(value) - value) > 1e-06:
                        truncated += 1
                    words.extend([utils.to_unicode(id2word[wordid])] * int(value))
                offsets.append(fout.tell())
                fout.write(utils.to_utf8('%s %s %s\n' % (doc_id, doc_lang, ' '.join(words))))
        if truncated:
            logger.warning('Mallet format can only save vectors with integer elements; %i float entries were truncated to integer value', truncated)
        return offsets

    def docbyoffset(self, offset):
        if False:
            i = 10
            return i + 15
        'Get the document stored in file by `offset` position.\n\n        Parameters\n        ----------\n        offset : int\n            Offset (in bytes) to begin of document.\n\n        Returns\n        -------\n        list of (int, int)\n            Document in BoW format (+"document_id" and "lang" if metadata=True).\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.corpora import MalletCorpus\n            >>>\n            >>> data = MalletCorpus(datapath("testcorpus.mallet"))\n            >>> data.docbyoffset(1)  # end of first line\n            [(3, 1), (4, 1)]\n            >>> data.docbyoffset(4)  # start of second line\n            [(4, 1)]\n\n        '
        with utils.open(self.fname, 'rb') as f:
            f.seek(offset)
            return self.line2doc(f.readline())