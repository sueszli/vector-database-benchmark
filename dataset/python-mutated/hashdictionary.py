"""Implements the `"hashing trick" <https://en.wikipedia.org/wiki/Hashing-Trick>`_ -- a mapping between words
and their integer ids using a fixed, static mapping (hash function).

Notes
-----

The static mapping has a constant memory footprint, regardless of the number of word-types (features) in your corpus,
so it's suitable for processing extremely large corpora. The ids are computed as `hash(word) %% id_range`,
where `hash` is a user-configurable function (`zlib.adler32` by default).

Advantages:

* New words can be represented immediately, without an extra pass through the corpus
  to collect all the ids first.
* Can be used with non-repeatable (once-only) streams of documents.
* Able to represent any token (not only those present in training documents)

Disadvantages:

* Multiple words may map to the same id, causing hash collisions. The word <-> id mapping is no longer a bijection.

"""
import logging
import itertools
import zlib
from gensim import utils
logger = logging.getLogger(__name__)

class HashDictionary(utils.SaveLoad, dict):
    """Mapping between words and their integer ids, using a hashing function.

    Unlike :class:`~gensim.corpora.dictionary.Dictionary`,
    building a :class:`~gensim.corpora.hashdictionary.HashDictionary` before using it **isn't a necessary step**.

    You can start converting words to ids immediately, without training on a corpus.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora import HashDictionary
        >>>
        >>> dct = HashDictionary(debug=False)  # needs no training corpus!
        >>>
        >>> texts = [['human', 'interface', 'computer']]
        >>> dct.doc2bow(texts[0])
        [(10608, 1), (12466, 1), (31002, 1)]

    """

    def __init__(self, documents=None, id_range=32000, myhash=zlib.adler32, debug=True):
        if False:
            while True:
                i = 10
        '\n\n        Parameters\n        ----------\n        documents : iterable of iterable of str, optional\n            Iterable of documents. If given, used to collect additional corpus statistics.\n            :class:`~gensim.corpora.hashdictionary.HashDictionary` can work\n            without these statistics (optional parameter).\n        id_range : int, optional\n            Number of hash-values in table, used as `id = myhash(key) %% id_range`.\n        myhash : function, optional\n            Hash function, should support interface `myhash(str) -> int`, uses `zlib.adler32` by default.\n        debug : bool, optional\n            Store which tokens have mapped to a given id? **Will use a lot of RAM**.\n            If you find yourself running out of memory (or not sure that you really need raw tokens),\n            keep `debug=False`.\n\n        '
        self.myhash = myhash
        self.id_range = id_range
        self.debug = debug
        self.token2id = {}
        self.id2token = {}
        self.dfs = {}
        self.dfs_debug = {}
        self.num_docs = 0
        self.num_pos = 0
        self.num_nnz = 0
        self.allow_update = True
        if documents is not None:
            self.add_documents(documents)

    def __getitem__(self, tokenid):
        if False:
            for i in range(10):
                print('nop')
        'Get all words that have mapped to the given id so far, as a set.\n\n        Warnings\n        --------\n        Works only if you initialized your :class:`~gensim.corpora.hashdictionary.HashDictionary` object\n        with `debug=True`.\n\n        Parameters\n        ----------\n        tokenid : int\n            Token identifier (result of hashing).\n\n        Return\n        ------\n        set of str\n            Set of all words that have mapped to this id.\n\n        '
        return self.id2token.get(tokenid, set())

    def restricted_hash(self, token):
        if False:
            print('Hello World!')
        'Calculate id of the given token.\n        Also keep track of what words were mapped to what ids, if `debug=True` was set in the constructor.\n\n        Parameters\n        ----------\n        token : str\n            Input token.\n\n        Return\n        ------\n        int\n            Hash value of `token`.\n\n        '
        h = self.myhash(utils.to_utf8(token)) % self.id_range
        if self.debug:
            self.token2id[token] = h
            self.id2token.setdefault(h, set()).add(token)
        return h

    def __len__(self):
        if False:
            print('Hello World!')
        'Get the number of distinct ids = the entire dictionary size.'
        return self.id_range

    def keys(self):
        if False:
            while True:
                i = 10
        'Get a list of all token ids.'
        return range(len(self))

    def __str__(self):
        if False:
            print('Hello World!')
        return 'HashDictionary(%i id range)' % len(self)

    @staticmethod
    def from_documents(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return HashDictionary(*args, **kwargs)

    def add_documents(self, documents):
        if False:
            return 10
        'Collect corpus statistics from a corpus.\n\n        Warnings\n        --------\n        Useful only if `debug=True`, to build the reverse `id=>set(words)` mapping.\n\n        Notes\n        -----\n        This is only a convenience wrapper for calling `doc2bow` on each document with `allow_update=True`.\n\n        Parameters\n        ----------\n        documents : iterable of list of str\n            Collection of documents.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora import HashDictionary\n            >>>\n            >>> dct = HashDictionary(debug=True)  # needs no training corpus!\n            >>>\n            >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]\n            >>> "sparta" in dct.token2id\n            False\n            >>> dct.add_documents([["this", "is", "sparta"], ["just", "joking"]])\n            >>> "sparta" in dct.token2id\n            True\n\n        '
        for (docno, document) in enumerate(documents):
            if docno % 10000 == 0:
                logger.info('adding document #%i to %s', docno, self)
            self.doc2bow(document, allow_update=True)
        logger.info('built %s from %i documents (total %i corpus positions)', self, self.num_docs, self.num_pos)

    def doc2bow(self, document, allow_update=False, return_missing=False):
        if False:
            for i in range(10):
                print('nop')
        'Convert a sequence of words `document` into the bag-of-words format of `[(word_id, word_count)]`\n        (e.g. `[(1, 4), (150, 1), (2005, 2)]`).\n\n        Notes\n        -----\n        Each word is assumed to be a **tokenized and normalized** string. No further preprocessing\n        is done on the words in `document`: you have to apply tokenization, stemming etc before calling this method.\n\n        If `allow_update` or `self.allow_update` is set, then also update the dictionary in the process: update overall\n        corpus statistics and document frequencies. For each id appearing in this document, increase its document\n        frequency (`self.dfs`) by one.\n\n        Parameters\n        ----------\n        document : sequence of str\n            A sequence of word tokens = **tokenized and normalized** strings.\n        allow_update : bool, optional\n            Update corpus statistics and if `debug=True`, also the reverse id=>word mapping?\n        return_missing : bool, optional\n            Not used. Only here for compatibility with the Dictionary class.\n\n        Return\n        ------\n        list of (int, int)\n            Document in Bag-of-words (BoW) format.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora import HashDictionary\n            >>>\n            >>> dct = HashDictionary()\n            >>> dct.doc2bow(["this", "is", "máma"])\n            [(1721, 1), (5280, 1), (22493, 1)]\n\n        '
        result = {}
        missing = {}
        document = sorted(document)
        for (word_norm, group) in itertools.groupby(document):
            frequency = len(list(group))
            tokenid = self.restricted_hash(word_norm)
            result[tokenid] = result.get(tokenid, 0) + frequency
            if self.debug:
                self.dfs_debug[word_norm] = self.dfs_debug.get(word_norm, 0) + 1
        if allow_update or self.allow_update:
            self.num_docs += 1
            self.num_pos += len(document)
            self.num_nnz += len(result)
            if self.debug:
                for tokenid in result.keys():
                    self.dfs[tokenid] = self.dfs.get(tokenid, 0) + 1
        result = sorted(result.items())
        if return_missing:
            return (result, missing)
        else:
            return result

    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000):
        if False:
            i = 10
            return i + 15
        'Filter tokens in the debug dictionary by their frequency.\n\n        Since :class:`~gensim.corpora.hashdictionary.HashDictionary` id range is fixed and doesn\'t depend on the number\n        of tokens seen, this doesn\'t really "remove" anything. It only clears some\n        internal corpus statistics, for easier debugging and a smaller RAM footprint.\n\n        Warnings\n        --------\n        Only makes sense when `debug=True`.\n\n        Parameters\n        ----------\n        no_below : int, optional\n            Keep tokens which are contained in at least `no_below` documents.\n        no_above : float, optional\n            Keep tokens which are contained in no more than `no_above` documents\n            (fraction of total corpus size, not an absolute number).\n        keep_n : int, optional\n            Keep only the first `keep_n` most frequent tokens.\n\n        Notes\n        -----\n        For tokens that appear in:\n\n        #. Less than `no_below` documents (absolute number) or \n\n        #. More than `no_above` documents (fraction of total corpus size, **not absolute number**).\n        #. After (1) and (2), keep only the first `keep_n` most frequent tokens (or keep all if `None`).\n\n        '
        no_above_abs = int(no_above * self.num_docs)
        ok = [item for item in self.dfs_debug.items() if no_below <= item[1] <= no_above_abs]
        ok = frozenset((word for (word, freq) in sorted(ok, key=lambda x: -x[1])[:keep_n]))
        self.dfs_debug = {word: freq for (word, freq) in self.dfs_debug.items() if word in ok}
        self.token2id = {token: tokenid for (token, tokenid) in self.token2id.items() if token in self.dfs_debug}
        self.id2token = {tokenid: {token for token in tokens if token in self.dfs_debug} for (tokenid, tokens) in self.id2token.items()}
        self.dfs = {tokenid: freq for (tokenid, freq) in self.dfs.items() if self.id2token.get(tokenid, False)}
        logger.info('kept statistics for which were in no less than %i and no more than %i (=%.1f%%) documents', no_below, no_above_abs, 100.0 * no_above)

    def save_as_text(self, fname):
        if False:
            for i in range(10):
                print('nop')
        'Save the debug token=>id mapping to a text file.\n\n        Warnings\n        --------\n        Only makes sense when `debug=True`, for debugging.\n\n        Parameters\n        ----------\n        fname : str\n            Path to output file.\n\n        Notes\n        -----\n        The format is:\n        `id[TAB]document frequency of this id[TAB]tab-separated set of words in UTF8 that map to this id[NEWLINE]`.\n\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora import HashDictionary\n            >>> from gensim.test.utils import get_tmpfile\n            >>>\n            >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]\n            >>> data = HashDictionary(corpus)\n            >>> data.save_as_text(get_tmpfile("dictionary_in_text_format"))\n\n        '
        logger.info('saving %s mapping to %s' % (self, fname))
        with utils.open(fname, 'wb') as fout:
            for tokenid in self.keys():
                words = sorted(self[tokenid])
                if words:
                    words_df = [(word, self.dfs_debug.get(word, 0)) for word in words]
                    words_df = ['%s(%i)' % item for item in sorted(words_df, key=lambda x: -x[1])]
                    words_df = '\t'.join(words_df)
                    fout.write(utils.to_utf8('%i\t%i\t%s\n' % (tokenid, self.dfs.get(tokenid, 0), words_df)))