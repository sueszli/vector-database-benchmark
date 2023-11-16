"""This module implements word vectors, and more generally sets of vectors keyed by lookup tokens/ints,
 and various similarity look-ups.

Since trained word vectors are independent from the way they were trained (:class:`~gensim.models.word2vec.Word2Vec`,
:class:`~gensim.models.fasttext.FastText` etc), they can be represented by a standalone structure,
as implemented in this module.

The structure is called "KeyedVectors" and is essentially a mapping between *keys*
and *vectors*. Each vector is identified by its lookup key, most often a short string token, so this is usually
a mapping between {str => 1D numpy array}.

The key is, in the original motivating case, a word (so the mapping maps words to 1D vectors),
but for some models, the key can also correspond to a document, a graph node etc.

(Because some applications may maintain their own integral identifiers, compact and contiguous
starting at zero, this class also supports use of plain ints as keys – in that case using them as literal
pointers to the position of the desired vector in the underlying array, and saving the overhead of
a lookup map entry.)

Why use KeyedVectors instead of a full model?
=============================================

+---------------------------+--------------+------------+-------------------------------------------------------------+
|        capability         | KeyedVectors | full model |                               note                          |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| continue training vectors | ❌           | ✅         | You need the full model to train or update vectors.         |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| smaller objects           | ✅           | ❌         | KeyedVectors are smaller and need less RAM, because they    |
|                           |              |            | don't need to store the model state that enables training.  |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| save/load from native     |              |            | Vectors exported by the Facebook and Google tools           |
| fasttext/word2vec format  | ✅           | ❌         | do not support further training, but you can still load     |
|                           |              |            | them into KeyedVectors.                                     |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| append new vectors        | ✅           | ✅         | Add new-vector entries to the mapping dynamically.          |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| concurrency               | ✅           | ✅         | Thread-safe, allows concurrent vector queries.              |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| shared RAM                | ✅           | ✅         | Multiple processes can re-use the same data, keeping only   |
|                           |              |            | a single copy in RAM using                                  |
|                           |              |            | `mmap <https://en.wikipedia.org/wiki/Mmap>`_.               |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| fast load                 | ✅           | ✅         | Supports `mmap <https://en.wikipedia.org/wiki/Mmap>`_       |
|                           |              |            | to load data from disk instantaneously.                     |
+---------------------------+--------------+------------+-------------------------------------------------------------+

TL;DR: the main difference is that KeyedVectors do not support further training.
On the other hand, by shedding the internal data structures necessary for training, KeyedVectors offer a smaller RAM
footprint and a simpler interface.

How to obtain word vectors?
===========================

Train a full model, then access its `model.wv` property, which holds the standalone keyed vectors.
For example, using the Word2Vec algorithm to train the vectors

.. sourcecode:: pycon

    >>> from gensim.test.utils import lee_corpus_list
    >>> from gensim.models import Word2Vec
    >>>
    >>> model = Word2Vec(lee_corpus_list, vector_size=24, epochs=100)
    >>> word_vectors = model.wv

Persist the word vectors to disk with

.. sourcecode:: pycon

    >>> from gensim.models import KeyedVectors
    >>>
    >>> word_vectors.save('vectors.kv')
    >>> reloaded_word_vectors = KeyedVectors.load('vectors.kv')

The vectors can also be instantiated from an existing file on disk
in the original Google's word2vec C format as a KeyedVectors instance

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)  # C text format
    >>> wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)  # C bin format

What can I do with word vectors?
================================

You can perform various syntactic/semantic NLP word tasks with the trained vectors.
Some of them are already built-in

.. sourcecode:: pycon

    >>> import gensim.downloader as api
    >>>
    >>> word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
    >>>
    >>> # Check the "most similar words", using the default "cosine similarity" measure.
    >>> result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
    >>> most_similar_key, similarity = result[0]  # look at the first match
    >>> print(f"{most_similar_key}: {similarity:.4f}")
    queen: 0.7699
    >>>
    >>> # Use a different similarity measure: "cosmul".
    >>> result = word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
    >>> most_similar_key, similarity = result[0]  # look at the first match
    >>> print(f"{most_similar_key}: {similarity:.4f}")
    queen: 0.8965
    >>>
    >>> print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))
    cereal
    >>>
    >>> similarity = word_vectors.similarity('woman', 'man')
    >>> similarity > 0.8
    True
    >>>
    >>> result = word_vectors.similar_by_word("cat")
    >>> most_similar_key, similarity = result[0]  # look at the first match
    >>> print(f"{most_similar_key}: {similarity:.4f}")
    dog: 0.8798
    >>>
    >>> sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
    >>> sentence_president = 'The president greets the press in Chicago'.lower().split()
    >>>
    >>> similarity = word_vectors.wmdistance(sentence_obama, sentence_president)
    >>> print(f"{similarity:.4f}")
    3.4893
    >>>
    >>> distance = word_vectors.distance("media", "media")
    >>> print(f"{distance:.1f}")
    0.0
    >>>
    >>> similarity = word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
    >>> print(f"{similarity:.4f}")
    0.7067
    >>>
    >>> vector = word_vectors['computer']  # numpy vector of a word
    >>> vector.shape
    (100,)
    >>>
    >>> vector = word_vectors.wv.get_vector('office', norm=True)
    >>> vector.shape
    (100,)

Correlation with human opinion on word similarity

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

And on word analogies

.. sourcecode:: pycon

    >>> analogy_scores = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

and so on.

"""
import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import dot, float32 as REAL, double, zeros, vstack, ndarray, sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
logger = logging.getLogger(__name__)
_KEY_TYPES = (str, int, np.integer)
_EXTENDED_KEY_TYPES = (str, int, np.integer, np.ndarray)

def _ensure_list(value):
    if False:
        i = 10
        return i + 15
    'Ensure that the specified value is wrapped in a list, for those supported cases\n    where we also accept a single key or vector.'
    if value is None:
        return []
    if isinstance(value, _KEY_TYPES) or (isinstance(value, ndarray) and len(value.shape) == 1):
        return [value]
    if isinstance(value, ndarray) and len(value.shape) == 2:
        return list(value)
    return value

class KeyedVectors(utils.SaveLoad):

    def __init__(self, vector_size, count=0, dtype=np.float32, mapfile_path=None):
        if False:
            for i in range(10):
                print('nop')
        'Mapping between keys (such as words) and vectors for :class:`~gensim.models.Word2Vec`\n        and related models.\n\n        Used to perform operations on the vectors such as vector lookup, distance, similarity etc.\n\n        To support the needs of specific models and other downstream uses, you can also set\n        additional attributes via the :meth:`~gensim.models.keyedvectors.KeyedVectors.set_vecattr`\n        and :meth:`~gensim.models.keyedvectors.KeyedVectors.get_vecattr` methods.\n        Note that all such attributes under the same `attr` name must have compatible `numpy`\n        types, as the type and storage array for such attributes is established by the 1st time such\n        `attr` is set.\n\n        Parameters\n        ----------\n        vector_size : int\n            Intended number of dimensions for all contained vectors.\n        count : int, optional\n            If provided, vectors wil be pre-allocated for at least this many vectors. (Otherwise\n            they can be added later.)\n        dtype : type, optional\n            Vector dimensions will default to `np.float32` (AKA `REAL` in some Gensim code) unless\n            another type is provided here.\n        mapfile_path : string, optional\n            Currently unused.\n        '
        self.vector_size = vector_size
        self.index_to_key = [None] * count
        self.next_index = 0
        self.key_to_index = {}
        self.vectors = zeros((count, vector_size), dtype=dtype)
        self.norms = None
        self.expandos = {}
        self.mapfile_path = mapfile_path

    def __str__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}<vector_size={self.vector_size}, {len(self)} keys>'

    def _load_specials(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Handle special requirements of `.load()` protocol, usually up-converting older versions.'
        super(KeyedVectors, self)._load_specials(*args, **kwargs)
        if hasattr(self, 'doctags'):
            self._upconvert_old_d2vkv()
        if not hasattr(self, 'index_to_key'):
            self.index_to_key = self.__dict__.pop('index2word', self.__dict__.pop('index2entity', None))
        if not hasattr(self, 'vectors'):
            self.vectors = self.__dict__.pop('syn0', None)
            self.vector_size = self.vectors.shape[1]
        if not hasattr(self, 'norms'):
            self.norms = None
        if not hasattr(self, 'expandos'):
            self.expandos = {}
        if 'key_to_index' not in self.__dict__:
            self._upconvert_old_vocab()
        if not hasattr(self, 'next_index'):
            self.next_index = len(self)

    def _upconvert_old_vocab(self):
        if False:
            for i in range(10):
                print('nop')
        "Convert a loaded, pre-gensim-4.0.0 version instance that had a 'vocab' dict of data objects."
        old_vocab = self.__dict__.pop('vocab', None)
        self.key_to_index = {}
        for k in old_vocab.keys():
            old_v = old_vocab[k]
            self.key_to_index[k] = old_v.index
            for attr in old_v.__dict__.keys():
                self.set_vecattr(old_v.index, attr, old_v.__dict__[attr])
        if 'sample_int' in self.expandos:
            self.expandos['sample_int'] = self.expandos['sample_int'].astype(np.uint32)

    def allocate_vecattrs(self, attrs=None, types=None):
        if False:
            i = 10
            return i + 15
        "Ensure arrays for given per-vector extra-attribute names & types exist, at right size.\n\n        The length of the index_to_key list is canonical 'intended size' of KeyedVectors,\n        even if other properties (vectors array) hasn't yet been allocated or expanded.\n        So this allocation targets that size.\n\n        "
        if attrs is None:
            attrs = list(self.expandos.keys())
            types = [self.expandos[attr].dtype for attr in attrs]
        target_size = len(self.index_to_key)
        for (attr, t) in zip(attrs, types):
            if t is int:
                t = np.int64
            if t is str:
                t = object
            if attr not in self.expandos:
                self.expandos[attr] = np.zeros(target_size, dtype=t)
                continue
            prev_expando = self.expandos[attr]
            if not np.issubdtype(t, prev_expando.dtype):
                raise TypeError(f"Can't allocate type {t} for attribute {attr}, conflicts with its existing type {prev_expando.dtype}")
            if len(prev_expando) == target_size:
                continue
            prev_count = len(prev_expando)
            self.expandos[attr] = np.zeros(target_size, dtype=prev_expando.dtype)
            self.expandos[attr][:min(prev_count, target_size),] = prev_expando[:min(prev_count, target_size),]

    def set_vecattr(self, key, attr, val):
        if False:
            while True:
                i = 10
        'Set attribute associated with the given key to value.\n\n        Parameters\n        ----------\n\n        key : str\n            Store the attribute for this vector key.\n        attr : str\n            Name of the additional attribute to store for the given key.\n        val : object\n            Value of the additional attribute to store for the given key.\n\n        Returns\n        -------\n\n        None\n\n        '
        self.allocate_vecattrs(attrs=[attr], types=[type(val)])
        index = self.get_index(key)
        self.expandos[attr][index] = val

    def get_vecattr(self, key, attr):
        if False:
            while True:
                i = 10
        'Get attribute value associated with given key.\n\n        Parameters\n        ----------\n\n        key : str\n            Vector key for which to fetch the attribute value.\n        attr : str\n            Name of the additional attribute to fetch for the given key.\n\n        Returns\n        -------\n\n        object\n            Value of the additional attribute fetched for the given key.\n\n        '
        index = self.get_index(key)
        return self.expandos[attr][index]

    def resize_vectors(self, seed=0):
        if False:
            print('Hello World!')
        'Make underlying vectors match index_to_key size; random-initialize any new rows.'
        target_shape = (len(self.index_to_key), self.vector_size)
        self.vectors = prep_vectors(target_shape, prior_vectors=self.vectors, seed=seed)
        self.allocate_vecattrs()
        self.norms = None

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.index_to_key)

    def __getitem__(self, key_or_keys):
        if False:
            return 10
        'Get vector representation of `key_or_keys`.\n\n        Parameters\n        ----------\n        key_or_keys : {str, list of str, int, list of int}\n            Requested key or list-of-keys.\n\n        Returns\n        -------\n        numpy.ndarray\n            Vector representation for `key_or_keys` (1D if `key_or_keys` is single key, otherwise - 2D).\n\n        '
        if isinstance(key_or_keys, _KEY_TYPES):
            return self.get_vector(key_or_keys)
        return vstack([self.get_vector(key) for key in key_or_keys])

    def get_index(self, key, default=None):
        if False:
            print('Hello World!')
        "Return the integer index (slot/position) where the given key's vector is stored in the\n        backing vectors array.\n\n        "
        val = self.key_to_index.get(key, -1)
        if val >= 0:
            return val
        elif isinstance(key, (int, np.integer)) and 0 <= key < len(self.index_to_key):
            return key
        elif default is not None:
            return default
        else:
            raise KeyError(f"Key '{key}' not present")

    def get_vector(self, key, norm=False):
        if False:
            for i in range(10):
                print('nop')
        "Get the key's vector, as a 1D numpy array.\n\n        Parameters\n        ----------\n\n        key : str\n            Key for vector to return.\n        norm : bool, optional\n            If True, the resulting vector will be L2-normalized (unit Euclidean length).\n\n        Returns\n        -------\n\n        numpy.ndarray\n            Vector for the specified key.\n\n        Raises\n        ------\n\n        KeyError\n            If the given key doesn't exist.\n\n        "
        index = self.get_index(key)
        if norm:
            self.fill_norms()
            result = self.vectors[index] / self.norms[index]
        else:
            result = self.vectors[index]
        result.setflags(write=False)
        return result

    @deprecated('Use get_vector instead')
    def word_vec(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Compatibility alias for get_vector(); must exist so subclass calls reach subclass get_vector().'
        return self.get_vector(*args, **kwargs)

    def get_mean_vector(self, keys, weights=None, pre_normalize=True, post_normalize=False, ignore_missing=True):
        if False:
            for i in range(10):
                print('nop')
        "Get the mean vector for a given list of keys.\n\n        Parameters\n        ----------\n\n        keys : list of (str or int or ndarray)\n            Keys specified by string or int ids or numpy array.\n        weights : list of float or numpy.ndarray, optional\n            1D array of same size of `keys` specifying the weight for each key.\n        pre_normalize : bool, optional\n            Flag indicating whether to normalize each keyvector before taking mean.\n            If False, individual keyvector will not be normalized.\n        post_normalize: bool, optional\n            Flag indicating whether to normalize the final mean vector.\n            If True, normalized mean vector will be return.\n        ignore_missing : bool, optional\n            If False, will raise error if a key doesn't exist in vocabulary.\n\n        Returns\n        -------\n\n        numpy.ndarray\n            Mean vector for the list of keys.\n\n        Raises\n        ------\n\n        ValueError\n            If the size of the list of `keys` and `weights` doesn't match.\n        KeyError\n            If any of the key doesn't exist in vocabulary and `ignore_missing` is false.\n\n        "
        if len(keys) == 0:
            raise ValueError('cannot compute mean with no input')
        if isinstance(weights, list):
            weights = np.array(weights)
        if weights is None:
            weights = np.ones(len(keys))
        if len(keys) != weights.shape[0]:
            raise ValueError('keys and weights array must have same number of elements')
        mean = np.zeros(self.vector_size, self.vectors.dtype)
        total_weight = 0
        for (idx, key) in enumerate(keys):
            if isinstance(key, ndarray):
                mean += weights[idx] * key
                total_weight += abs(weights[idx])
            elif self.__contains__(key):
                vec = self.get_vector(key, norm=pre_normalize)
                mean += weights[idx] * vec
                total_weight += abs(weights[idx])
            elif not ignore_missing:
                raise KeyError(f"Key '{key}' not present in vocabulary")
        if total_weight > 0:
            mean = mean / total_weight
        if post_normalize:
            mean = matutils.unitvec(mean).astype(REAL)
        return mean

    def add_vector(self, key, vector):
        if False:
            while True:
                i = 10
        "Add one new vector at the given key, into existing slot if available.\n\n        Warning: using this repeatedly is inefficient, requiring a full reallocation & copy,\n        if this instance hasn't been preallocated to be ready for such incremental additions.\n\n        Parameters\n        ----------\n\n        key: str\n            Key identifier of the added vector.\n        vector: numpy.ndarray\n            1D numpy array with the vector values.\n\n        Returns\n        -------\n        int\n            Index of the newly added vector, so that ``self.vectors[result] == vector`` and\n            ``self.index_to_key[result] == key``.\n\n        "
        target_index = self.next_index
        if target_index >= len(self) or self.index_to_key[target_index] is not None:
            target_index = len(self)
            warnings.warn('Adding single vectors to a KeyedVectors which grows by one each time can be costly. Consider adding in batches or preallocating to the required size.', UserWarning)
            self.add_vectors([key], [vector])
            self.allocate_vecattrs()
            self.next_index = target_index + 1
        else:
            self.index_to_key[target_index] = key
            self.key_to_index[key] = target_index
            self.vectors[target_index] = vector
            self.next_index += 1
        return target_index

    def add_vectors(self, keys, weights, extras=None, replace=False):
        if False:
            return 10
        'Append keys and their vectors in a manual way.\n        If some key is already in the vocabulary, the old vector is kept unless `replace` flag is True.\n\n        Parameters\n        ----------\n        keys : list of (str or int)\n            Keys specified by string or int ids.\n        weights: list of numpy.ndarray or numpy.ndarray\n            List of 1D np.array vectors or a 2D np.array of vectors.\n        replace: bool, optional\n            Flag indicating whether to replace vectors for keys which already exist in the map;\n            if True - replace vectors, otherwise - keep old vectors.\n\n        '
        if isinstance(keys, _KEY_TYPES):
            keys = [keys]
            weights = np.array(weights).reshape(1, -1)
        elif isinstance(weights, list):
            weights = np.array(weights)
        if extras is None:
            extras = {}
        self.allocate_vecattrs(extras.keys(), [extras[k].dtype for k in extras.keys()])
        in_vocab_mask = np.zeros(len(keys), dtype=bool)
        for (idx, key) in enumerate(keys):
            if key in self.key_to_index:
                in_vocab_mask[idx] = True
        for idx in np.nonzero(~in_vocab_mask)[0]:
            key = keys[idx]
            self.key_to_index[key] = len(self.index_to_key)
            self.index_to_key.append(key)
        self.vectors = vstack((self.vectors, weights[~in_vocab_mask].astype(self.vectors.dtype)))
        for (attr, extra) in extras:
            self.expandos[attr] = np.vstack((self.expandos[attr], extra[~in_vocab_mask]))
        if replace:
            in_vocab_idxs = [self.get_index(keys[idx]) for idx in np.nonzero(in_vocab_mask)[0]]
            self.vectors[in_vocab_idxs] = weights[in_vocab_mask]
            for (attr, extra) in extras:
                self.expandos[attr][in_vocab_idxs] = extra[in_vocab_mask]

    def __setitem__(self, keys, weights):
        if False:
            for i in range(10):
                print('nop')
        'Add keys and theirs vectors in a manual way.\n        If some key is already in the vocabulary, old vector is replaced with the new one.\n\n        This method is an alias for :meth:`~gensim.models.keyedvectors.KeyedVectors.add_vectors`\n        with `replace=True`.\n\n        Parameters\n        ----------\n        keys : {str, int, list of (str or int)}\n            keys specified by their string or int ids.\n        weights: list of numpy.ndarray or numpy.ndarray\n            List of 1D np.array vectors or 2D np.array of vectors.\n\n        '
        if not isinstance(keys, list):
            keys = [keys]
            weights = weights.reshape(1, -1)
        self.add_vectors(keys, weights, replace=True)

    def has_index_for(self, key):
        if False:
            while True:
                i = 10
        'Can this model return a single index for this key?\n\n        Subclasses that synthesize vectors for out-of-vocabulary words (like\n        :class:`~gensim.models.fasttext.FastText`) may respond True for a\n        simple `word in wv` (`__contains__()`) check but False for this\n        more-specific check.\n\n        '
        return self.get_index(key, -1) >= 0

    def __contains__(self, key):
        if False:
            print('Hello World!')
        return self.has_index_for(key)

    def most_similar_to_given(self, key1, keys_list):
        if False:
            while True:
                i = 10
        'Get the `key` from `keys_list` most similar to `key1`.'
        return keys_list[argmax([self.similarity(key1, key) for key in keys_list])]

    def closer_than(self, key1, key2):
        if False:
            print('Hello World!')
        'Get all keys that are closer to `key1` than `key2` is to `key1`.'
        all_distances = self.distances(key1)
        e1_index = self.get_index(key1)
        e2_index = self.get_index(key2)
        closer_node_indices = np.where(all_distances < all_distances[e2_index])[0]
        return [self.index_to_key[index] for index in closer_node_indices if index != e1_index]

    @deprecated('Use closer_than instead')
    def words_closer_than(self, word1, word2):
        if False:
            print('Hello World!')
        return self.closer_than(word1, word2)

    def rank(self, key1, key2):
        if False:
            while True:
                i = 10
        'Rank of the distance of `key2` from `key1`, in relation to distances of all keys from `key1`.'
        return len(self.closer_than(key1, key2)) + 1

    @property
    def vectors_norm(self):
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError('The `.vectors_norm` attribute is computed dynamically since Gensim 4.0.0. Use `.get_normed_vectors()` instead.\nSee https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4')

    @vectors_norm.setter
    def vectors_norm(self, _):
        if False:
            while True:
                i = 10
        pass

    def get_normed_vectors(self):
        if False:
            while True:
                i = 10
        'Get all embedding vectors normalized to unit L2 length (euclidean), as a 2D numpy array.\n\n        To see which key corresponds to which vector = which array row, refer\n        to the :attr:`~gensim.models.keyedvectors.KeyedVectors.index_to_key` attribute.\n\n        Returns\n        -------\n        numpy.ndarray:\n            2D numpy array of shape ``(number_of_keys, embedding dimensionality)``, L2-normalized\n            along the rows (key vectors).\n\n        '
        self.fill_norms()
        return self.vectors / self.norms[..., np.newaxis]

    def fill_norms(self, force=False):
        if False:
            return 10
        "\n        Ensure per-vector norms are available.\n\n        Any code which modifies vectors should ensure the accompanying norms are\n        either recalculated or 'None', to trigger a full recalculation later on-request.\n\n        "
        if self.norms is None or force:
            self.norms = np.linalg.norm(self.vectors, axis=1)

    @property
    def index2entity(self):
        if False:
            while True:
                i = 10
        raise AttributeError('The index2entity attribute has been replaced by index_to_key since Gensim 4.0.0.\nSee https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4')

    @index2entity.setter
    def index2entity(self, value):
        if False:
            print('Hello World!')
        self.index_to_key = value

    @property
    def index2word(self):
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError('The index2word attribute has been replaced by index_to_key since Gensim 4.0.0.\nSee https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4')

    @index2word.setter
    def index2word(self, value):
        if False:
            print('Hello World!')
        self.index_to_key = value

    @property
    def vocab(self):
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError("The vocab attribute was removed from KeyedVector in Gensim 4.0.0.\nUse KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.\nSee https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4")

    @vocab.setter
    def vocab(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.vocab()

    def sort_by_descending_frequency(self):
        if False:
            return 10
        'Sort the vocabulary so the most frequent words have the lowest indexes.'
        if not len(self):
            return
        count_sorted_indexes = np.argsort(self.expandos['count'])[::-1]
        self.index_to_key = [self.index_to_key[idx] for idx in count_sorted_indexes]
        self.allocate_vecattrs()
        for k in self.expandos:
            self.expandos[k] = self.expandos[k][count_sorted_indexes]
        if len(self.vectors):
            logger.warning('sorting after vectors have been allocated is expensive & error-prone')
            self.vectors = self.vectors[count_sorted_indexes]
        self.key_to_index = {word: i for (i, word) in enumerate(self.index_to_key)}

    def save(self, *args, **kwargs):
        if False:
            return 10
        'Save KeyedVectors to a file.\n\n        Parameters\n        ----------\n        fname : str\n            Path to the output file.\n\n        See Also\n        --------\n        :meth:`~gensim.models.keyedvectors.KeyedVectors.load`\n            Load a previously saved model.\n\n        '
        super(KeyedVectors, self).save(*args, **kwargs)

    def most_similar(self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None, restrict_vocab=None, indexer=None):
        if False:
            for i in range(10):
                print('nop')
        "Find the top-N most similar keys.\n        Positive keys contribute positively towards the similarity, negative keys negatively.\n\n        This method computes cosine similarity between a simple mean of the projection\n        weight vectors of the given keys and the vectors for each key in the model.\n        The method corresponds to the `word-analogy` and `distance` scripts in the original\n        word2vec implementation.\n\n        Parameters\n        ----------\n        positive : list of (str or int or ndarray) or list of ((str,float) or (int,float) or (ndarray,float)), optional\n            List of keys that contribute positively. If tuple, second element specifies the weight (default `1.0`)\n        negative : list of (str or int or ndarray) or list of ((str,float) or (int,float) or (ndarray,float)), optional\n            List of keys that contribute negatively. If tuple, second element specifies the weight (default `-1.0`)\n        topn : int or None, optional\n            Number of top-N similar keys to return, when `topn` is int. When `topn` is None,\n            then similarities for all keys are returned.\n        clip_start : int\n            Start clipping index.\n        clip_end : int\n            End clipping index.\n        restrict_vocab : int, optional\n            Optional integer which limits the range of vectors which\n            are searched for most-similar values. For example, restrict_vocab=10000 would\n            only check the first 10000 key vectors in the vocabulary order. (This may be\n            meaningful if you've sorted the vocabulary by descending frequency.) If\n            specified, overrides any values of ``clip_start`` or ``clip_end``.\n\n        Returns\n        -------\n        list of (str, float) or numpy.array\n            When `topn` is int, a sequence of (key, similarity) is returned.\n            When `topn` is None, then similarities for all keys are returned as a\n            one-dimensional numpy array with the size of the vocabulary.\n\n        "
        if isinstance(topn, Integral) and topn < 1:
            return []
        positive = _ensure_list(positive)
        negative = _ensure_list(negative)
        self.fill_norms()
        clip_end = clip_end or len(self.vectors)
        if restrict_vocab:
            clip_start = 0
            clip_end = restrict_vocab
        keys = []
        weight = np.concatenate((np.ones(len(positive)), -1.0 * np.ones(len(negative))))
        for (idx, item) in enumerate(positive + negative):
            if isinstance(item, _EXTENDED_KEY_TYPES):
                keys.append(item)
            else:
                keys.append(item[0])
                weight[idx] = item[1]
        mean = self.get_mean_vector(keys, weight, pre_normalize=True, post_normalize=True, ignore_missing=False)
        all_keys = [self.get_index(key) for key in keys if isinstance(key, _KEY_TYPES) and self.has_index_for(key)]
        if indexer is not None and isinstance(topn, int):
            return indexer.most_similar(mean, topn)
        dists = dot(self.vectors[clip_start:clip_end], mean) / self.norms[clip_start:clip_end]
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_keys), reverse=True)
        result = [(self.index_to_key[sim + clip_start], float(dists[sim])) for sim in best if sim + clip_start not in all_keys]
        return result[:topn]

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        if False:
            for i in range(10):
                print('nop')
        'Compatibility alias for similar_by_key().'
        return self.similar_by_key(word, topn, restrict_vocab)

    def similar_by_key(self, key, topn=10, restrict_vocab=None):
        if False:
            i = 10
            return i + 15
        "Find the top-N most similar keys.\n\n        Parameters\n        ----------\n        key : str\n            Key\n        topn : int or None, optional\n            Number of top-N similar keys to return. If topn is None, similar_by_key returns\n            the vector of similarity scores.\n        restrict_vocab : int, optional\n            Optional integer which limits the range of vectors which\n            are searched for most-similar values. For example, restrict_vocab=10000 would\n            only check the first 10000 key vectors in the vocabulary order. (This may be\n            meaningful if you've sorted the vocabulary by descending frequency.)\n\n        Returns\n        -------\n        list of (str, float) or numpy.array\n            When `topn` is int, a sequence of (key, similarity) is returned.\n            When `topn` is None, then similarities for all keys are returned as a\n            one-dimensional numpy array with the size of the vocabulary.\n\n        "
        return self.most_similar(positive=[key], topn=topn, restrict_vocab=restrict_vocab)

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        if False:
            print('Hello World!')
        "Find the top-N most similar keys by vector.\n\n        Parameters\n        ----------\n        vector : numpy.array\n            Vector from which similarities are to be computed.\n        topn : int or None, optional\n            Number of top-N similar keys to return, when `topn` is int. When `topn` is None,\n            then similarities for all keys are returned.\n        restrict_vocab : int, optional\n            Optional integer which limits the range of vectors which\n            are searched for most-similar values. For example, restrict_vocab=10000 would\n            only check the first 10000 key vectors in the vocabulary order. (This may be\n            meaningful if you've sorted the vocabulary by descending frequency.)\n\n        Returns\n        -------\n        list of (str, float) or numpy.array\n            When `topn` is int, a sequence of (key, similarity) is returned.\n            When `topn` is None, then similarities for all keys are returned as a\n            one-dimensional numpy array with the size of the vocabulary.\n\n        "
        return self.most_similar(positive=[vector], topn=topn, restrict_vocab=restrict_vocab)

    def wmdistance(self, document1, document2, norm=True):
        if False:
            print('Hello World!')
        'Compute the Word Mover\'s Distance between two documents.\n\n        When using this code, please consider citing the following papers:\n\n        * `Rémi Flamary et al. "POT: Python Optimal Transport"\n          <https://jmlr.org/papers/v22/20-451.html>`_\n        * `Matt Kusner et al. "From Word Embeddings To Document Distances"\n          <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.\n\n        Parameters\n        ----------\n        document1 : list of str\n            Input document.\n        document2 : list of str\n            Input document.\n        norm : boolean\n            Normalize all word vectors to unit length before computing the distance?\n            Defaults to True.\n\n        Returns\n        -------\n        float\n            Word Mover\'s distance between `document1` and `document2`.\n\n        Warnings\n        --------\n        This method only works if `POT <https://pypi.org/project/POT/>`_ is installed.\n\n        If one of the documents have no words that exist in the vocab, `float(\'inf\')` (i.e. infinity)\n        will be returned.\n\n        Raises\n        ------\n        ImportError\n            If `POT <https://pypi.org/project/POT/>`_  isn\'t installed.\n\n        '
        from ot import emd2
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in self]
        document2 = [token for token in document2 if token in self]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)
        if not document1 or not document2:
            logger.warning('At least one of the documents had no words that were in the vocabulary.')
            return float('inf')
        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)
        if vocab_len == 1:
            return 0.0
        doclist1 = list(set(document1))
        doclist2 = list(set(document2))
        v1 = np.array([self.get_vector(token, norm=norm) for token in doclist1])
        v2 = np.array([self.get_vector(token, norm=norm) for token in doclist2])
        doc1_indices = dictionary.doc2idx(doclist1)
        doc2_indices = dictionary.doc2idx(doclist2)
        distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
        distance_matrix[np.ix_(doc1_indices, doc2_indices)] = cdist(v1, v2)
        if abs(np_sum(distance_matrix)) < 1e-08:
            logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def nbow(document):
            if False:
                print('Hello World!')
            d = zeros(vocab_len, dtype=double)
            nbow = dictionary.doc2bow(document)
            doc_len = len(document)
            for (idx, freq) in nbow:
                d[idx] = freq / float(doc_len)
            return d
        d1 = nbow(document1)
        d2 = nbow(document2)
        return emd2(d1, d2, distance_matrix)

    def most_similar_cosmul(self, positive=None, negative=None, topn=10, restrict_vocab=None):
        if False:
            print('Hello World!')
        'Find the top-N most similar words, using the multiplicative combination objective,\n        proposed by `Omer Levy and Yoav Goldberg "Linguistic Regularities in Sparse and Explicit Word Representations"\n        <http://www.aclweb.org/anthology/W14-1618>`_. Positive words still contribute positively towards the similarity,\n        negative words negatively, but with less susceptibility to one large distance dominating the calculation.\n        In the common analogy-solving case, of two positive and one negative examples,\n        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.\n\n        Additional positive or negative examples contribute to the numerator or denominator,\n        respectively - a potentially sensible but untested extension of the method.\n        With a single positive example, rankings will be the same as in the default\n        :meth:`~gensim.models.keyedvectors.KeyedVectors.most_similar`.\n\n        Allows calls like most_similar_cosmul(\'dog\', \'cat\'), as a shorthand for\n        most_similar_cosmul([\'dog\'], [\'cat\']) where \'dog\' is positive and \'cat\' negative\n\n        Parameters\n        ----------\n        positive : list of str, optional\n            List of words that contribute positively.\n        negative : list of str, optional\n            List of words that contribute negatively.\n        topn : int or None, optional\n            Number of top-N similar words to return, when `topn` is int. When `topn` is None,\n            then similarities for all words are returned.\n        restrict_vocab : int or None, optional\n            Optional integer which limits the range of vectors which are searched for most-similar values.\n            For example, restrict_vocab=10000 would only check the first 10000 node vectors in the vocabulary order.\n            This may be meaningful if vocabulary is sorted by descending frequency.\n\n\n        Returns\n        -------\n        list of (str, float) or numpy.array\n            When `topn` is int, a sequence of (word, similarity) is returned.\n            When `topn` is None, then similarities for all words are returned as a\n            one-dimensional numpy array with the size of the vocabulary.\n\n        '
        if isinstance(topn, Integral) and topn < 1:
            return []
        positive = _ensure_list(positive)
        negative = _ensure_list(negative)
        self.init_sims()
        if isinstance(positive, str):
            positive = [positive]
        if isinstance(negative, str):
            negative = [negative]
        all_words = {self.get_index(word) for word in positive + negative if not isinstance(word, ndarray) and word in self.key_to_index}
        positive = [self.get_vector(word, norm=True) if isinstance(word, str) else word for word in positive]
        negative = [self.get_vector(word, norm=True) if isinstance(word, str) else word for word in negative]
        if not positive:
            raise ValueError('cannot compute similarity with no input')
        pos_dists = [(1 + dot(self.vectors, term) / self.norms) / 2 for term in positive]
        neg_dists = [(1 + dot(self.vectors, term) / self.norms) / 2 for term in negative]
        dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 1e-06)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        result = [(self.index_to_key[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def rank_by_centrality(self, words, use_norm=True):
        if False:
            print('Hello World!')
        'Rank the given words by similarity to the centroid of all the words.\n\n        Parameters\n        ----------\n        words : list of str\n            List of keys.\n        use_norm : bool, optional\n            Whether to calculate centroid using unit-normed vectors; default True.\n\n        Returns\n        -------\n        list of (float, str)\n            Ranked list of (similarity, key), most-similar to the centroid first.\n\n        '
        self.fill_norms()
        used_words = [word for word in words if word in self]
        if len(used_words) != len(words):
            ignored_words = set(words) - set(used_words)
            logger.warning('vectors for words %s are not present in the model, ignoring these words', ignored_words)
        if not used_words:
            raise ValueError('cannot select a word from an empty list')
        vectors = vstack([self.get_vector(word, norm=use_norm) for word in used_words]).astype(REAL)
        mean = self.get_mean_vector(vectors, post_normalize=True)
        dists = dot(vectors, mean)
        return sorted(zip(dists, used_words), reverse=True)

    def doesnt_match(self, words):
        if False:
            for i in range(10):
                print('nop')
        "Which key from the given list doesn't go with the others?\n\n        Parameters\n        ----------\n        words : list of str\n            List of keys.\n\n        Returns\n        -------\n        str\n            The key further away from the mean of all keys.\n\n        "
        return self.rank_by_centrality(words)[-1][1]

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        if False:
            return 10
        'Compute cosine similarities between one vector and a set of other vectors.\n\n        Parameters\n        ----------\n        vector_1 : numpy.ndarray\n            Vector from which similarities are to be computed, expected shape (dim,).\n        vectors_all : numpy.ndarray\n            For each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).\n\n        Returns\n        -------\n        numpy.ndarray\n            Contains cosine distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).\n\n        '
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities

    def distances(self, word_or_vector, other_words=()):
        if False:
            for i in range(10):
                print('nop')
        'Compute cosine distances from given word or vector to all words in `other_words`.\n        If `other_words` is empty, return distance between `word_or_vector` and all words in vocab.\n\n        Parameters\n        ----------\n        word_or_vector : {str, numpy.ndarray}\n            Word or vector from which distances are to be computed.\n        other_words : iterable of str\n            For each word in `other_words` distance from `word_or_vector` is computed.\n            If None or empty, distance of `word_or_vector` from all words in vocab is computed (including itself).\n\n        Returns\n        -------\n        numpy.array\n            Array containing distances to all words in `other_words` from input `word_or_vector`.\n\n        Raises\n        -----\n        KeyError\n            If either `word_or_vector` or any word in `other_words` is absent from vocab.\n\n        '
        if isinstance(word_or_vector, _KEY_TYPES):
            input_vector = self.get_vector(word_or_vector)
        else:
            input_vector = word_or_vector
        if not other_words:
            other_vectors = self.vectors
        else:
            other_indices = [self.get_index(word) for word in other_words]
            other_vectors = self.vectors[other_indices]
        return 1 - self.cosine_similarities(input_vector, other_vectors)

    def distance(self, w1, w2):
        if False:
            while True:
                i = 10
        'Compute cosine distance between two keys.\n        Calculate 1 - :meth:`~gensim.models.keyedvectors.KeyedVectors.similarity`.\n\n        Parameters\n        ----------\n        w1 : str\n            Input key.\n        w2 : str\n            Input key.\n\n        Returns\n        -------\n        float\n            Distance between `w1` and `w2`.\n\n        '
        return 1 - self.similarity(w1, w2)

    def similarity(self, w1, w2):
        if False:
            for i in range(10):
                print('nop')
        'Compute cosine similarity between two keys.\n\n        Parameters\n        ----------\n        w1 : str\n            Input key.\n        w2 : str\n            Input key.\n\n        Returns\n        -------\n        float\n            Cosine similarity between `w1` and `w2`.\n\n        '
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def n_similarity(self, ws1, ws2):
        if False:
            return 10
        'Compute cosine similarity between two sets of keys.\n\n        Parameters\n        ----------\n        ws1 : list of str\n            Sequence of keys.\n        ws2: list of str\n            Sequence of keys.\n\n        Returns\n        -------\n        numpy.ndarray\n            Similarities between `ws1` and `ws2`.\n\n        '
        if not (len(ws1) and len(ws2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')
        mean1 = self.get_mean_vector(ws1, pre_normalize=False)
        mean2 = self.get_mean_vector(ws2, pre_normalize=False)
        return dot(matutils.unitvec(mean1), matutils.unitvec(mean2))

    @staticmethod
    def _log_evaluate_word_analogies(section):
        if False:
            print('Hello World!')
        'Calculate score by section, helper for\n        :meth:`~gensim.models.keyedvectors.KeyedVectors.evaluate_word_analogies`.\n\n        Parameters\n        ----------\n        section : dict of (str, (str, str, str, str))\n            Section given from evaluation.\n\n        Returns\n        -------\n        float\n            Accuracy score if at least one prediction was made (correct or incorrect).\n\n            Or return 0.0 if there were no predictions at all in this section.\n\n        '
        (correct, incorrect) = (len(section['correct']), len(section['incorrect']))
        if correct + incorrect == 0:
            return 0.0
        score = correct / (correct + incorrect)
        logger.info('%s: %.1f%% (%i/%i)', section['section'], 100.0 * score, correct, correct + incorrect)
        return score

    def evaluate_word_analogies(self, analogies, restrict_vocab=300000, case_insensitive=True, dummy4unknown=False, similarity_function='most_similar'):
        if False:
            return 10
        'Compute performance of the model on an analogy test set.\n\n        The accuracy is reported (printed to log and returned as a score) for each section separately,\n        plus there\'s one aggregate summary at the end.\n\n        This method corresponds to the `compute-accuracy` script of the original C word2vec.\n        See also `Analogy (State of the art) <https://aclweb.org/aclwiki/Analogy_(State_of_the_art)>`_.\n\n        Parameters\n        ----------\n        analogies : str\n            Path to file, where lines are 4-tuples of words, split into sections by ": SECTION NAME" lines.\n            See `gensim/test/test_data/questions-words.txt` as example.\n        restrict_vocab : int, optional\n            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.\n            This may be meaningful if you\'ve sorted the model vocabulary by descending frequency (which is standard\n            in modern word embedding models).\n        case_insensitive : bool, optional\n            If True - convert all words to their uppercase form before evaluating the performance.\n            Useful to handle case-mismatch between training tokens and words in the test set.\n            In case of multiple case variants of a single word, the vector for the first occurrence\n            (also the most frequent if vocabulary is sorted) is taken.\n        dummy4unknown : bool, optional\n            If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.\n            Otherwise, these tuples are skipped entirely and not used in the evaluation.\n        similarity_function : str, optional\n            Function name used for similarity calculation.\n\n        Returns\n        -------\n        score : float\n            The overall evaluation score on the entire evaluation set\n        sections : list of dict of {str : str or list of tuple of (str, str, str, str)}\n            Results broken down by each section of the evaluation set. Each dict contains the name of the section\n            under the key \'section\', and lists of correctly and incorrectly predicted 4-tuples of words under the\n            keys \'correct\' and \'incorrect\'.\n\n        '
        ok_keys = self.index_to_key[:restrict_vocab]
        if case_insensitive:
            ok_vocab = {k.upper(): self.get_index(k) for k in reversed(ok_keys)}
        else:
            ok_vocab = {k: self.get_index(k) for k in reversed(ok_keys)}
        oov = 0
        logger.info('Evaluating word analogies for top %i words in the model on %s', restrict_vocab, analogies)
        (sections, section) = ([], None)
        quadruplets_no = 0
        with utils.open(analogies, 'rb') as fin:
            for (line_no, line) in enumerate(fin):
                line = utils.to_unicode(line)
                if line.startswith(': '):
                    if section:
                        sections.append(section)
                        self._log_evaluate_word_analogies(section)
                    section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
                else:
                    if not section:
                        raise ValueError('Missing section header before line #%i in %s' % (line_no, analogies))
                    try:
                        if case_insensitive:
                            (a, b, c, expected) = [word.upper() for word in line.split()]
                        else:
                            (a, b, c, expected) = [word for word in line.split()]
                    except ValueError:
                        logger.info('Skipping invalid line #%i in %s', line_no, analogies)
                        continue
                    quadruplets_no += 1
                    if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or (expected not in ok_vocab):
                        oov += 1
                        if dummy4unknown:
                            logger.debug('Zero accuracy for line #%d with OOV words: %s', line_no, line.strip())
                            section['incorrect'].append((a, b, c, expected))
                        else:
                            logger.debug('Skipping line #%i with OOV words: %s', line_no, line.strip())
                        continue
                    original_key_to_index = self.key_to_index
                    self.key_to_index = ok_vocab
                    ignore = {a, b, c}
                    predicted = None
                    sims = self.most_similar(positive=[b, c], negative=[a], topn=5, restrict_vocab=restrict_vocab)
                    self.key_to_index = original_key_to_index
                    for element in sims:
                        predicted = element[0].upper() if case_insensitive else element[0]
                        if predicted in ok_vocab and predicted not in ignore:
                            if predicted != expected:
                                logger.debug('%s: expected %s, predicted %s', line.strip(), expected, predicted)
                            break
                    if predicted == expected:
                        section['correct'].append((a, b, c, expected))
                    else:
                        section['incorrect'].append((a, b, c, expected))
        if section:
            sections.append(section)
            self._log_evaluate_word_analogies(section)
        total = {'section': 'Total accuracy', 'correct': list(itertools.chain.from_iterable((s['correct'] for s in sections))), 'incorrect': list(itertools.chain.from_iterable((s['incorrect'] for s in sections)))}
        oov_ratio = float(oov) / quadruplets_no * 100
        logger.info('Quadruplets with out-of-vocabulary words: %.1f%%', oov_ratio)
        if not dummy4unknown:
            logger.info('NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"')
        analogies_score = self._log_evaluate_word_analogies(total)
        sections.append(total)
        return (analogies_score, sections)

    @staticmethod
    def log_accuracy(section):
        if False:
            i = 10
            return i + 15
        (correct, incorrect) = (len(section['correct']), len(section['incorrect']))
        if correct + incorrect > 0:
            logger.info('%s: %.1f%% (%i/%i)', section['section'], 100.0 * correct / (correct + incorrect), correct, correct + incorrect)

    @staticmethod
    def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
        if False:
            while True:
                i = 10
        logger.info('Pearson correlation coefficient against %s: %.4f', pairs, pearson[0])
        logger.info('Spearman rank-order correlation coefficient against %s: %.4f', pairs, spearman[0])
        logger.info('Pairs with unknown words ratio: %.1f%%', oov)

    def evaluate_word_pairs(self, pairs, delimiter='\t', encoding='utf8', restrict_vocab=300000, case_insensitive=True, dummy4unknown=False):
        if False:
            i = 10
            return i + 15
        "Compute correlation of the model with human similarity judgments.\n\n        Notes\n        -----\n        More datasets can be found at\n        * http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html\n        * https://www.cl.cam.ac.uk/~fh295/simlex.html.\n\n        Parameters\n        ----------\n        pairs : str\n            Path to file, where lines are 3-tuples, each consisting of a word pair and a similarity value.\n            See `test/test_data/wordsim353.tsv` as example.\n        delimiter : str, optional\n            Separator in `pairs` file.\n        restrict_vocab : int, optional\n            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.\n            This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard\n            in modern word embedding models).\n        case_insensitive : bool, optional\n            If True - convert all words to their uppercase form before evaluating the performance.\n            Useful to handle case-mismatch between training tokens and words in the test set.\n            In case of multiple case variants of a single word, the vector for the first occurrence\n            (also the most frequent if vocabulary is sorted) is taken.\n        dummy4unknown : bool, optional\n            If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.\n            Otherwise, these tuples are skipped entirely and not used in the evaluation.\n\n        Returns\n        -------\n        pearson : tuple of (float, float)\n            Pearson correlation coefficient with 2-tailed p-value.\n        spearman : tuple of (float, float)\n            Spearman rank-order correlation coefficient between the similarities from the dataset and the\n            similarities produced by the model itself, with 2-tailed p-value.\n        oov_ratio : float\n            The ratio of pairs with unknown words.\n\n        "
        ok_keys = self.index_to_key[:restrict_vocab]
        if case_insensitive:
            ok_vocab = {k.upper(): self.get_index(k) for k in reversed(ok_keys)}
        else:
            ok_vocab = {k: self.get_index(k) for k in reversed(ok_keys)}
        similarity_gold = []
        similarity_model = []
        oov = 0
        (original_key_to_index, self.key_to_index) = (self.key_to_index, ok_vocab)
        try:
            with utils.open(pairs, encoding=encoding) as fin:
                for (line_no, line) in enumerate(fin):
                    if not line or line.startswith('#'):
                        continue
                    try:
                        if case_insensitive:
                            (a, b, sim) = [word.upper() for word in line.split(delimiter)]
                        else:
                            (a, b, sim) = [word for word in line.split(delimiter)]
                        sim = float(sim)
                    except (ValueError, TypeError):
                        logger.info('Skipping invalid line #%d in %s', line_no, pairs)
                        continue
                    if a not in ok_vocab or b not in ok_vocab:
                        oov += 1
                        if dummy4unknown:
                            logger.debug('Zero similarity for line #%d with OOV words: %s', line_no, line.strip())
                            similarity_model.append(0.0)
                            similarity_gold.append(sim)
                        else:
                            logger.info('Skipping line #%d with OOV words: %s', line_no, line.strip())
                        continue
                    similarity_gold.append(sim)
                    similarity_model.append(self.similarity(a, b))
        finally:
            self.key_to_index = original_key_to_index
        assert len(similarity_gold) == len(similarity_model)
        if not similarity_gold:
            raise ValueError(f'No valid similarity judgements found in {pairs}: either invalid format or all are out-of-vocabulary in {self}')
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        if dummy4unknown:
            oov_ratio = float(oov) / len(similarity_gold) * 100
        else:
            oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100
        logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
        logger.debug('Spearman rank-order correlation coefficient against %s: %f with p-value %f', pairs, spearman[0], spearman[1])
        logger.debug('Pairs with unknown words: %d', oov)
        self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
        return (pearson, spearman, oov_ratio)

    @deprecated('Use fill_norms() instead. See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4')
    def init_sims(self, replace=False):
        if False:
            while True:
                i = 10
        "Precompute data helpful for bulk similarity calculations.\n\n        :meth:`~gensim.models.keyedvectors.KeyedVectors.fill_norms` now preferred for this purpose.\n\n        Parameters\n        ----------\n\n        replace : bool, optional\n            If True - forget the original vectors and only keep the normalized ones.\n\n        Warnings\n        --------\n\n        You **cannot sensibly continue training** after doing a replace on a model's\n        internal KeyedVectors, and a replace is no longer necessary to save RAM. Do not use this method.\n\n        "
        self.fill_norms()
        if replace:
            logger.warning('destructive init_sims(replace=True) deprecated & no longer required for space-efficiency')
            self.unit_normalize_all()

    def unit_normalize_all(self):
        if False:
            i = 10
            return i + 15
        'Destructively scale all vectors to unit-length.\n\n        You cannot sensibly continue training after such a step.\n\n        '
        self.fill_norms()
        self.vectors /= self.norms[..., np.newaxis]
        self.norms = np.ones((len(self.vectors),))

    def relative_cosine_similarity(self, wa, wb, topn=10):
        if False:
            i = 10
            return i + 15
        'Compute the relative cosine similarity between two words given top-n similar words,\n        by `Artuur Leeuwenberga, Mihaela Velab , Jon Dehdaribc, Josef van Genabithbc "A Minimally Supervised Approach\n        for Synonym Extraction with Word Embeddings" <https://ufal.mff.cuni.cz/pbml/105/art-leeuwenberg-et-al.pdf>`_.\n\n        To calculate relative cosine similarity between two words, equation (1) of the paper is used.\n        For WordNet synonyms, if rcs(topn=10) is greater than 0.10 then wa and wb are more similar than\n        any arbitrary word pairs.\n\n        Parameters\n        ----------\n        wa: str\n            Word for which we have to look top-n similar word.\n        wb: str\n            Word for which we evaluating relative cosine similarity with wa.\n        topn: int, optional\n            Number of top-n similar words to look with respect to wa.\n\n        Returns\n        -------\n        numpy.float64\n            Relative cosine similarity between wa and wb.\n\n        '
        sims = self.similar_by_word(wa, topn)
        if not sims:
            raise ValueError('Cannot calculate relative cosine similarity without any similar words.')
        rcs = float(self.similarity(wa, wb)) / sum((sim for (_, sim) in sims))
        return rcs

    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None, write_header=True, prefix='', append=False, sort_attr='count'):
        if False:
            return 10
        "Store the input-hidden weight matrix in the same format used by the original\n        C word2vec-tool, for compatibility.\n\n        Parameters\n        ----------\n        fname : str\n            File path to save the vectors to.\n        fvocab : str, optional\n            File path to save additional vocabulary information to. `None` to not store the vocabulary.\n        binary : bool, optional\n            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.\n        total_vec : int, optional\n            Explicitly specify total number of vectors\n            (in case word vectors are appended with document vectors afterwards).\n        write_header : bool, optional\n            If False, don't write the 1st line declaring the count of vectors and dimensions.\n            This is the format used by e.g. gloVe vectors.\n        prefix : str, optional\n            String to prepend in front of each stored word. Default = no prefix.\n        append : bool, optional\n            If set, open `fname` in `ab` mode instead of the default `wb` mode.\n        sort_attr : str, optional\n            Sort the output vectors in descending order of this attribute. Default: most frequent keys first.\n\n        "
        if total_vec is None:
            total_vec = len(self.index_to_key)
        mode = 'wb' if not append else 'ab'
        if sort_attr in self.expandos:
            store_order_vocab_keys = sorted(self.key_to_index.keys(), key=lambda k: -self.get_vecattr(k, sort_attr))
        else:
            if fvocab is not None:
                raise ValueError(f"Cannot store vocabulary with '{sort_attr}' because that attribute does not exist")
            logger.warning('attribute %s not present in %s; will store in internal index_to_key order', sort_attr, self)
            store_order_vocab_keys = self.index_to_key
        if fvocab is not None:
            logger.info('storing vocabulary in %s', fvocab)
            with utils.open(fvocab, mode) as vout:
                for word in store_order_vocab_keys:
                    vout.write(f'{prefix}{word} {self.get_vecattr(word, sort_attr)}\n'.encode('utf8'))
        logger.info('storing %sx%s projection weights into %s', total_vec, self.vector_size, fname)
        assert (len(self.index_to_key), self.vector_size) == self.vectors.shape
        index_id_count = 0
        for (i, val) in enumerate(self.index_to_key):
            if i != val:
                break
            index_id_count += 1
        keys_to_write = itertools.chain(range(0, index_id_count), store_order_vocab_keys)
        with utils.open(fname, mode) as fout:
            if write_header:
                fout.write(f'{total_vec} {self.vector_size}\n'.encode('utf8'))
            for key in keys_to_write:
                key_vector = self[key]
                if binary:
                    fout.write(f'{prefix}{key} '.encode('utf8') + key_vector.astype(REAL).tobytes())
                else:
                    fout.write(f"{prefix}{key} {' '.join((repr(val) for val in key_vector))}\n".encode('utf8'))

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict', limit=None, datatype=REAL, no_header=False):
        if False:
            while True:
                i = 10
        "Load KeyedVectors from a file produced by the original C word2vec-tool format.\n\n        Warnings\n        --------\n        The information stored in the file is incomplete (the binary tree is missing),\n        so while you can query for word similarity etc., you cannot continue training\n        with a model loaded this way.\n\n        Parameters\n        ----------\n        fname : str\n            The file path to the saved word2vec-format file.\n        fvocab : str, optional\n            File path to the vocabulary.Word counts are read from `fvocab` filename, if set\n            (this is the file generated by `-save-vocab` flag of the original C tool).\n        binary : bool, optional\n            If True, indicates whether the data is in binary word2vec format.\n        encoding : str, optional\n            If you trained the C model using non-utf8 encoding for words, specify that encoding in `encoding`.\n        unicode_errors : str, optional\n            default 'strict', is a string suitable to be passed as the `errors`\n            argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source\n            file may include word tokens truncated in the middle of a multibyte unicode character\n            (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.\n        limit : int, optional\n            Sets a maximum number of word-vectors to read from the file. The default,\n            None, means read all.\n        datatype : type, optional\n            (Experimental) Can coerce dimensions to a non-default float type (such as `np.float16`) to save memory.\n            Such types may result in much slower bulk operations or incompatibility with optimized routines.)\n        no_header : bool, optional\n            Default False means a usual word2vec-format file, with a 1st line declaring the count of\n            following vectors & number of dimensions. If True, the file is assumed to lack a declaratory\n            (vocab_size, vector_size) header and instead start with the 1st vector, and an extra\n            reading-pass will be used to discover the number of vectors. Works only with `binary=False`.\n\n        Returns\n        -------\n        :class:`~gensim.models.keyedvectors.KeyedVectors`\n            Loaded model.\n\n        "
        return _load_word2vec_format(cls, fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors, limit=limit, datatype=datatype, no_header=no_header)

    def intersect_word2vec_format(self, fname, lockf=0.0, binary=False, encoding='utf8', unicode_errors='strict'):
        if False:
            while True:
                i = 10
        "Merge in an input-hidden weight matrix loaded from the original C word2vec-tool format,\n        where it intersects with the current vocabulary.\n\n        No words are added to the existing vocabulary, but intersecting words adopt the file's weights, and\n        non-intersecting words are left alone.\n\n        Parameters\n        ----------\n        fname : str\n            The file path to load the vectors from.\n        lockf : float, optional\n            Lock-factor value to be set for any imported word-vectors; the\n            default value of 0.0 prevents further updating of the vector during subsequent\n            training. Use 1.0 to allow further training updates of merged vectors.\n        binary : bool, optional\n            If True, `fname` is in the binary word2vec C format.\n        encoding : str, optional\n            Encoding of `text` for `unicode` function (python2 only).\n        unicode_errors : str, optional\n            Error handling behaviour, used as parameter for `unicode` function (python2 only).\n\n        "
        overlap_count = 0
        logger.info('loading projection weights from %s', fname)
        with utils.open(fname, 'rb') as fin:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            (vocab_size, vector_size) = (int(x) for x in header.split())
            if not vector_size == self.vector_size:
                raise ValueError('incompatible vector size %d in file %s' % (vector_size, fname))
            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                for _ in range(vocab_size):
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = np.fromstring(fin.read(binary_len), dtype=REAL)
                    if word in self.key_to_index:
                        overlap_count += 1
                        self.vectors[self.get_index(word)] = weights
                        self.vectors_lockf[self.get_index(word)] = lockf
            else:
                for (line_no, line) in enumerate(fin):
                    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(' ')
                    if len(parts) != vector_size + 1:
                        raise ValueError('invalid vector on line %s (is this really the text format?)' % line_no)
                    (word, weights) = (parts[0], [REAL(x) for x in parts[1:]])
                    if word in self.key_to_index:
                        overlap_count += 1
                        self.vectors[self.get_index(word)] = weights
                        self.vectors_lockf[self.get_index(word)] = lockf
        self.add_lifecycle_event('intersect_word2vec_format', msg=f'merged {overlap_count} vectors into {self.vectors.shape} matrix from {fname}')

    def vectors_for_all(self, keys: Iterable, allow_inference: bool=True, copy_vecattrs: bool=False) -> 'KeyedVectors':
        if False:
            return 10
        "Produce vectors for all given keys as a new :class:`KeyedVectors` object.\n\n        Notes\n        -----\n        The keys will always be deduplicated. For optimal performance, you should not pass entire\n        corpora to the method. Instead, you should construct a dictionary of unique words in your\n        corpus:\n\n        >>> from collections import Counter\n        >>> import itertools\n        >>>\n        >>> from gensim.models import FastText\n        >>> from gensim.test.utils import datapath, common_texts\n        >>>\n        >>> model_corpus_file = datapath('lee_background.cor')  # train word vectors on some corpus\n        >>> model = FastText(corpus_file=model_corpus_file, vector_size=20, min_count=1)\n        >>> corpus = common_texts  # infer word vectors for words from another corpus\n        >>> word_counts = Counter(itertools.chain.from_iterable(corpus))  # count words in your corpus\n        >>> words_by_freq = (k for k, v in word_counts.most_common())\n        >>> word_vectors = model.wv.vectors_for_all(words_by_freq)  # create word-vectors for words in your corpus\n\n        Parameters\n        ----------\n        keys : iterable\n            The keys that will be vectorized.\n        allow_inference : bool, optional\n            In subclasses such as :class:`~gensim.models.fasttext.FastTextKeyedVectors`,\n            vectors for out-of-vocabulary keys (words) may be inferred. Default is True.\n        copy_vecattrs : bool, optional\n            Additional attributes set via the :meth:`KeyedVectors.set_vecattr` method\n            will be preserved in the produced :class:`KeyedVectors` object. Default is False.\n            To ensure that *all* the produced vectors will have vector attributes assigned,\n            you should set `allow_inference=False`.\n\n        Returns\n        -------\n        keyedvectors : :class:`~gensim.models.keyedvectors.KeyedVectors`\n            Vectors for all the given keys.\n\n        "
        (vocab, seen) = ([], set())
        for key in keys:
            if key not in seen:
                seen.add(key)
                if key in (self if allow_inference else self.key_to_index):
                    vocab.append(key)
        kv = KeyedVectors(self.vector_size, len(vocab), dtype=self.vectors.dtype)
        for key in vocab:
            weights = self[key]
            _add_word_to_kv(kv, None, key, weights, len(vocab))
            if copy_vecattrs:
                for attr in self.expandos:
                    try:
                        kv.set_vecattr(key, attr, self.get_vecattr(key, attr))
                    except KeyError:
                        pass
        return kv

    def _upconvert_old_d2vkv(self):
        if False:
            print('Hello World!')
        'Convert a deserialized older Doc2VecKeyedVectors instance to latest generic KeyedVectors'
        self.vocab = self.doctags
        self._upconvert_old_vocab()
        for k in self.key_to_index.keys():
            old_offset = self.get_vecattr(k, 'offset')
            true_index = old_offset + self.max_rawint + 1
            self.key_to_index[k] = true_index
        del self.expandos['offset']
        if self.max_rawint > -1:
            self.index_to_key = list(range(0, self.max_rawint + 1)) + self.offset2doctag
        else:
            self.index_to_key = self.offset2doctag
        self.vectors = self.vectors_docs
        del self.doctags
        del self.vectors_docs
        del self.count
        del self.max_rawint
        del self.offset2doctag

    def similarity_unseen_docs(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Call similarity_unseen_docs on a Doc2Vec model instead.')
Word2VecKeyedVectors = KeyedVectors
Doc2VecKeyedVectors = KeyedVectors
EuclideanKeyedVectors = KeyedVectors

class CompatVocab:

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        'A single vocabulary item, used internally for collecting per-word frequency/sampling info,\n        and for constructing binary trees (incl. both word leaves and inner nodes).\n\n        Retained for now to ease the loading of older models.\n        '
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.count < other.count

    def __str__(self):
        if False:
            while True:
                i = 10
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return '%s<%s>' % (self.__class__.__name__, ', '.join(vals))
Vocab = CompatVocab

def _add_word_to_kv(kv, counts, word, weights, vocab_size):
    if False:
        for i in range(10):
            print('nop')
    if kv.has_index_for(word):
        logger.warning("duplicate word '%s' in word2vec file, ignoring all but first", word)
        return
    word_id = kv.add_vector(word, weights)
    if counts is None:
        word_count = vocab_size - word_id
    elif word in counts:
        word_count = counts[word]
    else:
        logger.warning("vocabulary file is incomplete: '%s' is missing", word)
        word_count = None
    kv.set_vecattr(word, 'count', word_count)

def _add_bytes_to_kv(kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding):
    if False:
        print('Hello World!')
    start = 0
    processed_words = 0
    bytes_per_vector = vector_size * dtype(REAL).itemsize
    max_words = vocab_size - kv.next_index
    assert max_words > 0
    for _ in range(max_words):
        i_space = chunk.find(b' ', start)
        i_vector = i_space + 1
        if i_space == -1 or len(chunk) - i_vector < bytes_per_vector:
            break
        word = chunk[start:i_space].decode(encoding, errors=unicode_errors)
        word = word.lstrip('\n')
        vector = frombuffer(chunk, offset=i_vector, count=vector_size, dtype=REAL).astype(datatype)
        _add_word_to_kv(kv, counts, word, vector, vocab_size)
        start = i_vector + bytes_per_vector
        processed_words += 1
    return (processed_words, chunk[start:])

def _word2vec_read_binary(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size, encoding='utf-8'):
    if False:
        return 10
    chunk = b''
    tot_processed_words = 0
    while tot_processed_words < vocab_size:
        new_chunk = fin.read(binary_chunk_size)
        chunk += new_chunk
        (processed_words, chunk) = _add_bytes_to_kv(kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding)
        tot_processed_words += processed_words
        if len(new_chunk) < binary_chunk_size:
            break
    if tot_processed_words != vocab_size:
        raise EOFError('unexpected end of input; is count incorrect or file otherwise damaged?')

def _word2vec_read_text(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, encoding):
    if False:
        return 10
    for line_no in range(vocab_size):
        line = fin.readline()
        if line == b'':
            raise EOFError('unexpected end of input; is count incorrect or file otherwise damaged?')
        (word, weights) = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        _add_word_to_kv(kv, counts, word, weights, vocab_size)

def _word2vec_line_to_vector(line, datatype, unicode_errors, encoding):
    if False:
        for i in range(10):
            print('nop')
    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(' ')
    (word, weights) = (parts[0], [datatype(x) for x in parts[1:]])
    return (word, weights)

def _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding):
    if False:
        i = 10
        return i + 15
    vector_size = None
    for vocab_size in itertools.count():
        line = fin.readline()
        if line == b'' or vocab_size == limit:
            break
        if vector_size:
            continue
        (word, weights) = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        vector_size = len(weights)
    return (vocab_size, vector_size)

def _load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict', limit=sys.maxsize, datatype=REAL, no_header=False, binary_chunk_size=100 * 1024):
    if False:
        print('Hello World!')
    "Load the input-hidden weight matrix from the original C word2vec-tool format.\n\n    Note that the information stored in the file is incomplete (the binary tree is missing),\n    so while you can query for word similarity etc., you cannot continue training\n    with a model loaded this way.\n\n    Parameters\n    ----------\n    fname : str\n        The file path to the saved word2vec-format file.\n    fvocab : str, optional\n        File path to the vocabulary. Word counts are read from `fvocab` filename, if set\n        (this is the file generated by `-save-vocab` flag of the original C tool).\n    binary : bool, optional\n        If True, indicates whether the data is in binary word2vec format.\n    encoding : str, optional\n        If you trained the C model using non-utf8 encoding for words, specify that encoding in `encoding`.\n    unicode_errors : str, optional\n        default 'strict', is a string suitable to be passed as the `errors`\n        argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source\n        file may include word tokens truncated in the middle of a multibyte unicode character\n        (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.\n    limit : int, optional\n        Sets a maximum number of word-vectors to read from the file. The default,\n        None, means read all.\n    datatype : type, optional\n        (Experimental) Can coerce dimensions to a non-default float type (such as `np.float16`) to save memory.\n        Such types may result in much slower bulk operations or incompatibility with optimized routines.)\n    binary_chunk_size : int, optional\n        Read input file in chunks of this many bytes for performance reasons.\n\n    Returns\n    -------\n    object\n        Returns the loaded model as an instance of :class:`cls`.\n\n    "
    counts = None
    if fvocab is not None:
        logger.info('loading word counts from %s', fvocab)
        counts = {}
        with utils.open(fvocab, 'rb') as fin:
            for line in fin:
                (word, count) = utils.to_unicode(line, errors=unicode_errors).strip().split()
                counts[word] = int(count)
    logger.info('loading projection weights from %s', fname)
    with utils.open(fname, 'rb') as fin:
        if no_header:
            if binary:
                raise NotImplementedError('no_header only available for text-format files')
            else:
                (vocab_size, vector_size) = _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding)
            fin.close()
            fin = utils.open(fname, 'rb')
        else:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            (vocab_size, vector_size) = [int(x) for x in header.split()]
        if limit:
            vocab_size = min(vocab_size, limit)
        kv = cls(vector_size, vocab_size, dtype=datatype)
        if binary:
            _word2vec_read_binary(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size, encoding)
        else:
            _word2vec_read_text(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, encoding)
    if kv.vectors.shape[0] != len(kv):
        logger.info('duplicate words detected, shrinking matrix size from %i to %i', kv.vectors.shape[0], len(kv))
        kv.vectors = ascontiguousarray(kv.vectors[:len(kv)])
    assert (len(kv), vector_size) == kv.vectors.shape
    kv.add_lifecycle_event('load_word2vec_format', msg=f'loaded {kv.vectors.shape} matrix of type {kv.vectors.dtype} from {fname}', binary=binary, encoding=encoding)
    return kv

def load_word2vec_format(*args, **kwargs):
    if False:
        return 10
    'Alias for :meth:`~gensim.models.keyedvectors.KeyedVectors.load_word2vec_format`.'
    return KeyedVectors.load_word2vec_format(*args, **kwargs)

def pseudorandom_weak_vector(size, seed_string=None, hashfxn=hash):
    if False:
        i = 10
        return i + 15
    'Get a random vector, derived deterministically from `seed_string` if supplied.\n\n    Useful for initializing KeyedVectors that will be the starting projection/input layers of _2Vec models.\n\n    '
    if seed_string:
        once = np.random.Generator(np.random.SFC64(hashfxn(seed_string) & 4294967295))
    else:
        once = utils.default_prng
    return (once.random(size).astype(REAL) - 0.5) / size

def prep_vectors(target_shape, prior_vectors=None, seed=0, dtype=REAL):
    if False:
        return 10
    'Return a numpy array of the given shape. Reuse prior_vectors object or values\n    to extent possible. Initialize new values randomly if requested.\n\n    '
    if prior_vectors is None:
        prior_vectors = np.zeros((0, 0))
    if prior_vectors.shape == target_shape:
        return prior_vectors
    (target_count, vector_size) = target_shape
    rng = np.random.default_rng(seed=seed)
    new_vectors = rng.random(target_shape, dtype=dtype)
    new_vectors *= 2.0
    new_vectors -= 1.0
    new_vectors /= vector_size
    new_vectors[0:prior_vectors.shape[0], 0:prior_vectors.shape[1]] = prior_vectors
    return new_vectors