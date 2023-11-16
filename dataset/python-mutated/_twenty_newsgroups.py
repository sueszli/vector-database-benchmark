"""Caching loader for the 20 newsgroups text classification dataset.


The description of the dataset is available on the official website at:

    http://people.csail.mit.edu/jrennie/20Newsgroups/

Quoting the introduction:

    The 20 Newsgroups data set is a collection of approximately 20,000
    newsgroup documents, partitioned (nearly) evenly across 20 different
    newsgroups. To the best of my knowledge, it was originally collected
    by Ken Lang, probably for his Newsweeder: Learning to filter netnews
    paper, though he does not explicitly mention this collection. The 20
    newsgroups collection has become a popular data set for experiments
    in text applications of machine learning techniques, such as text
    classification and text clustering.

This dataset loader will download the recommended "by date" variant of the
dataset and which features a point in time split between the train and
test sets. The compressed dataset size is around 14 Mb compressed. Once
uncompressed the train set is 52 MB and the test set is 34 MB.
"""
import codecs
import logging
import os
import pickle
import re
import shutil
import tarfile
from contextlib import suppress
import joblib
import numpy as np
import scipy.sparse as sp
from .. import preprocessing
from ..feature_extraction.text import CountVectorizer
from ..utils import Bunch, check_random_state
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home, load_files
from ._base import RemoteFileMetadata, _convert_data_dataframe, _fetch_remote, _pkl_filepath, load_descr
logger = logging.getLogger(__name__)
ARCHIVE = RemoteFileMetadata(filename='20news-bydate.tar.gz', url='https://ndownloader.figshare.com/files/5975967', checksum='8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610')
CACHE_NAME = '20news-bydate.pkz'
TRAIN_FOLDER = '20news-bydate-train'
TEST_FOLDER = '20news-bydate-test'

def _download_20newsgroups(target_dir, cache_path):
    if False:
        print('Hello World!')
    'Download the 20 newsgroups data and stored it as a zipped pickle.'
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)
    os.makedirs(target_dir, exist_ok=True)
    logger.info('Downloading dataset from %s (14 MB)', ARCHIVE.url)
    archive_path = _fetch_remote(ARCHIVE, dirname=target_dir)
    logger.debug('Decompressing %s', archive_path)
    tarfile.open(archive_path, 'r:gz').extractall(path=target_dir)
    with suppress(FileNotFoundError):
        os.remove(archive_path)
    cache = dict(train=load_files(train_path, encoding='latin1'), test=load_files(test_path, encoding='latin1'))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)
    shutil.rmtree(target_dir)
    return cache

def strip_newsgroup_header(text):
    if False:
        return 10
    '\n    Given text in "news" format, strip the headers, by removing everything\n    before the first blank line.\n\n    Parameters\n    ----------\n    text : str\n        The text from which to remove the signature block.\n    '
    (_before, _blankline, after) = text.partition('\n\n')
    return after
_QUOTE_RE = re.compile('(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\\||^>)')

def strip_newsgroup_quoting(text):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given text in "news" format, strip lines beginning with the quote\n    characters > or |, plus lines that often introduce a quoted section\n    (for example, because they contain the string \'writes:\'.)\n\n    Parameters\n    ----------\n    text : str\n        The text from which to remove the signature block.\n    '
    good_lines = [line for line in text.split('\n') if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)

def strip_newsgroup_footer(text):
    if False:
        i = 10
        return i + 15
    '\n    Given text in "news" format, attempt to remove a signature block.\n\n    As a rough heuristic, we assume that signatures are set apart by either\n    a blank line or a line made of hyphens, and that it is the last such line\n    in the file (disregarding blank lines at the end).\n\n    Parameters\n    ----------\n    text : str\n        The text from which to remove the signature block.\n    '
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break
    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text

@validate_params({'data_home': [str, os.PathLike, None], 'subset': [StrOptions({'train', 'test', 'all'})], 'categories': ['array-like', None], 'shuffle': ['boolean'], 'random_state': ['random_state'], 'remove': [tuple], 'download_if_missing': ['boolean'], 'return_X_y': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_20newsgroups(*, data_home=None, subset='train', categories=None, shuffle=True, random_state=42, remove=(), download_if_missing=True, return_X_y=False):
    if False:
        for i in range(10):
            print('nop')
    "Load the filenames and data from the 20 newsgroups dataset (classification).\n\n    Download it if necessary.\n\n    =================   ==========\n    Classes                     20\n    Samples total            18846\n    Dimensionality               1\n    Features                  text\n    =================   ==========\n\n    Read more in the :ref:`User Guide <20newsgroups_dataset>`.\n\n    Parameters\n    ----------\n    data_home : str or path-like, default=None\n        Specify a download and cache folder for the datasets. If None,\n        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.\n\n    subset : {'train', 'test', 'all'}, default='train'\n        Select the dataset to load: 'train' for the training set, 'test'\n        for the test set, 'all' for both, with shuffled ordering.\n\n    categories : array-like, dtype=str, default=None\n        If None (default), load all the categories.\n        If not None, list of category names to load (other categories\n        ignored).\n\n    shuffle : bool, default=True\n        Whether or not to shuffle the data: might be important for models that\n        make the assumption that the samples are independent and identically\n        distributed (i.i.d.), such as stochastic gradient descent.\n\n    random_state : int, RandomState instance or None, default=42\n        Determines random number generation for dataset shuffling. Pass an int\n        for reproducible output across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    remove : tuple, default=()\n        May contain any subset of ('headers', 'footers', 'quotes'). Each of\n        these are kinds of text that will be detected and removed from the\n        newsgroup posts, preventing classifiers from overfitting on\n        metadata.\n\n        'headers' removes newsgroup headers, 'footers' removes blocks at the\n        ends of posts that look like signatures, and 'quotes' removes lines\n        that appear to be quoting another post.\n\n        'headers' follows an exact standard; the other filters are not always\n        correct.\n\n    download_if_missing : bool, default=True\n        If False, raise an OSError if the data is not locally available\n        instead of trying to download the data from the source site.\n\n    return_X_y : bool, default=False\n        If True, returns `(data.data, data.target)` instead of a Bunch\n        object.\n\n        .. versionadded:: 0.22\n\n    Returns\n    -------\n    bunch : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        data : list of shape (n_samples,)\n            The data list to learn.\n        target: ndarray of shape (n_samples,)\n            The target labels.\n        filenames: list of shape (n_samples,)\n            The path to the location of the data.\n        DESCR: str\n            The full description of the dataset.\n        target_names: list of shape (n_classes,)\n            The names of target classes.\n\n    (data, target) : tuple if `return_X_y=True`\n        A tuple of two ndarrays. The first contains a 2D array of shape\n        (n_samples, n_classes) with each row representing one sample and each\n        column representing the features. The second array of shape\n        (n_samples,) contains the target samples.\n\n        .. versionadded:: 0.22\n    "
    data_home = get_data_home(data_home=data_home)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    twenty_home = os.path.join(data_home, '20news_home')
    cache = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)
    if cache is None:
        if download_if_missing:
            logger.info('Downloading 20news dataset. This may take a few minutes.')
            cache = _download_20newsgroups(target_dir=twenty_home, cache_path=cache_path)
        else:
            raise OSError('20Newsgroups dataset not found')
    if subset in ('train', 'test'):
        data = cache[subset]
    elif subset == 'all':
        data_lst = list()
        target = list()
        filenames = list()
        for subset in ('train', 'test'):
            data = cache[subset]
            data_lst.extend(data.data)
            target.extend(data.target)
            filenames.extend(data.filenames)
        data.data = data_lst
        data.target = np.array(target)
        data.filenames = np.array(filenames)
    fdescr = load_descr('twenty_newsgroups.rst')
    data.DESCR = fdescr
    if 'headers' in remove:
        data.data = [strip_newsgroup_header(text) for text in data.data]
    if 'footers' in remove:
        data.data = [strip_newsgroup_footer(text) for text in data.data]
    if 'quotes' in remove:
        data.data = [strip_newsgroup_quoting(text) for text in data.data]
    if categories is not None:
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        labels.sort()
        (labels, categories) = zip(*labels)
        mask = np.isin(data.target, labels)
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()
    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()
    if return_X_y:
        return (data.data, data.target)
    return data

@validate_params({'subset': [StrOptions({'train', 'test', 'all'})], 'remove': [tuple], 'data_home': [str, os.PathLike, None], 'download_if_missing': ['boolean'], 'return_X_y': ['boolean'], 'normalize': ['boolean'], 'as_frame': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_20newsgroups_vectorized(*, subset='train', remove=(), data_home=None, download_if_missing=True, return_X_y=False, normalize=True, as_frame=False):
    if False:
        return 10
    "Load and vectorize the 20 newsgroups dataset (classification).\n\n    Download it if necessary.\n\n    This is a convenience function; the transformation is done using the\n    default settings for\n    :class:`~sklearn.feature_extraction.text.CountVectorizer`. For more\n    advanced usage (stopword filtering, n-gram extraction, etc.), combine\n    fetch_20newsgroups with a custom\n    :class:`~sklearn.feature_extraction.text.CountVectorizer`,\n    :class:`~sklearn.feature_extraction.text.HashingVectorizer`,\n    :class:`~sklearn.feature_extraction.text.TfidfTransformer` or\n    :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.\n\n    The resulting counts are normalized using\n    :func:`sklearn.preprocessing.normalize` unless normalize is set to False.\n\n    =================   ==========\n    Classes                     20\n    Samples total            18846\n    Dimensionality          130107\n    Features                  real\n    =================   ==========\n\n    Read more in the :ref:`User Guide <20newsgroups_dataset>`.\n\n    Parameters\n    ----------\n    subset : {'train', 'test', 'all'}, default='train'\n        Select the dataset to load: 'train' for the training set, 'test'\n        for the test set, 'all' for both, with shuffled ordering.\n\n    remove : tuple, default=()\n        May contain any subset of ('headers', 'footers', 'quotes'). Each of\n        these are kinds of text that will be detected and removed from the\n        newsgroup posts, preventing classifiers from overfitting on\n        metadata.\n\n        'headers' removes newsgroup headers, 'footers' removes blocks at the\n        ends of posts that look like signatures, and 'quotes' removes lines\n        that appear to be quoting another post.\n\n    data_home : str or path-like, default=None\n        Specify an download and cache folder for the datasets. If None,\n        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.\n\n    download_if_missing : bool, default=True\n        If False, raise an OSError if the data is not locally available\n        instead of trying to download the data from the source site.\n\n    return_X_y : bool, default=False\n        If True, returns ``(data.data, data.target)`` instead of a Bunch\n        object.\n\n        .. versionadded:: 0.20\n\n    normalize : bool, default=True\n        If True, normalizes each document's feature vector to unit norm using\n        :func:`sklearn.preprocessing.normalize`.\n\n        .. versionadded:: 0.22\n\n    as_frame : bool, default=False\n        If True, the data is a pandas DataFrame including columns with\n        appropriate dtypes (numeric, string, or categorical). The target is\n        a pandas DataFrame or Series depending on the number of\n        `target_columns`.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n    bunch : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        data: {sparse matrix, dataframe} of shape (n_samples, n_features)\n            The input data matrix. If ``as_frame`` is `True`, ``data`` is\n            a pandas DataFrame with sparse columns.\n        target: {ndarray, series} of shape (n_samples,)\n            The target labels. If ``as_frame`` is `True`, ``target`` is a\n            pandas Series.\n        target_names: list of shape (n_classes,)\n            The names of target classes.\n        DESCR: str\n            The full description of the dataset.\n        frame: dataframe of shape (n_samples, n_features + 1)\n            Only present when `as_frame=True`. Pandas DataFrame with ``data``\n            and ``target``.\n\n            .. versionadded:: 0.24\n\n    (data, target) : tuple if ``return_X_y`` is True\n        `data` and `target` would be of the format defined in the `Bunch`\n        description above.\n\n        .. versionadded:: 0.20\n    "
    data_home = get_data_home(data_home=data_home)
    filebase = '20newsgroup_vectorized'
    if remove:
        filebase += 'remove-' + '-'.join(remove)
    target_file = _pkl_filepath(data_home, filebase + '.pkl')
    data_train = fetch_20newsgroups(data_home=data_home, subset='train', categories=None, shuffle=True, random_state=12, remove=remove, download_if_missing=download_if_missing)
    data_test = fetch_20newsgroups(data_home=data_home, subset='test', categories=None, shuffle=True, random_state=12, remove=remove, download_if_missing=download_if_missing)
    if os.path.exists(target_file):
        try:
            (X_train, X_test, feature_names) = joblib.load(target_file)
        except ValueError as e:
            raise ValueError(f'The cached dataset located in {target_file} was fetched with an older scikit-learn version and it is not compatible with the scikit-learn version imported. You need to manually delete the file: {target_file}.') from e
    else:
        vectorizer = CountVectorizer(dtype=np.int16)
        X_train = vectorizer.fit_transform(data_train.data).tocsr()
        X_test = vectorizer.transform(data_test.data).tocsr()
        feature_names = vectorizer.get_feature_names_out()
        joblib.dump((X_train, X_test, feature_names), target_file, compress=9)
    if normalize:
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        preprocessing.normalize(X_train, copy=False)
        preprocessing.normalize(X_test, copy=False)
    target_names = data_train.target_names
    if subset == 'train':
        data = X_train
        target = data_train.target
    elif subset == 'test':
        data = X_test
        target = data_test.target
    elif subset == 'all':
        data = sp.vstack((X_train, X_test)).tocsr()
        target = np.concatenate((data_train.target, data_test.target))
    fdescr = load_descr('twenty_newsgroups.rst')
    frame = None
    target_name = ['category_class']
    if as_frame:
        (frame, data, target) = _convert_data_dataframe('fetch_20newsgroups_vectorized', data, target, feature_names, target_names=target_name, sparse_data=True)
    if return_X_y:
        return (data, target)
    return Bunch(data=data, target=target, frame=frame, target_names=target_names, feature_names=feature_names, DESCR=fdescr)