from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from ._feature_engineering import Transformer
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits._internal_utils import _precomputed_field
from turicreate.toolkits._private_utils import _summarize_accessible_fields
from turicreate.util import _raise_error_if_not_of_type
from . import _internal_utils
_fit_examples_doc = "\n            >>> import turicreate as tc\n\n            # Create the data\n            >>> sf = tc.SFrame(\n            ...    {'dict': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},\n            ...              {'This': 1, 'is': 1, 'example': 1, 'EXample': 2}],\n            ...     'string': ['sentence one', 'sentence two...'],\n            ...     'list': [['one', 'One'], ['two']]})\n\n            # Create a RareWordTrimmer object that transforms all string/dict/list\n            # columns by default.\n            >>> encoder = tc.feature_engineering.RareWordTrimmer()\n\n            # Fit the encoder for a given dataset.\n            >>> trimmer = trimmer.fit(sf)\n\n            # Inspect the object and verify that it includes all columns as\n            # features.\n            >>> trimmer['features']\n            ['dict', 'list', 'string']\n\n            # Inspect the retained vocabulary\n            >>> trimmer['vocabulary']\n            Columns:\n                column  str\n                word    str\n                count   int\n\n            Rows: 6\n\n            Data:\n            +--------+----------+-------+\n            | column |   word   | count |\n            +--------+----------+-------+\n            |  dict  |   this   |   2   |\n            |  dict  |    a     |   2   |\n            |  dict  | example  |   2   |\n            |  dict  |    is    |   2   |\n            |  list  |   one    |   2   |\n            | string | sentence |   2   |\n            +--------+----------+-------+\n            [6 rows x 3 columns]\n"
_fit_transform_examples_doc = "\n            >>> import turicreate as tc\n\n            # Create the data\n            >>> sf = tc.SFrame(\n            ...    {'dict': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},\n            ...              {'This': 1, 'is': 1, 'example': 1, 'EXample': 2}],\n            ...     'string': ['sentence one', 'sentence two...'],\n            ...     'list': [['one', 'One'], ['two', 'two', 'Three']]})\n\n            # Transform the data\n            >>> trimmer = tc.feature_engineering.RareWordTrimmer()\n            >>> trimmer = trimmer.fit(sf)\n            >>> output_sf = trimmer.transform(sf)\n            >>> output_sf[0]\n            {'dict': {'a': 2, 'is': 1, 'this': 1},\n             'list': ['one', 'one'],\n             'string': 'sentence'}\n\n            # Alternatively, fit and transform the data in one step\n            >>> output2 = tc.feature_engineering.RareWordTrimmer().fit_transform(sf)\n            >>> output2\n            Columns:\n                dict    dict\n                list    list\n                string  str\n\n            Rows: 2\n\n            Data:\n            +-------------------------------+------------+----------+\n            |              dict             |    list    |  string  |\n            +-------------------------------+------------+----------+\n            |  {'this': 1, 'a': 2, 'is': 1} | [one, one] | sentence |\n            | {'this': 1, 'is': 1, 'exam... | [two, two] | sentence |\n            +-------------------------------+------------+----------+\n            [2 rows x 3 columns]\n"
_transform_examples_doc = "\n            >>> import turicreate as tc\n\n            # For list columns (string elements converted to lower case by default):\n\n            >>> l1 = ['a','good','example']\n            >>> l2 = ['a','better','example']\n            >>> sf = tc.SFrame({'a' : [l1,l2]})\n            >>> wt = tc.feature_engineering.RareWordTrimmer('a')\n            >>> fit_wt = wt.fit(sf)\n            >>> transformed_sf = fit_wt.transform(sf)\n            Columns:\n                a   list\n\n            Rows: 2\n\n            Data:\n            +--------------+\n            |      a       |\n            +--------------+\n            | [a, example] |\n            | [a, example] |\n            +--------------+\n            [2 rows x 1 columns]\n\n            # For string columns (converted to lower case by default):\n\n            >>> sf = tc.SFrame({'a' : ['a good example', 'a better example']})\n            >>> wc = tc.feature_engineering.RareWordTrimmer('a')\n            >>> fit_wt = wt.fit(sf)\n            >>> transformed_sf = fit_wt.transform(sf)\n            Columns:\n                a\tstr\n\n            Rows: 2\n\n            Data:\n            +-----------+\n            |     a     |\n            +-----------+\n            | a example |\n            | a example |\n            +-----------+\n            [2 rows x 1 columns]\n\n            # For dictionary columns (keys converted to lower case by default):\n            >>> sf = tc.SFrame(\n            ...    {'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},\n            ...              {'this': 1, 'IS': 1, 'another': 2, 'example': 3}]})\n            >>> wt = tc.feature_engineering.RareWordTrimmer('docs')\n            >>> fit_wt = wt.fit(sf)\n            >>> transformed_sf = fit_wt.transform(sf)\n            Columns:\n                docs    dict\n\n            Rows: 2\n\n            Data:\n            +-------------------------------+\n            |              docs             |\n            +-------------------------------+\n            |  {'this': 1, 'a': 2, 'is': 1} |\n            | {'this': 1, 'is': 1, 'exam... |\n            +-------------------------------+\n            [2 rows x 1 columns]\n"

class RareWordTrimmer(Transformer):
    """
    Remove words that occur below a certain number of times in a given column.
    This is a common method of cleaning text before it is used, and can increase the
    quality and explainability of the models learned on the transformed data.

    RareWordTrimmer can be applied to all the string-, dictionary-, and list-typed
    columns in a given SFrame. Its behavior for each supported input column
    type is as follows. (See :func:`~turicreate.feature_engineering.RareWordTrimmer.transform`
    for usage examples).

    * **string** : The string is first tokenized. By default, all letters are
      first converted to lower case, then tokenized by space characters. Each
      token is taken to be a word, and the words occurring below a threshold
      number of times across the entire column are removed, then the remaining
      tokens are concatenated back into a string.

    * **list** : Each element of the list must be a string, where each element
      is assumed to be a token. The remaining tokens are then filtered
      by count occurrences and a threshold value.

    * **dict** : The method first obtains the list of keys in the dictionary.
      This list is then processed as a standard list, except the value of each
      key must be of integer type and is considered to be the count of that key.

    Parameters
    ----------
    features : list[str] | str | None, optional
        Name(s) of feature column(s) to be transformed. If set to None, then all
        feature columns are used.

    excluded_features : list[str] | str | None, optional
        Name(s) of feature columns in the input dataset to be ignored. Either
        `excluded_features` or `features` can be passed, but not both.

    threshold : int, optional
        The count below which words are removed from the input.

    stopwords: list[str], optional
        A manually specified list of stopwords, which are removed regardless
        of count.

    to_lower : bool, optional
        Indicates whether to map the input strings to lower case before counting.

    delimiters: list[string], optional
        A list of delimiter characters for tokenization. By default, the list
        is defined to be the list of space characters. The user can define
        any custom list of single-character delimiters. Alternatively, setting
        `delimiters=None` will use a Penn treebank type tokenization, which
        is better at handling punctuations. (See reference below for details.)

    output_column_prefix : str, optional
        The prefix to use for the column name of each transformed column.
        When provided, the transformation will add columns to the input data,
        where the new name is "`output_column_prefix`.original_column_name".
        If `output_column_prefix=None` (default), then the output column name
        is the same as the original feature column name.

    Returns
    -------
    out : RareWordTrimmer
        A RareWordTrimmer feature engineering object which is initialized with
        the defined parameters.

    Notes
    -----
    If the SFrame to be transformed already contains a column with the
    designated output column name, then that column will be replaced with the
    new output. In particular, this means that `output_column_prefix=None` will
    overwrite the original feature columns.

    References
    ----------
    - `Penn treebank tokenization <https://web.archive.org/web/19970614072242/http://www.cis.upenn.edu:80/~treebank/tokenization.html>`_

    See Also
    --------
    turicreate.toolkits.text_analytics.count_words,
    turicreate.toolkits.feature_engineering._ngram_counter.NGramCounter,
    turicreate.toolkits.feature_engineering._tfidf.TFIDF,
    turicreate.toolkits.feature_engineering._tokenizer.Tokenizer,
    turicreate.toolkits.feature_engineering.create

    Examples
    --------

    .. sourcecode:: python

        >>> import turicreate as tc

        # Create data.
        >>> sf = tc.SFrame({
        ...    'string': ['sentences Sentences', 'another sentence another year'],
        ...    'dict': [{'bob': 1, 'Bob': 2}, {'a': 0, 'cat': 5}],
        ...    'list': [['one', 'two', 'three', 'Three'], ['a', 'cat', 'Cat']]})

        # Create a RareWordTrimmer transformer.
        >>> from turicreate.feature_engineering import RareWordTrimmer
        >>> trimmer = RareWordTrimmer()

        # Fit and transform the data.
        >>> transformed_sf = trimmer.fit_transform(sf)
        Columns:
            dict    dict
            list    list
            string  str

        Rows: 2

        Data:
        +------------+----------------+---------------------+
        |    dict    |      list      |        string       |
        +------------+----------------+---------------------+
        | {'bob': 2} | [three, three] | sentences sentences |
        | {'cat': 5} |   [cat, cat]   |   another another   |
        +------------+----------------+---------------------+
        [2 rows x 3 columns]

       # Save the transformer.
       >>> trimmer.save('save-path')
"""
    _fit_examples_doc = _fit_examples_doc
    _transform_examples_doc = _transform_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc

    def __init__(self, features=None, excluded_features=None, threshold=2, stopwords=None, to_lower=True, delimiters=['\r', '\x0b', '\n', '\x0c', '\t', ' '], output_column_prefix=None):
        if False:
            return 10
        (_features, _exclude) = _internal_utils.process_features(features, excluded_features)
        _raise_error_if_not_of_type(features, [list, str, type(None)])
        _raise_error_if_not_of_type(threshold, [int, type(None)])
        _raise_error_if_not_of_type(output_column_prefix, [str, type(None)])
        _raise_error_if_not_of_type(stopwords, [list, set, type(None)])
        _raise_error_if_not_of_type(to_lower, [bool])
        _raise_error_if_not_of_type(delimiters, [list, type(None)])
        if delimiters is not None:
            for delim in delimiters:
                _raise_error_if_not_of_type(delim, str, 'delimiters')
                if len(delim) != 1:
                    raise ValueError('Delimiters must be single-character strings')
        opts = {'threshold': threshold, 'output_column_prefix': output_column_prefix, 'to_lower': to_lower, 'stopwords': stopwords, 'delimiters': delimiters}
        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features
        proxy = _tc.extensions._RareWordTrimmer()
        proxy.init_transformer(opts)
        super(RareWordTrimmer, self).__init__(proxy, self.__class__)

    def _get_summary_struct(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a structured description of the model, including (where relevant)\n        the schema of the training data, description of the training data,\n        training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        _features = _precomputed_field(_internal_utils.pretty_print_list(self.get('features')))
        _exclude = _precomputed_field(_internal_utils.pretty_print_list(self.get('excluded_features')))
        _stopwords = _precomputed_field(_internal_utils.pretty_print_list(self.get('stopwords')))
        fields = [('Features', _features), ('Excluded features', _exclude), ('Output column name', 'output_column_prefix'), ('Word count threshold', 'threshold'), ('Manually specified stopwords', _stopwords), ('Whether to convert to lowercase', 'to_lower'), ('Delimiters', 'delimiters')]
        section_titles = ['Model fields']
        return ([fields], section_titles)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a string description of the model, including a description of\n        the training data, training statistics, and model hyper-parameters.\n\n        Returns\n        -------\n        out : string\n            A description of the model.\n        '
        accessible_fields = {'vocabulary': 'The vocabulary of the trimmed input.'}
        (sections, section_titles) = self._get_summary_struct()
        out = _toolkit_repr_print(self, sections, section_titles, width=30)
        out2 = _summarize_accessible_fields(accessible_fields, width=30)
        return out + '\n' + out2

    @classmethod
    def _get_instance_and_data(cls):
        if False:
            while True:
                i = 10
        sf = _tc.SFrame({'a': ['dog', 'dog', 'dog'], 'b': ['cat', 'one', 'one']})
        trimmer = RareWordTrimmer(features=['a', 'b'])
        return (trimmer.fit(sf), sf)