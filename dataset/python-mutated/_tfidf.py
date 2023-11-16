from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from ._feature_engineering import Transformer
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits._internal_utils import _precomputed_field
from turicreate.util import _raise_error_if_not_of_type
from . import _internal_utils
_fit_examples_doc = "\n            import turicreate as tc\n\n            # Create the data\n            >>> sf = tc.SFrame(\n                {'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},\n                          {'this': 1, 'is': 1, 'another': 2, 'example': 3}]})\n\n            # Create a TFIDF encoder object.\n            >>> encoder = tc.feature_engineering.TFIDF('docs')\n\n            # Fit the encoder for a given dataset.\n            >>> encoder = encoder.fit(sf)\n\n            # Return the indices in the encoding.\n            >>> encoder['document_frequencies']\n\n            Columns:\n                    feature_column  str\n                    term    str\n                    document_frequency  str\n\n            Rows: 6\n\n            Data:\n            +----------------+---------+--------------------+\n            | feature_column |   term  | document_frequency |\n            +----------------+---------+--------------------+\n            |      docs      |    is   |         2          |\n            |      docs      |    a    |         1          |\n            |      docs      |   this  |         2          |\n            |      docs      |  sample |         1          |\n            |      docs      | another |         1          |\n            |      docs      | example |         1          |\n            +----------------+---------+--------------------+\n            [6 rows x 3 columns]\n"
_fit_transform_examples_doc = "\n            import turicreate as tc\n\n            # Create the data\n            >>> sf = tc.SFrame(\n                  {'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},\n                            {'this': 1, 'is': 1, 'another': 2, 'example': 3}]})\n\n            # Transform the data\n            >>> encoder = tc.feature_engineering.TFIDF('docs')\n            >>> encoder = encoder.fit(sf)\n            >>> tfidf = encoder.transform(sf)\n            >>> tfidf[0]\n            {'a': 1.3862943611198906, 'is': 0.0,\n             'sample': 0.6931471805599453, 'this': 0.0}\n\n            # Alternatively, fit and transform the data in one step\n            >>> tfidf = tc.feature_engineering.TFIDF('docs').fit_transform(sf)\n            >>> tfidf\n            Columns:\n                    docs        dict\n\n            Rows: 2\n\n            Data:\n            +-------------------------------+\n            |              docs             |\n            +-------------------------------+\n            | {'this': 0.0, 'a': 1.38629... |\n            | {'this': 0.0, 'is': 0.0, '... |\n            +-------------------------------+\n            [2 rows x 1 columns]\n"
_transform_examples_doc = "\n            # For list columns:\n\n            >>> l1 = ['a','good','example']\n            >>> l2 = ['a','better','example']\n            >>> sf = tc.SFrame({'a' : [l1,l2]})\n            >>> tfidf = tc.feature_engineering.TFIDF('a')\n            >>> fit_tfidf = tfidf.fit(sf)\n            >>> transformed_sf = fit_tfidf.transform(sf)\n            Columns:\n                    a   dict\n\n            Rows: 2\n\n            Data:\n            +-------------------------------+\n            |               a               |\n            +-------------------------------+\n            | {'a': 0.0, 'good': 0.69314... |\n            | {'better': 0.6931471805599... |\n            +-------------------------------+\n            [2 rows x 1 columns]\n\n            # For string columns:\n\n            >>> sf = tc.SFrame({'a' : ['a good example', 'a better example']})\n            >>> tfidf = tc.feature_engineering.TFIDF('a')\n            >>> fit_tfidf = tfidf.fit(sf)\n            >>> transformed_sf = fit_tfidf.transform(sf)\n            Columns:\n                    a   dict\n\n            Rows: 2\n\n            Data:\n            +-------------------------------+\n            |               a               |\n            +-------------------------------+\n            | {'a': 0.0, 'good': 0.69314... |\n            | {'better': 0.6931471805599... |\n            +-------------------------------+\n            [2 rows x 1 columns]\n\n            # For dictionary columns:\n            >>> sf = tc.SFrame(\n                {'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},\n                          {'this': 1, 'is': 1, 'another': 2, 'example': 3}]})\n            >>> tfidf = tc.feature_engineering.TFIDF('docs')\n            >>> fit_tfidf = tfidf.fit(sf)\n            >>> transformed_sf = fit_tfidf.transform(sf)\n            Columns:\n                docs  dict\n\n            Rows: 2\n\n            Data:\n            +-------------------------------+\n            |              docs             |\n            +-------------------------------+\n            | {'this': 0.0, 'a': 1.38629... |\n            | {'this': 0.0, 'is': 0.0, '... |\n            +-------------------------------+\n            [2 rows x 1 columns]\n"

class TFIDF(Transformer):
    """
    Transform an SFrame into TF-IDF scores.

    The prototypical application of TF-IDF transformations involves
    document collections, where each element represents a document in
    bag-of-words format, i.e. a dictionary whose keys are words and whose
    values are the number of times the word occurs in the document. For more
    details, check the reference section for further reading.

    The TF-IDF transformation performs the following computation

    .. math::
        \\mbox{TF-IDF}(w, d) = tf(w, d) * log(N / f(w))

    where :math:`tf(w, d)` is the number of times word :math:`w` appeared in
    document :math:`d`, :math:`f(w)` is the number of documents word :math:`w`
    appeared in, :math:`N` is the number of documents, and we use the
    natural logarithm.

    The transformed output is a column of type dictionary
    (`max_categories` per column dimension sparse vector) where the key
    corresponds to the index of the categorical variable and the value is `1`.

    The behavior of TF-IDF for each input data column type for supported types
    is as follows. (see :func:`~turicreate.feature_engineering.TFIDF.transform`
    for examples of the same).


    * **dict** : Each (key, value) pair is treated as count associated with
      the key for this row. A common example is to have a dict element contain
      a bag-of-words representation of a document, where each key is a word
      and each value is the number of times that word occurs in the document.
      All non-numeric values are ignored.

    * **list** : The list is converted to bag of words of format, where the keys
      are the unique elements in the list and the values are the counts of
      those unique elements. After this step, the behaviour is identical to
      dict.

    * **string** : Behaves identically to a **dict**, where the dictionary is
      generated by converting the string into a bag-of-words format. For
      example, 'I really like really fluffy dogs" would get converted to
      {'I' : 1, 'really': 2, 'like': 1, 'fluffy': 1, 'dogs':1}.


    Parameters
    ----------
    features : str
        Name of feature column to be transformed.

    max_document_frequency: float
        The maximum ratio of document_frequency to num_documents that is
        encoded. All terms with a document frequency higher than this are
        discarded. This value must be between 0 and 1.

    min_document_frequency: int, optional
        The minimum ratio of document_frequency to num_documents that is
        encoded. All terms with a document frequency lower than this are
        discarded. This value must be between 0 and 1.

    output_column_prefix : str, optional
        The prefix to use for the column name of each transformed column.
        When provided, the transformation will add columns to the input data,
        where the new name is "`output_column_prefix`.original_column_name".
        If `output_column_prefix=None` (default), then the output column name
        is the same as the original feature column name.

    Returns
    -------
    out : TFIDF
        A TFIDF object which is initialized with the defined
        parameters.

    Notes
    -------
    - `None` values are treated as separate categories and are encoded along
      with the rest of the values.
    - If the SFrame to be transformed already contains a column with the
      designated output column name, then that column will be replaced with the
      new output. In particular, this means that `output_column_prefix=None` will
      overwrite the original feature columns.

    References
    ----------
    For more details about tf-idf,
    see http://en.wikipedia.org/wiki/Tf%E2%80%93idf

    See Also
    --------
    turicreate.toolkits.feature_engineering._tfidf.TFIDF,
    turicreate.toolkits.feature_engineering.create

    Examples
    --------

    .. sourcecode:: python

        >>> import turicreate as tc

        # Create the data
        >>> sf = tc.SFrame(
            {'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},
                      {'this': 1, 'is': 1, 'another': 2, 'example': 3}]})

        # Create a TFIDF encoder object.
        >>> encoder = tc.feature_engineering.TFIDF('docs')

        # Fit the encoder for a given dataset.
        >>> encoder = encoder.fit(sf)

        >>> result = transformed_sf = encoder.transform(sf)
        >>> result.print_rows(max_column_width=60)
        +-------------------------------------------------------------+
        |                             docs                            |
        +-------------------------------------------------------------+
        | {'this': 0.0, 'a': 1.3862943611198906, 'is': 0.0, 'sampl... |
        | {'this': 0.0, 'is': 0.0, 'example': 2.0794415416798357, ... |
        +-------------------------------------------------------------+

        """
    _fit_examples_doc = _fit_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc
    _transform_examples_doc = _transform_examples_doc

    def __init__(self, features=None, excluded_features=None, min_document_frequency=0.0, max_document_frequency=1.0, output_column_prefix=None):
        if False:
            for i in range(10):
                print('nop')
        (_features, _exclude) = _internal_utils.process_features(features, excluded_features)
        _raise_error_if_not_of_type(min_document_frequency, [float, int])
        _raise_error_if_not_of_type(max_document_frequency, [float, int])
        _raise_error_if_not_of_type(output_column_prefix, [str, type(None)])
        opts = {'min_document_frequency': min_document_frequency, 'max_document_frequency': max_document_frequency, 'output_column_prefix': output_column_prefix}
        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features
        proxy = _tc.extensions._TFIDF()
        proxy.init_transformer(opts)
        super(TFIDF, self).__init__(proxy, self.__class__)

    def _get_summary_struct(self):
        if False:
            i = 10
            return i + 15
        _features = _precomputed_field(_internal_utils.pretty_print_list(self.get('features')))
        fields = [('Features', _features), ('Minimum Document Frequency', 'min_document_frequency'), ('Maximum Document Frequency', 'max_document_frequency'), ('Output Column Prefix', 'output_column_prefix')]
        section_titles = ['Model fields']
        return ([fields], section_titles)

    def __repr__(self):
        if False:
            return 10
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, 30)

    @classmethod
    def _get_instance_and_data(self):
        if False:
            while True:
                i = 10
        sf = _tc.SFrame({'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1}, {'this': 1, 'is': 1, 'another': 2, 'example': 3}]})
        encoder = TFIDF(features=['docs'])
        encoder = encoder.fit(sf)
        return (encoder, sf)