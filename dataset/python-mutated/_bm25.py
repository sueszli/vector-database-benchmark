from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from ._feature_engineering import Transformer
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits._internal_utils import _precomputed_field
from turicreate.util import _raise_error_if_not_of_type
from . import _internal_utils
_fit_examples_doc = "\n            >>> import turicreate as tc\n\n            # Create the data\n            >>> sf = tc.SFrame(\n                {'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},\n                          {'this': 1, 'is': 1, 'another': 2, 'example': 3},\n                          {'final': 1, 'doc': 1, 'here': 2}]})\n\n            # Create a query set\n            >>> query = ['a','query','example']\n\n            # Create a BM25 encoder object\n            >>> encoder = tc.feature_engineering.BM25(feature = 'docs', query = query)\n\n            # Fit the encoder for a given dataset\n            >>> encoder = encoder.fit(data = sf)\n\n            # Return the document frequencies\n            >>> encoder['document_frequencies']\n\n            Data:\n            +----------------+---------+--------------------+\n            | feature_column |   term  | document_frequency |\n            +----------------+---------+--------------------+\n            |      docs      |    a    |         1          |\n            |      docs      | example |         1          |\n            +----------------+---------+--------------------+\n            [2 rows x 3 columns]\n"
_fit_transform_examples_doc = "\n            >>> import turicreate as tc\n\n            # Create the data\n            >>> sf = tc.SFrame(\n                {'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},\n                          {'this': 1, 'is': 1, 'another': 2, 'example': 3},\n                          {'final': 1, 'doc': 1, 'here': 2}]})\n\n            # Create a query set\n            >>> query = ['a','query','example']\n\n            # Transform the data\n            >>> encoder = tc.feature_engineering.BM25(feature = 'docs', query = query)\n            >>> encoder = encoder.fit(data = sf)\n            >>> transformed_sf = encoder.transform(data = sf)\n\n            # Alternatively, fit and transform the data in one step\n            >>> transformed_sf = tc.feature_engineering.BM25(feature = 'docs', query = query).fit_transform(data = sf)\n\n            # Display transformed data\n            >>> transformed_sf\n\n            Data:\n            +----------------+\n            |      docs      |\n            +----------------+\n            | 0.744711615513 |\n            | 0.789682123696 |\n            |      0.0       |\n            +----------------+\n            [3 rows x 1 columns]\n\n"
_transform_examples_doc = "\n            >>>import turicreate as tc\n\n            # Dictionary Input:\n            >>> sf = tc.SFrame(\n                {'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},\n                          {'this': 1, 'is': 1, 'another': 2, 'example': 3},\n                          {'final': 1, 'doc': 1, 'here': 2}]})\n            # Create a query set\n            >>> query = ['a','query','example']\n            >>> encoder = tc.feature_engineering.BM25(feature = 'docs', query = query)\n            >>> encoder = encoder.fit(data = sf)\n            >>> transformed_sf = encoder.transform(data = sf)\n            >>> transformed_sf\n\n            +----------------+\n            |      docs      |\n            +----------------+\n            | 0.744711615513 |\n            | 0.789682123696 |\n            |      0.0       |\n            +----------------+\n            [3 rows x 1 columns]\n\n            # List Input:\n            >>> l1 = ['this', 'is', 'a', 'a', 'sample']\n            >>> l2 = ['this', 'is', 'another', 'another', 'example', 'example', 'example']\n            >>> l3 = ['final', 'doc', 'here', 'here']\n            >>> sf = tc.SFrame({'docs' : [l1,l2,l3]})\n            # Create a query set\n            >>> query = ['a','query','example']\n            >>> encoder = tc.feature_engineering.BM25(feature = 'docs', query = query)\n            >>> encoder = encoder.fit(data = sf)\n            >>> transformed_sf = encoder.transform(data = sf)\n            >>> transformed_sf\n\n            +----------------+\n            |      docs      |\n            +----------------+\n            | 0.744711615513 |\n            | 0.789682123696 |\n            |      0.0       |\n            +----------------+\n            [3 rows x 1 columns]\n\n            # String Input:\n            >>> s1 = 'this is a a sample'\n            >>> s2 = 'this is another another example example example'\n            >>> s3 = 'final doc here here'\n            >>> sf = tc.SFrame({'docs' : [s1,s2,s3]})\n            # Create a query set\n            >>> query = ['a','query','example']\n            >>> encoder = tc.feature_engineering.BM25(feature = 'docs', query = query)\n            >>> encoder = encoder.fit(data = sf)\n            >>> transformed_sf = encoder.transform(data = sf)\n            >>> transformed_sf\n\n            +----------------+\n            |      docs      |\n            +----------------+\n            | 0.744711615513 |\n            | 0.789682123696 |\n            |      0.0       |\n            +----------------+\n            [3 rows x 1 columns]\n"

class BM25(Transformer):
    """
    Transform an SFrame into BM25 scores for a given query.

    If we have a query with words :math:`q_1, ..., q_n` the BM25 score for
    a document is:

    .. math:: \\sum_{i=1}^N IDF(q_i)\\frac{f(q_i) * (k_1+1)}{f(q_i) + k_1 * (1-b+b*|D|/d_{avg}))}

    where we use the natural logarithm and

      * :math:`\\mbox{IDF}(q_i) = log((N - n(q_i) + .5)/(n(q_i) + .5)` is the inverse document frequency of :math:`q_i`
      * :math:`N` is the number of documents (in the training corpus)
      * :math:`n(q_i)` is the number of documents (in the training corpus) containing :math:`q_i`
      * :math:`f(q_i)` is the number of times :math:`q_i` occurs in the document
      * :math:`|D|` is the number of words in the document
      * :math:`d_{avg}` is the average number of words per document (in the training corpus)
      * :math:`k_1` and :math:`b` are free parameters.

    The transformed output is a column of type float with the BM25 score for each document.

    The behavior of BM25 for different input data column types is as follows:

    * **dict** : Each (key, value) pair is treated as count associated with
      the key for this row. A common example is to have a dict
      element contain a bag-of-words representation of a document,
      where each key is a word and each value is the number of times
      that word occurs in the document. All non-numeric values are
      ignored.
    * **list** : The list is converted to bag of words of format, where the keys
      are the unique elements in the list and the values are the
      counts of those unique elements. After this step, the behaviour
      is identical to dict.
    * **string** : Behaves identically to a **dict**, where the dictionary is
      generated by converting the string into a bag-of-words format. For
      example, "I really like really fluffy dogs" would get converted to
      {'I' : 1, 'really': 2, 'like': 1, 'fluffy': 1, 'dogs':1}.

    Parameters
    ----------

    features : str
        Name of feature column to be transformed.

    query : A list, set, or SArray of type str
        A list, set or SArray where each element is a word.

    k1 : float, optional
        Free parameter which controls the relative importance of term frequencies.
        Recommend values are [1.2, 2.0]. Default is 1.5.

    b : float, optional
        Free parameter which controls how much to downweight scores for long documents.
        Recommend value is 0.75. Default is 0.75.

    max_document_frequency: float, optional
        The maximum ratio of document_frequency to num_documents that is
        encoded. All query terms with a document frequency higher than this are
        discarded. This value must be between 0 and 1.

    min_document_frequency: float, optional
        The minimum ratio of document_frequency to num_documents that is
        encoded. All query terms with a document frequency lower than this are
        discarded. This value must be between 0 and 1.


    output_column_name: str, optional
        The output column name of the transform. If specified, it a new column
        name with the specified column name is added to the input SFrame.
        Otherwise, the 'feature' column is overwritten.

    Returns
    -------

    out : BM25
        A BM25 object which is initialized with the defined
        parameters.

    Notes
    -----
    - `None` values are treated as separate categories and are encoded
       along with the rest of the values.

    References
    ----------

    - For more details about BM25,
      see http://en.wikipedia.org/wiki/Okapi_BM25

    See Also
    --------

    turicreate.toolkits.feature_engineering._tfidf.TFIDF,
    turicreate.toolkits.feature_engineering.create

    Examples
    --------
    .. sourcecode:: python

      >>> import turicreate as tc

      # Create data
      >>> sf = tc.SFrame(
          {'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1},
          {'this': 1, 'is': 1, 'another': 2, 'example': 3},
          {'final': 1, 'doc': 1, 'here': 2}]})

      # Create a query set
      >>> query = ['a','query','example']

      # Create a BM25 encoder
      >>> from turicreate.feature_engineering import BM25
      >>> encoder = tc.feature_engineering.create(dataset = sf, transformers = BM25('docs'))

      # Transform the data
      >>> transformed_sf = encoder.transform(data = sf)
      Data:
      +----------------+
      |      docs      |
      +----------------+
      | 0.744711615513 |
      | 0.789682123696 |
      |      0.0       |
      +----------------+
      [3 rows x 1 columns]

      # Save the transformer.
      >>> encoder.save('save-path')

      # Return the indices in the encoding.
      >>> encoder['document_frequencies']
      Data:
      +----------------+---------+--------------------+
      | feature_column |   term  | document_frequency |
      +----------------+---------+--------------------+
      |      docs      |    a    |         1          |
      |      docs      | example |         1          |
      +----------------+---------+--------------------+

    """
    _fit_examples_doc = _fit_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc
    _transform_examples_doc = _transform_examples_doc

    def __init__(self, feature, query, k1=1.5, b=0.75, min_document_frequency=0.0, max_document_frequency=1.0, output_column_name=None):
        if False:
            return 10
        if isinstance(query, _tc.SArray):
            query = list(query)
        if isinstance(query, set):
            query = list(query)
        _raise_error_if_not_of_type(feature, [str])
        for q in query:
            _raise_error_if_not_of_type(q, [str])
        _raise_error_if_not_of_type(k1, [float, int])
        _raise_error_if_not_of_type(b, [float, int])
        _raise_error_if_not_of_type(min_document_frequency, [float, int])
        _raise_error_if_not_of_type(max_document_frequency, [float, int])
        _raise_error_if_not_of_type(output_column_name, [str, type(None)])
        opts = {'features': [feature], 'query': query, 'k1': k1, 'b': b, 'min_document_frequency': min_document_frequency, 'max_document_frequency': max_document_frequency, 'output_column_name': output_column_name}
        proxy = _tc.extensions._BM25()
        proxy.init_transformer(opts)
        super(BM25, self).__init__(proxy, self.__class__)

    def _get_summary_struct(self):
        if False:
            for i in range(10):
                print('nop')
        _features = _precomputed_field(_internal_utils.pretty_print_list(self.get('features')))
        fields = [('Features', _features), ('query', 'query'), ('k1', 'k1'), ('b', 'b'), ('Minimum Document Frequency', 'min_document_frequency'), ('Maximum Document Frequency', 'max_document_frequency'), ('Output Column Name', 'output_column_name')]
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
            for i in range(10):
                print('nop')
        sf = _tc.SFrame({'docs': ['this is a test', 'this is another test']})
        encoder = _tc.feature_engineering.BM25('docs', ['a', 'test'])
        encoder = encoder.fit(sf)
        return (encoder, sf)