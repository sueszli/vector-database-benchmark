from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from ._feature_engineering import Transformer
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits._internal_utils import _precomputed_field
from turicreate.util import _raise_error_if_not_of_type
from . import _internal_utils
_fit_examples_doc = '\n# Create data\n>>> sf = turicreate.SFrame({\'values\': [{"a" : {"b" : 3}, "c": 2},\n...                      { "a" : { "b" : 3, "c" : 2.5 }, "c" : 2 },\n...                      {"a" : [1,2,4] , "c" : 2 },\n...                      { "a" : "b", "c" : 2 }]})\n\n>>> ft = turicreate.feature_engineering.TransformToFlatDictionary(features = [\'values\'])\n>>> ft = ft.fit(sf)\n>>> ft\nClass                          : TransformToFlatDictionary\n\nModel fields\n------------\nFeatures                       : [\'values\']\nSeparator                      : .\nNone Tag                       : __none__\nOutput Column Prefix           :\n\n'
_fit_transform_examples_doc = '\n# Create data\n>>> sf = turicreate.SFrame({\'values\': [{"a" : {"b" : 3}, "c": 2},\n...                      { "a" : { "b" : 3, "c" : 2.5 }, "c" : 2 },\n...                      {"a" : [1,2,4] , "c" : 2 },\n...                      { "a" : "b", "c" : 2 }]})\n\n>>> ft = turicreate.feature_engineering.TransformToFlatDictionary(features = [\'values\'])\n\n>>> ft.fit_transform(sf)\nColumns:\n        values  dict\n\nRows: 4\n\nData:\n+--------------------------------+\n|             values             |\n+--------------------------------+\n|       {\'c\': 2, \'a.b\': 3}       |\n| {\'c\': 2, \'a.b\': 3, \'a.c\': 2.5} |\n| {\'c\': 2, \'a.0\': 1.0, \'a.1\'...  |\n|       {\'c\': 2, \'a.b\': 1}       |\n+--------------------------------+\n[4 rows x 1 columns]\n'
_transform_examples_doc = '\n# Create data\n>>> sf = turicreate.SFrame({\'values\': [{"a" : {"b" : 3}, "c": 2},\n...                      { "a" : { "b" : 3, "c" : 2.5 }, "c" : 2 },\n...                      {"a" : [1,2,4] , "c" : 2 },\n...                      { "a" : "b", "c" : 2 }]})\n\n>>> ft = turicreate.feature_engineering.TransformToFlatDictionary(features = [\'values\'])\n>>> ft = ft.fit(sf)\n>>> ft\nClass                          : TransformToFlatDictionary\n\nModel fields\n------------\nFeatures                       : [\'values\']\nSeparator                      : .\nNone Tag                       : __none__\nOutput Column Prefix           :\n\n>>> ft.transform(sf)\nColumns:\n        values  dict\n\nRows: 4\n\nData:\n+--------------------------------+\n|             values             |\n+--------------------------------+\n|       {\'c\': 2, \'a.b\': 3}       |\n| {\'c\': 2, \'a.b\': 3, \'a.c\': 2.5} |\n| {\'c\': 2, \'a.0\': 1.0, \'a.1\'...  |\n|       {\'c\': 2, \'a.b\': 1}       |\n+--------------------------------+\n[4 rows x 1 columns]\n'

class TransformToFlatDictionary(Transformer):
    """
    Transforms column values into dictionaries with flat, non-nested
    string keys and numeric values.  Each key in nested containers is a
    concatenation of the keys in each dictionary with `separator`
    separating them.  For example, if ``separator = "."``, then

      {"a" : {"b" : 1}, "c" : 2}

    becomes

      {"a.b" : 1, "c" : 2}.

    - List and vector elements are handled by converting the index of
      the appropriate element to a string, then treating that as the key.

    - String values are handled by treating them as a single
      {"string_value" : 1} pair.

    - None values are handled by replacing them with the
      string contents of `none_tag`.

    - image and datetime values are currently not supported and raise an
      error.


    Parameters
    ----------
    features : list, str
        Name of feature column(s) to be transformed.

    exclude : list, str
        Names of feature column(s) to be excluded from the transformation.

    separator : str
        The separator string added between keys of nested dicts or lists.

    output_column_prefix : str, optional
        The prefix to use for the column name of each transformed column.
        When provided, the transformation will add columns to the input data,
        where the new name is "`output_column_prefix`.original_column_name".
        If `output_column_prefix=None` (default), then the output column name
        is the same as the original feature column name.

    Returns
    -------
    out : TransformToFlatDictionary
        A TransformToFlatDictionary object which is initialized with the defined
        parameters.

    Examples
    --------

    .. sourcecode:: python

        >>> import turicreate as tc

        # Create the data
        >>> sf = tc.SFrame(
            {'values': [{"a" : {"b" : 3}, "c": 2},
                        { "a" : { "b" : 3, "c" : 2.5 }, "c" : 2 },
                        {"a" : [1,2,4] , "c" : 2 },
                        { "a" : "b", "c" : 2 }]}

        # Create a TransformToFlatDictionary transformer object.
        >>> ft = tc.feature_engineering.TransformToFlatDictionary('values')

        # Fit the encoder for a given dataset.
        >>> ft = ft.fit(sf)

        >>> transformed_sf = ft.transform(sf)
        >>> transformed_sf.print_rows(max_column_width=60)
        +----------------------------------------------+
        |                    values                    |
        +----------------------------------------------+
        |              {'c': 2, 'a.b': 3}              |
        |        {'c': 2, 'a.b': 3, 'a.c': 2.5}        |
        | {'c': 2, 'a.0': 1.0, 'a.1': 2.0, 'a.2': 4.0} |
        |              {'c': 2, 'a.b': 1}              |
        +----------------------------------------------+
        [4 rows x 1 columns]
        """
    _fit_examples_doc = _fit_examples_doc
    _fit_transform_examples_doc = _fit_transform_examples_doc
    _transform_examples_doc = _transform_examples_doc

    def __init__(self, features=None, excluded_features=None, separator='.', none_tag='__none__', output_column_prefix=None):
        if False:
            print('Hello World!')
        (_features, _exclude) = _internal_utils.process_features(features, excluded_features)
        _raise_error_if_not_of_type(output_column_prefix, [str, type(None)])
        if output_column_prefix is None:
            output_column_prefix = ''
        opts = {'separator': separator, 'none_tag': none_tag, 'output_column_prefix': output_column_prefix}
        if _exclude:
            opts['exclude'] = True
            opts['features'] = _exclude
        else:
            opts['exclude'] = False
            opts['features'] = _features
        proxy = _tc.extensions._TransformToFlatDictionary()
        proxy.init_transformer(opts)
        super(TransformToFlatDictionary, self).__init__(proxy, self.__class__)

    def _get_summary_struct(self):
        if False:
            print('Hello World!')
        _features = _precomputed_field(_internal_utils.pretty_print_list(self.get('features')))
        _exclude = _precomputed_field(_internal_utils.pretty_print_list(self.get('excluded_features')))
        fields = [('Features', _features), ('Excluded_features', _exclude), ('Separator', 'separator'), ('None Tag', 'none_tag'), ('Output Column Prefix', 'output_column_prefix')]
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
            print('Hello World!')
        sf = _tc.SFrame({'docs': [{'this': 1, 'is': 1, 'a': 2, 'sample': 1}, {'this': 1, 'is': 1, 'another': 2, 'example': 3}]})
        encoder = _tc.feature_engineering.TFIDF(features=['docs'])
        encoder = encoder.fit(sf)
        return (encoder, sf)