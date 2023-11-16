"""
Methods for creating and querying a nearest neighbors model.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _turicreate
from turicreate.toolkits._model import Model as _Model
from turicreate.data_structures.sgraph import SGraph as _SGraph
import turicreate.toolkits._internal_utils as _tkutl
from turicreate.toolkits._private_utils import _validate_row_label
from turicreate.toolkits._private_utils import _validate_lists
from turicreate.toolkits._private_utils import _robust_column_name
from turicreate.toolkits._main import ToolkitError as _ToolkitError
from turicreate.toolkits.distances._util import _convert_distance_names_to_functions
from turicreate.toolkits.distances._util import _validate_composite_distance
from turicreate.toolkits.distances._util import _scrub_composite_distance_features
from turicreate.toolkits.distances._util import _get_composite_distance_features
from turicreate._cython.cy_server import QuietProgress
import array
import copy as _copy
import six as _six

def _construct_auto_distance(feature_names, column_names, column_types, sample):
    if False:
        for i in range(10):
            print('nop')
    '\n    Construct composite distance parameters based on selected features and their\n    types.\n    '
    col_type_dict = {k: v for (k, v) in zip(column_names, column_types)}
    composite_distance_params = []
    numeric_cols = []
    for c in feature_names:
        if col_type_dict[c] == str:
            composite_distance_params.append([[c], _turicreate.distances.levenshtein, 1])
        elif col_type_dict[c] == dict:
            composite_distance_params.append([[c], _turicreate.distances.jaccard, 1])
        elif col_type_dict[c] == array.array:
            composite_distance_params.append([[c], _turicreate.distances.euclidean, 1])
        elif col_type_dict[c] == list:
            only_str_lists = _validate_lists(sample[c], allowed_types=[str])
            if not only_str_lists:
                raise TypeError('Only lists of all str objects are currently supported')
            composite_distance_params.append([[c], _turicreate.distances.jaccard, 1])
        elif col_type_dict[c] in [int, float, array.array, list]:
            numeric_cols.append(c)
        else:
            raise TypeError('Unable to automatically determine a distance ' + 'for column {}'.format(c))
    if len(numeric_cols) > 0:
        composite_distance_params.append([numeric_cols, _turicreate.distances.euclidean, 1])
    return composite_distance_params

def create(dataset, label=None, features=None, distance=None, method='auto', verbose=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    Create a nearest neighbor model, which can be searched efficiently and\n    quickly for the nearest neighbors of a query observation. If the `method`\n    argument is specified as `auto`, the type of model is chosen automatically\n    based on the type of data in `dataset`.\n\n    Parameters\n    ----------\n    dataset : SFrame\n        Reference data. If the features for each observation are numeric, they\n        may be in separate columns of 'dataset' or a single column with lists\n        of values. The features may also be in the form of a column of sparse\n        vectors (i.e. dictionaries), with string keys and numeric values.\n\n    label : string, optional\n        Name of the SFrame column with row labels. If 'label' is not specified,\n        row numbers are used to identify reference dataset rows when the model\n        is queried.\n\n    features : list[string], optional\n        Name of the columns with features to use in computing distances between\n        observations and the query points. 'None' (the default) indicates that\n        all columns except the label should be used as features. Each column\n        can be one of the following types:\n\n        - *Numeric*: values of numeric type integer or float.\n\n        - *Array*: list of numeric (integer or float) values. Each list element\n          is treated as a separate variable in the model.\n\n        - *Dictionary*: key-value pairs with numeric (integer or float) values.\n          Each key indicates a separate variable in the model.\n\n        - *List*: list of integer or string values. Each element is treated as\n          a separate variable in the model.\n\n        - *String*: string values.\n\n        Please note: if a composite distance is also specified, this parameter\n        is ignored.\n\n    distance : string, function, or list[list], optional\n        Function to measure the distance between any two input data rows. This\n        may be one of three types:\n\n        - *String*: the name of a standard distance function. One of\n          'euclidean', 'squared_euclidean', 'manhattan', 'levenshtein',\n          'jaccard', 'weighted_jaccard', 'cosine' or 'transformed_dot_product'.\n\n        - *Function*: a function handle from the\n          :mod:`~turicreate.toolkits.distances` module.\n\n        - *Composite distance*: the weighted sum of several standard distance\n          functions applied to various features. This is specified as a list of\n          distance components, each of which is itself a list containing three\n          items:\n\n          1. list or tuple of feature names (strings)\n\n          2. standard distance name (string)\n\n          3. scaling factor (int or float)\n\n        For more information about Turi Create distance functions, please\n        see the :py:mod:`~turicreate.toolkits.distances` module.\n\n        If 'distance' is left unspecified or set to 'auto', a composite\n        distance is constructed automatically based on feature types.\n\n    method : {'auto', 'ball_tree', 'brute_force', 'lsh'}, optional\n        Method for computing nearest neighbors. The options are:\n\n        - *auto* (default): the method is chosen automatically, based on the\n          type of data and the distance. If the distance is 'manhattan' or\n          'euclidean' and the features are numeric or vectors of numeric\n          values, then the 'ball_tree' method is used. Otherwise, the\n          'brute_force' method is used.\n\n        - *ball_tree*: use a tree structure to find the k-closest neighbors to\n          each query point. The ball tree model is slower to construct than the\n          brute force model, but queries are faster than linear time. This\n          method is not applicable for the cosine  or transformed_dot_product\n          distances.\n          See `Liu, et al (2004)\n          <http://papers.nips.cc/paper/2666-an-investigation-of-p\n          ractical-approximat e-nearest-neighbor-algorithms>`_ for\n          implementation details.\n\n        - *brute_force*: compute the distance from a query point to all\n          reference observations. There is no computation time for model\n          creation with the brute force method (although the reference data is\n          held in the model, but each query takes linear time.\n\n        - *lsh*: use Locality Sensitive Hashing (LSH) to find approximate\n          nearest neighbors efficiently. The LSH model supports 'euclidean',\n          'squared_euclidean', 'manhattan', 'cosine', 'jaccard', and\n          'transformed_dot_product' distances. Two options are provided for\n          LSH -- ``num_tables`` and ``num_projections_per_table``. See the\n          notes below for details.\n\n    verbose: bool, optional\n        If True, print progress updates and model details.\n\n    **kwargs : optional\n        Options for the distance function and query method.\n\n        - *leaf_size*: for the ball tree method, the number of points in each\n          leaf of the tree. The default is to use the max of 1,000 and\n          n/(2^11), which ensures a maximum tree depth of 12.\n\n        - *num_tables*: For the LSH method, the number of hash tables\n          constructed. The default value is 20. We recommend choosing values\n          from 10 to 30.\n\n        - *num_projections_per_table*: For the LSH method, the number of\n          projections/hash functions for each hash table. The default value is\n          4 for 'jaccard' distance, 16 for 'cosine' distance and 8 for other\n          distances. We recommend using number 2 ~ 6 for 'jaccard' distance, 8\n          ~ 20 for 'cosine' distance and 4 ~ 12 for other distances.\n\n    Returns\n    -------\n    out : NearestNeighborsModel\n        A structure for efficiently computing the nearest neighbors in 'dataset'\n        of new query points.\n\n    See Also\n    --------\n    NearestNeighborsModel.query, turicreate.toolkits.distances\n\n    Notes\n    -----\n    - Missing data is not allowed in the 'dataset' provided to this function.\n      Please use the :func:`turicreate.SFrame.fillna` and\n      :func:`turicreate.SFrame.dropna` utilities to handle missing data before\n      creating a nearest neighbors model.\n\n    - Missing keys in sparse vectors are assumed to have value 0.\n\n    - If the features should be weighted equally in the distance calculations\n      but are measured on different scales, it is important to standardize the\n      features. One way to do this is to subtract the mean of each column and\n      divide by the standard deviation.\n\n    **Locality Sensitive Hashing (LSH)**\n\n    There are several efficient nearest neighbors search algorithms that work\n    well for data with low dimensions :math:`d` (approximately 50). However,\n    most of the solutions suffer from either space or query time that is\n    exponential in :math:`d`. For large :math:`d`, they often provide little,\n    if any, improvement over the 'brute_force' method. This is a well-known\n    consequence of the phenomenon called `The Curse of Dimensionality`.\n\n    `Locality Sensitive Hashing (LSH)\n    <https://en.wikipedia.org/wiki/Locality-sensitive_hashing>`_ is an approach\n    that is designed to efficiently solve the *approximate* nearest neighbor\n    search problem for high dimensional data. The key idea of LSH is to hash\n    the data points using several hash functions, so that the probability of\n    collision is much higher for data points which are close to each other than\n    those which are far apart.\n\n    An LSH family is a family of functions :math:`h` which map points from the\n    metric space to a bucket, so that\n\n    - if :math:`d(p, q) \\leq R`, then :math:`h(p) = h(q)` with at least probability :math:`p_1`.\n    - if :math:`d(p, q) \\geq cR`, then :math:`h(p) = h(q)` with probability at most :math:`p_2`.\n\n    LSH for efficient approximate nearest neighbor search:\n\n    - We define a new family of hash functions :math:`g`, where each\n      function :math:`g` is obtained by concatenating :math:`k` functions\n      :math:`h_1, ..., h_k`, i.e., :math:`g(p)=[h_1(p),...,h_k(p)]`.\n      The algorithm constructs :math:`L` hash tables, each of which\n      corresponds to a different randomly chosen hash function :math:`g`.\n      There are :math:`k \\cdot L` hash functions used in total.\n\n    - In the preprocessing step, we hash all :math:`n` reference points\n      into each of the :math:`L` hash tables.\n\n    - Given a query point :math:`q`, the algorithm iterates over the\n      :math:`L` hash functions :math:`g`. For each :math:`g` considered, it\n      retrieves the data points that are hashed into the same bucket as q.\n      These data points from all the :math:`L` hash tables are considered as\n      candidates that are then re-ranked by their real distances with the query\n      data.\n\n    **Note** that the number of tables :math:`L` and the number of hash\n    functions per table :math:`k` are two main parameters. They can be set\n    using the options ``num_tables`` and ``num_projections_per_table``\n    respectively.\n\n    Hash functions for different distances:\n\n    - `euclidean` and `squared_euclidean`:\n      :math:`h(q) = \\lfloor \\frac{a \\cdot q + b}{w} \\rfloor` where\n      :math:`a` is a vector, of which the elements are independently\n      sampled from normal distribution, and :math:`b` is a number\n      uniformly sampled from :math:`[0, r]`. :math:`r` is a parameter for the\n      bucket width. We set :math:`r` using the average all-pair `euclidean`\n      distances from a small randomly sampled subset of the reference data.\n\n    - `manhattan`: The hash function of `manhattan` is similar with that of\n      `euclidean`. The only difference is that the elements of `a` are sampled\n      from Cauchy distribution, instead of normal distribution.\n\n    - `cosine`: Random Projection is designed to approximate the cosine\n      distance between vectors. The hash function is :math:`h(q) = sgn(a \\cdot\n      q)`, where :math:`a` is randomly sampled normal unit vector.\n\n    - `jaccard`: We use a recently proposed method one permutation hashing by\n      Shrivastava and Li. See the paper `[Shrivastava and Li, UAI 2014]\n      <http://www.auai.org/uai2014/proceedings/individuals/225.pdf>`_ for\n      details.\n\n    References\n    ----------\n    - `Wikipedia - nearest neighbor\n      search <http://en.wikipedia.org/wiki/Nearest_neighbor_search>`_\n\n    - `Wikipedia - ball tree <http://en.wikipedia.org/wiki/Ball_tree>`_\n\n    - Ball tree implementation: Liu, T., et al. (2004) `An Investigation of\n      Practical Approximate Nearest Neighbor Algorithms\n      <http://papers.nips.cc/paper/2666-an-investigation-of-p\n      ractical-approximat e-nearest-neighbor-algorithms>`_. Advances in Neural\n      Information Processing Systems pp. 825-832.\n\n    - `Wikipedia - Jaccard distance\n      <http://en.wikipedia.org/wiki/Jaccard_index>`_\n\n    - Weighted Jaccard distance: Chierichetti, F., et al. (2010) `Finding the\n      Jaccard Median\n      <http://theory.stanford.edu/~sergei/papers/soda10-jaccard.pdf>`_.\n      Proceedings of the Twenty-First Annual ACM-SIAM Symposium on Discrete\n      Algorithms. Society for Industrial and Applied Mathematics.\n\n    - `Wikipedia - Cosine distance\n      <http://en.wikipedia.org/wiki/Cosine_similarity>`_\n\n    - `Wikipedia - Levenshtein distance\n      <http://en.wikipedia.org/wiki/Levenshtein_distance>`_\n\n    - Locality Sensitive Hashing : Chapter 3 of the book `Mining Massive\n      Datasets <http://infolab.stanford.edu/~ullman/mmds/ch3.pdf>`_.\n\n    Examples\n    --------\n    Construct a nearest neighbors model with automatically determined method\n    and distance:\n\n    >>> sf = turicreate.SFrame({'X1': [0.98, 0.62, 0.11],\n    ...                       'X2': [0.69, 0.58, 0.36],\n    ...                       'str_feature': ['cat', 'dog', 'fossa']})\n    >>> model = turicreate.nearest_neighbors.create(sf, features=['X1', 'X2'])\n\n    For datasets with a large number of rows and up to about 100 variables, the\n    ball tree method often leads to much faster queries.\n\n    >>> model = turicreate.nearest_neighbors.create(sf, features=['X1', 'X2'],\n    ...                                           method='ball_tree')\n\n    Often the final determination of a neighbor is based on several distance\n    computations over different sets of features. Each part of this composite\n    distance may have a different relative weight.\n\n    >>> my_dist = [[['X1', 'X2'], 'euclidean', 2.],\n    ...            [['str_feature'], 'levenshtein', 3.]]\n    ...\n    >>> model = turicreate.nearest_neighbors.create(sf, distance=my_dist)\n    "
    _tkutl._raise_error_if_not_sframe(dataset, 'dataset')
    _tkutl._raise_error_if_sframe_empty(dataset, 'dataset')
    if features is not None and (not isinstance(features, list)):
        raise TypeError("If specified, input 'features' must be a list of " + 'strings.')
    allowed_kwargs = ['leaf_size', 'num_tables', 'num_projections_per_table']
    _method_options = {}
    for (k, v) in kwargs.items():
        if k in allowed_kwargs:
            _method_options[k] = v
        else:
            raise _ToolkitError("'{}' is not a valid keyword argument".format(k) + ' for the nearest neighbors model. Please ' + 'check for capitalization and other typos.')
    if method == 'ball_tree' and (distance == 'cosine' or distance == _turicreate.distances.cosine or distance == _turicreate.distances.dot_product or (distance == 'transformed_dot_product') or (distance == _turicreate.distances.transformed_dot_product)):
        raise TypeError("The ball tree method does not work with 'cosine' " + "or 'transformed_dot_product' distance." + "Please use the 'brute_force' method for these distances.")
    if method == 'lsh' and 'num_projections_per_table' not in _method_options:
        if distance == 'jaccard' or distance == _turicreate.distances.jaccard:
            _method_options['num_projections_per_table'] = 4
        elif distance == 'cosine' or distance == _turicreate.distances.cosine:
            _method_options['num_projections_per_table'] = 16
        else:
            _method_options['num_projections_per_table'] = 8
    if label is None:
        _label = _robust_column_name('__id', dataset.column_names())
        _dataset = dataset.add_row_number(_label)
    else:
        _label = label
        _dataset = _copy.copy(dataset)
    col_type_map = {c: _dataset[c].dtype for c in _dataset.column_names()}
    _validate_row_label(_label, col_type_map)
    ref_labels = _dataset[_label]
    if features is None:
        _features = _dataset.column_names()
    else:
        _features = _copy.deepcopy(features)
    free_features = set(_features).difference([_label])
    if len(free_features) < 1:
        raise _ToolkitError('The only available feature is the same as the ' + 'row label column. Please specify features ' + 'that are not also row labels.')
    if isinstance(distance, list):
        distance = _copy.deepcopy(distance)
    elif hasattr(distance, '__call__') or (isinstance(distance, str) and (not distance == 'auto')):
        distance = [[_features, distance, 1]]
    elif distance is None or distance == 'auto':
        sample = _dataset.head()
        distance = _construct_auto_distance(_features, _dataset.column_names(), _dataset.column_types(), sample)
    else:
        raise TypeError("Input 'distance' not understood. The 'distance'  argument must be a string, function handle, or " + 'composite distance.')
    distance = _scrub_composite_distance_features(distance, [_label])
    distance = _convert_distance_names_to_functions(distance)
    _validate_composite_distance(distance)
    list_features_to_check = []
    sparse_distances = ['jaccard', 'weighted_jaccard', 'cosine', 'transformed_dot_product']
    sparse_distances = [_turicreate.distances.__dict__[k] for k in sparse_distances]
    for d in distance:
        (feature_names, dist, _) = d
        list_features = [f for f in feature_names if _dataset[f].dtype == list]
        for f in list_features:
            if dist in sparse_distances:
                list_features_to_check.append(f)
            else:
                raise TypeError('The chosen distance cannot currently be used ' + 'on list-typed columns.')
    for f in list_features_to_check:
        only_str_lists = _validate_lists(_dataset[f], [str])
        if not only_str_lists:
            raise TypeError('Distances for sparse data, such as jaccard ' + 'and weighted_jaccard, can only be used on ' + 'lists containing only strings. Please modify ' + 'any list features accordingly before creating ' + 'the nearest neighbors model.')
    for d in distance:
        (feature_names, dist, _) = d
        if len(feature_names) > 1 and dist == _turicreate.distances.levenshtein:
            raise ValueError('Levenshtein distance cannot be used with multiple ' + 'columns. Please concatenate strings into a single ' + 'column before creating the nearest neighbors model.')
    clean_features = _get_composite_distance_features(distance)
    sf_clean = _tkutl._toolkits_select_columns(_dataset, clean_features)
    if len(distance) > 1:
        _method = 'brute_force'
        if method != 'brute_force' and verbose is True:
            print('Defaulting to brute force instead of ball tree because ' + 'there are multiple distance components.')
    elif method == 'auto':
        num_variables = sum([len(x) if hasattr(x, '__iter__') else 1 for x in _six.itervalues(sf_clean[0])])
        numeric_type_flag = all([x in [int, float, list, array.array] for x in sf_clean.column_types()])
        if distance[0][1] in ['euclidean', 'manhattan', _turicreate.distances.euclidean, _turicreate.distances.manhattan] and numeric_type_flag is True and (num_variables <= 200):
            _method = 'ball_tree'
        else:
            _method = 'brute_force'
    else:
        _method = method
    if _method == 'ball_tree':
        model_name = 'nearest_neighbors_ball_tree'
    elif _method == 'brute_force':
        model_name = 'nearest_neighbors_brute_force'
    elif _method == 'lsh':
        model_name = 'nearest_neighbors_lsh'
    else:
        raise ValueError("Method must be 'auto', 'ball_tree', 'brute_force', " + "or 'lsh'.")
    opts = {}
    opts.update(_method_options)
    opts.update({'model_name': model_name, 'ref_labels': ref_labels, 'label': label, 'sf_features': sf_clean, 'composite_params': distance})
    with QuietProgress(verbose):
        result = _turicreate.extensions._nearest_neighbors.train(opts)
    model_proxy = result['model']
    model = NearestNeighborsModel(model_proxy)
    return model

class NearestNeighborsModel(_Model):
    """
    The NearestNeighborsModel represents rows of an SFrame in a structure that
    is used to quickly and efficiently find the nearest neighbors of a query
    point.

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.nearest_neighbors.create` to create an instance of this
    model. A detailed list of parameter options and code samples are available
    in the documentation for the create function.
    """

    def __init__(self, model_proxy):
        if False:
            while True:
                i = 10
        '___init__(self)'
        self.__proxy__ = model_proxy
        self.__name__ = 'nearest_neighbors'

    @classmethod
    def _native_name(cls):
        if False:
            while True:
                i = 10
        return ['nearest_neighbors_ball_tree', 'nearest_neighbors_brute_force', 'nearest_neighbors_lsh']

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the NearestNeighborsModel.\n        '
        return self.__repr__()

    def _get_summary_struct(self):
        if False:
            return 10
        "\n        Returns a structured description of the model, including (where\n        relevant) the schema of the training data, description of the training\n        data, training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        model_fields = [('Method', 'method'), ('Number of distance components', 'num_distance_components'), ('Number of examples', 'num_examples'), ('Number of feature columns', 'num_features'), ('Number of unpacked features', 'num_unpacked_features'), ('Distance', 'distance_for_summary_struct'), ('Total training time (seconds)', 'training_time')]
        ball_tree_fields = [('Tree depth', 'tree_depth'), ('Leaf size', 'leaf_size')]
        lsh_fields = [('Number of hash tables', 'num_tables'), ('Number of projections per table', 'num_projections_per_table')]
        sections = [model_fields]
        section_titles = ['Attributes']
        if self.method == 'ball_tree':
            sections.append(ball_tree_fields)
            section_titles.append('Ball Tree Attributes')
        if self.method == 'lsh':
            sections.append(lsh_fields)
            section_titles.append('LSH Attributes')
        return (sections, section_titles)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Print a string description of the model when the model name is entered\n        in the terminal.\n        '
        (sections, section_titles) = self._get_summary_struct()
        return _tkutl._toolkit_repr_print(self, sections, section_titles, width=30)

    def _list_fields(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        List the fields stored in the model, including data, model, and\n        training options. Each field can be queried with the ``get`` method.\n\n        Returns\n        -------\n        out : list\n            List of fields queryable with the ``get`` method.\n        '
        opts = {'model': self.__proxy__, 'model_name': self.__name__}
        response = _turicreate.extensions._nearest_neighbors.list_fields(opts)
        return sorted(response.keys())

    def _get(self, field):
        if False:
            return 10
        '\n        Return the value of a given field. The list of all queryable fields is\n        detailed below, and can be obtained with the\n        :func:`~turicreate.nearest_neighbors.NearestNeighborsModel._list_fields`\n        method.\n\n        +-----------------------+----------------------------------------------+\n        |      Field            | Description                                  |\n        +=======================+==============================================+\n        | distance              | Measure of dissimilarity between two points  |\n        +-----------------------+----------------------------------------------+\n        | features              | Feature column names                         |\n        +-----------------------+----------------------------------------------+\n        | unpacked_features     | Names of the individual features used        |\n        +-----------------------+----------------------------------------------+\n        | label                 | Label column names                           |\n        +-----------------------+----------------------------------------------+\n        | leaf_size             | Max size of leaf nodes (ball tree only)      |\n        +-----------------------+----------------------------------------------+\n        | method                | Method of organizing reference data          |\n        +-----------------------+----------------------------------------------+\n        | num_examples          | Number of reference data observations        |\n        +-----------------------+----------------------------------------------+\n        | num_features          | Number of features for distance computation  |\n        +-----------------------+----------------------------------------------+\n        | num_unpacked_features | Number of unpacked features                  |\n        +-----------------------+----------------------------------------------+\n        | num_variables         | Number of variables for distance computation |\n        +-----------------------+----------------------------------------------+\n        | training_time         | Time to create the reference structure       |\n        +-----------------------+----------------------------------------------+\n        | tree_depth            | Number of levels in the tree (ball tree only)|\n        +-----------------------+----------------------------------------------+\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out\n            Value of the requested field.\n        '
        opts = {'model': self.__proxy__, 'model_name': self.__name__, 'field': field}
        response = _turicreate.extensions._nearest_neighbors.get_value(opts)
        return response['value']

    def _training_stats(self):
        if False:
            print('Hello World!')
        "\n        Return a dictionary of statistics collected during creation of the\n        model. These statistics are also available with the ``get`` method and\n        are described in more detail in that method's documentation.\n\n        Returns\n        -------\n        out : dict\n            Dictionary of statistics compiled during creation of the\n            NearestNeighborsModel.\n\n        See Also\n        --------\n        summary\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'label': range(3),\n        ...                       'feature1': [0.98, 0.62, 0.11],\n        ...                       'feature2': [0.69, 0.58, 0.36]})\n        >>> model = turicreate.nearest_neighbors.create(sf, 'label')\n        >>> model.training_stats()\n        {'features': 'feature1, feature2',\n         'label': 'label',\n         'leaf_size': 1000,\n         'num_examples': 3,\n         'num_features': 2,\n         'num_variables': 2,\n         'training_time': 0.023223,\n         'tree_depth': 1}\n        "
        opts = {'model': self.__proxy__, 'model_name': self.__name__}
        return _turicreate.extensions._nearest_neighbors.training_stats(opts)

    def query(self, dataset, label=None, k=5, radius=None, verbose=True):
        if False:
            while True:
                i = 10
        "\n        For each row of the input 'dataset', retrieve the nearest neighbors\n        from the model's stored data. In general, the query dataset does not\n        need to be the same as the reference data stored in the model, but if\n        it is, the 'include_self_edges' parameter can be set to False to\n        exclude results that match query points to themselves.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Query data. Must contain columns with the same names and types as\n            the features used to train the model. Additional columns are\n            allowed, but ignored. Please see the nearest neighbors\n            :func:`~turicreate.nearest_neighbors.create` documentation for more\n            detail on allowable data types.\n\n        label : str, optional\n            Name of the query SFrame column with row labels. If 'label' is not\n            specified, row numbers are used to identify query dataset rows in\n            the output SFrame.\n\n        k : int, optional\n            Number of nearest neighbors to return from the reference set for\n            each query observation. The default is 5 neighbors, but setting it\n            to ``None`` will return all neighbors within ``radius`` of the\n            query point.\n\n        radius : float, optional\n            Only neighbors whose distance to a query point is smaller than this\n            value are returned. The default is ``None``, in which case the\n            ``k`` nearest neighbors are returned for each query point,\n            regardless of distance.\n\n        verbose: bool, optional\n            If True, print progress updates and model details.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with the k-nearest neighbors of each query observation.\n            The result contains four columns: the first is the label of the\n            query observation, the second is the label of the nearby reference\n            observation, the third is the distance between the query and\n            reference observations, and the fourth is the rank of the reference\n            observation among the query's k-nearest neighbors.\n\n        See Also\n        --------\n        similarity_graph\n\n        Notes\n        -----\n        - The `dataset` input to this method *can* have missing values (in\n          contrast to the reference dataset used to create the nearest\n          neighbors model). Missing numeric values are imputed to be the mean\n          of the corresponding feature in the reference dataset, and missing\n          strings are imputed to be empty strings.\n\n        - If both ``k`` and ``radius`` are set to ``None``, each query point\n          returns all of the reference set. If the reference dataset has\n          :math:`n` rows and the query dataset has :math:`m` rows, the output\n          is an SFrame with :math:`nm` rows.\n\n        - For models created with the 'lsh' method, the query results may have\n          fewer query labels than input query points. Because LSH is an\n          approximate method, a query point may have fewer than 'k' neighbors.\n          If LSH returns no neighbors at all for a query, the query point is\n          omitted from the results.\n\n        Examples\n        --------\n        First construct a toy SFrame and create a nearest neighbors model:\n\n        >>> sf = turicreate.SFrame({'label': range(3),\n        ...                       'feature1': [0.98, 0.62, 0.11],\n        ...                       'feature2': [0.69, 0.58, 0.36]})\n        >>> model = turicreate.nearest_neighbors.create(sf, 'label')\n\n        A new SFrame contains query observations with same schema as the\n        reference SFrame. This SFrame is passed to the ``query`` method.\n\n        >>> queries = turicreate.SFrame({'label': range(3),\n        ...                            'feature1': [0.05, 0.61, 0.99],\n        ...                            'feature2': [0.06, 0.97, 0.86]})\n        >>> model.query(queries, 'label', k=2)\n        +-------------+-----------------+----------------+------+\n        | query_label | reference_label |    distance    | rank |\n        +-------------+-----------------+----------------+------+\n        |      0      |        2        | 0.305941170816 |  1   |\n        |      0      |        1        | 0.771556867638 |  2   |\n        |      1      |        1        | 0.390128184063 |  1   |\n        |      1      |        0        | 0.464004310325 |  2   |\n        |      2      |        0        | 0.170293863659 |  1   |\n        |      2      |        1        | 0.464004310325 |  2   |\n        +-------------+-----------------+----------------+------+\n        "
        _tkutl._raise_error_if_not_sframe(dataset, 'dataset')
        _tkutl._raise_error_if_sframe_empty(dataset, 'dataset')
        ref_features = self.features
        sf_features = _tkutl._toolkits_select_columns(dataset, ref_features)
        if label is None:
            query_labels = _turicreate.SArray.from_sequence(len(dataset))
        else:
            if not label in dataset.column_names():
                raise ValueError("Input 'label' must be a string matching the name of a " + "column in the reference SFrame 'dataset'.")
            if not dataset[label].dtype == str and (not dataset[label].dtype == int):
                raise TypeError('The label column must contain integers or strings.')
            if label in ref_features:
                raise ValueError('The label column cannot be one of the features.')
            query_labels = dataset[label]
        if k is not None:
            if not isinstance(k, int):
                raise ValueError("Input 'k' must be an integer.")
            if k <= 0:
                raise ValueError("Input 'k' must be larger than 0.")
        if radius is not None:
            if not isinstance(radius, (int, float)):
                raise ValueError("Input 'radius' must be an integer or float.")
            if radius < 0:
                raise ValueError("Input 'radius' must be non-negative.")
        if k is None:
            k = -1
        if radius is None:
            radius = -1.0
        opts = {'model': self.__proxy__, 'model_name': self.__name__, 'features': sf_features, 'query_labels': query_labels, 'k': k, 'radius': radius}
        with QuietProgress(verbose):
            result = _turicreate.extensions._nearest_neighbors.query(opts)
        return result['neighbors']

    def similarity_graph(self, k=5, radius=None, include_self_edges=False, output_type='SGraph', verbose=True):
        if False:
            print('Hello World!')
        "\n        Construct the similarity graph on the reference dataset, which is\n        already stored in the model. This is conceptually very similar to\n        running `query` with the reference set, but this method is optimized\n        for the purpose, syntactically simpler, and automatically removes\n        self-edges.\n\n        Parameters\n        ----------\n        k : int, optional\n            Maximum number of neighbors to return for each point in the\n            dataset. Setting this to ``None`` deactivates the constraint, so\n            that all neighbors are returned within ``radius`` of a given point.\n\n        radius : float, optional\n            For a given point, only neighbors within this distance are\n            returned. The default is ``None``, in which case the ``k`` nearest\n            neighbors are returned for each query point, regardless of\n            distance.\n\n        include_self_edges : bool, optional\n            For most distance functions, each point in the model's reference\n            dataset is its own nearest neighbor. If this parameter is set to\n            False, this result is ignored, and the nearest neighbors are\n            returned *excluding* the point itself.\n\n        output_type : {'SGraph', 'SFrame'}, optional\n            By default, the results are returned in the form of an SGraph,\n            where each point in the reference dataset is a vertex and an edge A\n            -> B indicates that vertex B is a nearest neighbor of vertex A. If\n            'output_type' is set to 'SFrame', the output is in the same form as\n            the results of the 'query' method: an SFrame with columns\n            indicating the query label (in this case the query data is the same\n            as the reference data), reference label, distance between the two\n            points, and the rank of the neighbor.\n\n        verbose : bool, optional\n            If True, print progress updates and model details.\n\n        Returns\n        -------\n        out : SFrame or SGraph\n            The type of the output object depends on the 'output_type'\n            parameter. See the parameter description for more detail.\n\n        Notes\n        -----\n        - If both ``k`` and ``radius`` are set to ``None``, each data point is\n          matched to the entire dataset. If the reference dataset has\n          :math:`n` rows, the output is an SFrame with :math:`n^2` rows (or an\n          SGraph with :math:`n^2` edges).\n\n        - For models created with the 'lsh' method, the output similarity graph\n          may have fewer vertices than there are data points in the original\n          reference set. Because LSH is an approximate method, a query point\n          may have fewer than 'k' neighbors. If LSH returns no neighbors at all\n          for a query and self-edges are excluded, the query point is omitted\n          from the results.\n\n        Examples\n        --------\n        First construct an SFrame and create a nearest neighbors model:\n\n        >>> sf = turicreate.SFrame({'x1': [0.98, 0.62, 0.11],\n        ...                       'x2': [0.69, 0.58, 0.36]})\n        ...\n        >>> model = turicreate.nearest_neighbors.create(sf, distance='euclidean')\n\n        Unlike the ``query`` method, there is no need for a second dataset with\n        ``similarity_graph``.\n\n        >>> g = model.similarity_graph(k=1)  # an SGraph\n        >>> g.edges\n        +----------+----------+----------------+------+\n        | __src_id | __dst_id |    distance    | rank |\n        +----------+----------+----------------+------+\n        |    0     |    1     | 0.376430604494 |  1   |\n        |    2     |    1     | 0.55542776308  |  1   |\n        |    1     |    0     | 0.376430604494 |  1   |\n        +----------+----------+----------------+------+\n        "
        if k is not None:
            if not isinstance(k, int):
                raise ValueError("Input 'k' must be an integer.")
            if k <= 0:
                raise ValueError("Input 'k' must be larger than 0.")
        if radius is not None:
            if not isinstance(radius, (int, float)):
                raise ValueError("Input 'radius' must be an integer or float.")
            if radius < 0:
                raise ValueError("Input 'radius' must be non-negative.")
        if k is None:
            k = -1
        if radius is None:
            radius = -1.0
        opts = {'model': self.__proxy__, 'model_name': self.__name__, 'k': k, 'radius': radius, 'include_self_edges': include_self_edges}
        with QuietProgress(verbose):
            result = _turicreate.extensions._nearest_neighbors.similarity_graph(opts)
        knn = result['neighbors']
        if output_type == 'SFrame':
            return knn
        else:
            sg = _SGraph(edges=knn, src_field='query_label', dst_field='reference_label')
            return sg