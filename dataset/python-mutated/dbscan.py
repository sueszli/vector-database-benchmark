"""
Class definition and create method for DBSCAN clustering.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import time as _time
import logging as _logging
import turicreate as _tc
import turicreate.aggregate as _agg
from turicreate.toolkits._model import CustomModel as _CustomModel
import turicreate.toolkits._internal_utils as _tkutl
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits._private_utils import _summarize_accessible_fields
from turicreate.toolkits._model import PythonProxy as _PythonProxy

def create(dataset, features=None, distance=None, radius=1.0, min_core_neighbors=10, verbose=True):
    if False:
        while True:
            i = 10
    '\n    Create a DBSCAN clustering model. The DBSCAN method partitions the input\n    dataset into three types of points, based on the estimated probability\n    density at each point.\n\n    - **Core** points have a large number of points within a given neighborhood.\n      Specifically, `min_core_neighbors` must be within distance `radius` of a\n      point for it to be considered a core point.\n\n    - **Boundary** points are within distance `radius` of a core point, but\n      don\'t have sufficient neighbors of their own to be considered core.\n\n    - **Noise** points comprise the remainder of the data. These points have too\n      few neighbors to be considered core points, and are further than distance\n      `radius` from all core points.\n\n    Clusters are formed by connecting core points that are neighbors of each\n    other, then assigning boundary points to their nearest core neighbor\'s\n    cluster.\n\n    Parameters\n    ----------\n    dataset : SFrame\n        Training data, with each row corresponding to an observation. Must\n        include all features specified in the `features` parameter, but may have\n        additional columns as well.\n\n    features : list[str], optional\n        Name of the columns with features to use in comparing records. \'None\'\n        (the default) indicates that all columns of the input `dataset` should\n        be used to train the model. All features must be numeric, i.e. integer\n        or float types.\n\n    distance : str or list[list], optional\n        Function to measure the distance between any two input data rows. This\n        may be one of two types:\n\n        - *String*: the name of a standard distance function. One of\n          \'euclidean\', \'squared_euclidean\', \'manhattan\', \'levenshtein\',\n          \'jaccard\', \'weighted_jaccard\', \'cosine\', or \'transformed_dot_product\'.\n\n        - *Composite distance*: the weighted sum of several standard distance\n          functions applied to various features. This is specified as a list of\n          distance components, each of which is itself a list containing three\n          items:\n\n          1. list or tuple of feature names (str)\n\n          2. standard distance name (str)\n\n          3. scaling factor (int or float)\n\n        For more information about Turi Create distance functions, please\n        see the :py:mod:`~turicreate.toolkits.distances` module.\n\n        For sparse vectors, missing keys are assumed to have value 0.0.\n\n        If \'distance\' is left unspecified, a composite distance is constructed\n        automatically based on feature types.\n\n    radius : int or float, optional\n        Size of each point\'s neighborhood, with respect to the specified\n        distance function.\n\n    min_core_neighbors : int, optional\n        Number of neighbors that must be within distance `radius` of a point in\n        order for that point to be considered a "core point" of a cluster.\n\n    verbose : bool, optional\n        If True, print progress updates and model details during model creation.\n\n    Returns\n    -------\n    out : DBSCANModel\n        A model containing a cluster label for each row in the input `dataset`.\n        Also contains the indices of the core points, cluster boundary points,\n        and noise points.\n\n    See Also\n    --------\n    DBSCANModel, turicreate.toolkits.distances\n\n    Notes\n    -----\n    - Our implementation of DBSCAN first computes the similarity graph on the\n      input dataset, which can be a computationally intensive process. In the\n      current implementation, some distances are substantially faster than\n      others; in particular "euclidean", "squared_euclidean", "cosine", and\n      "transformed_dot_product" are quite fast, while composite distances can be\n      slow.\n\n    - Any distance function in the Turi Create library may be used with DBSCAN but\n      the results may be poor for distances that violate the standard metric\n      properties, i.e. symmetry, non-negativity, triangle inequality, and\n      identity of indiscernibles. In particular, the DBSCAN algorithm is based\n      on the concept of connecting high-density points that are *close* to each\n      other into a single cluster, but the notion of *close* may be very\n      counterintuitive if the chosen distance function is not a valid metric.\n      The distances "euclidean", "manhattan", "jaccard", and "levenshtein" will\n      likely yield the best results.\n\n    References\n    ----------\n    - Ester, M., et al. (1996) `A Density-Based Algorithm for Discovering\n      Clusters in Large Spatial Databases with Noise\n      <https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf>`_. In Proceedings of the\n      Second International Conference on Knowledge Discovery and Data Mining.\n      pp. 226-231.\n\n    - `Wikipedia - DBSCAN <https://en.wikipedia.org/wiki/DBSCAN>`_\n\n    - `Visualizing DBSCAN Clustering\n      <http://www.naftaliharris.com/blog/visualizing-dbscan-clustering/>`_\n\n    Examples\n    --------\n    >>> sf = turicreate.SFrame({\n    ...     \'x1\': [0.6777, -9.391, 7.0385, 2.2657, 7.7864, -10.16, -8.162,\n    ...            8.8817, -9.525, -9.153, 2.0860, 7.6619, 6.5511, 2.7020],\n    ...     \'x2\': [5.6110, 8.5139, 5.3913, 5.4743, 8.3606, 7.8843, 2.7305,\n    ...            5.1679, 6.7231, 3.7051, 1.7682, 7.4608, 3.1270, 6.5624]})\n    ...\n    >>> model = turicreate.dbscan.create(sf, radius=4.25, min_core_neighbors=3)\n    >>> model.cluster_id.print_rows(15)\n    +--------+------------+----------+\n    | row_id | cluster_id |   type   |\n    +--------+------------+----------+\n    |   8    |     0      |   core   |\n    |   7    |     2      |   core   |\n    |   0    |     1      |   core   |\n    |   2    |     2      |   core   |\n    |   3    |     1      |   core   |\n    |   11   |     2      |   core   |\n    |   4    |     2      |   core   |\n    |   1    |     0      | boundary |\n    |   6    |     0      | boundary |\n    |   5    |     0      | boundary |\n    |   9    |     0      | boundary |\n    |   12   |     2      | boundary |\n    |   10   |     1      | boundary |\n    |   13   |     1      | boundary |\n    +--------+------------+----------+\n    [14 rows x 3 columns]\n    '
    logger = _logging.getLogger(__name__)
    start_time = _time.time()
    _tkutl._raise_error_if_not_sframe(dataset, 'dataset')
    _tkutl._raise_error_if_sframe_empty(dataset, 'dataset')
    if not isinstance(min_core_neighbors, int) or min_core_neighbors < 0:
        raise ValueError("Input 'min_core_neighbors' must be a non-negative " + 'integer.')
    if not isinstance(radius, (int, float)) or radius < 0:
        raise ValueError("Input 'radius' must be a non-negative integer " + 'or float.')
    knn_model = _tc.nearest_neighbors.create(dataset, features=features, distance=distance, method='brute_force', verbose=verbose)
    knn = knn_model.similarity_graph(k=None, radius=radius, include_self_edges=False, output_type='SFrame', verbose=verbose)
    neighbor_counts = knn.groupby('query_label', _agg.COUNT)
    if verbose:
        logger.info('Identifying noise points and core points.')
    boundary_mask = neighbor_counts['Count'] < min_core_neighbors
    core_mask = 1 - boundary_mask
    boundary_idx = neighbor_counts[boundary_mask]['query_label']
    core_idx = neighbor_counts[core_mask]['query_label']
    if verbose:
        logger.info('Constructing the core point similarity graph.')
    core_vertices = knn.filter_by(core_idx, 'query_label')
    core_edges = core_vertices.filter_by(core_idx, 'reference_label')
    core_graph = _tc.SGraph()
    core_graph = core_graph.add_vertices(core_vertices[['query_label']], vid_field='query_label')
    core_graph = core_graph.add_edges(core_edges, src_field='query_label', dst_field='reference_label')
    cc = _tc.connected_components.create(core_graph, verbose=verbose)
    cc_labels = cc.component_size.add_row_number('__label')
    core_assignments = cc.component_id.join(cc_labels, on='component_id', how='left')[['__id', '__label']]
    core_assignments['type'] = 'core'
    if verbose:
        logger.info('Processing boundary points.')
    boundary_edges = knn.filter_by(boundary_idx, 'query_label')
    boundary_core_edges = boundary_edges.filter_by(core_idx, 'reference_label')
    boundary_assignments = boundary_core_edges.groupby('query_label', {'reference_label': _agg.ARGMIN('rank', 'reference_label')})
    boundary_assignments = boundary_assignments.join(core_assignments, on={'reference_label': '__id'})
    boundary_assignments = boundary_assignments.rename({'query_label': '__id'}, inplace=True)
    boundary_assignments = boundary_assignments.remove_column('reference_label', inplace=True)
    boundary_assignments['type'] = 'boundary'
    small_cluster_idx = set(boundary_idx).difference(boundary_assignments['__id'])
    noise_idx = set(range(dataset.num_rows())).difference(neighbor_counts['query_label'])
    noise_idx = noise_idx.union(small_cluster_idx)
    noise_assignments = _tc.SFrame({'row_id': _tc.SArray(list(noise_idx), int)})
    noise_assignments['cluster_id'] = None
    noise_assignments['cluster_id'] = noise_assignments['cluster_id'].astype(int)
    noise_assignments['type'] = 'noise'
    master_assignments = _tc.SFrame()
    num_clusters = 0
    if core_assignments.num_rows() > 0:
        core_assignments = core_assignments.rename({'__id': 'row_id', '__label': 'cluster_id'}, inplace=True)
        master_assignments = master_assignments.append(core_assignments)
        num_clusters = len(core_assignments['cluster_id'].unique())
    if boundary_assignments.num_rows() > 0:
        boundary_assignments = boundary_assignments.rename({'__id': 'row_id', '__label': 'cluster_id'}, inplace=True)
        master_assignments = master_assignments.append(boundary_assignments)
    if noise_assignments.num_rows() > 0:
        master_assignments = master_assignments.append(noise_assignments)
    state = {'verbose': verbose, 'radius': radius, 'min_core_neighbors': min_core_neighbors, 'distance': knn_model.distance, 'num_distance_components': knn_model.num_distance_components, 'num_examples': dataset.num_rows(), 'features': knn_model.features, 'num_features': knn_model.num_features, 'unpacked_features': knn_model.unpacked_features, 'num_unpacked_features': knn_model.num_unpacked_features, 'cluster_id': master_assignments, 'num_clusters': num_clusters, 'training_time': _time.time() - start_time}
    return DBSCANModel(state)

class DBSCANModel(_CustomModel):
    """
    DBSCAN clustering model. The DBSCAN model contains the results of DBSCAN
    clustering, which finds clusters by identifying "core points" of high
    probability density and building clusters around them.

    This model should not be constructed directly. Instead, use
    :func:`turicreate.clustering.dbscan.create` to create an instance of this
    model.
    """
    _PYTHON_DBSCAN_MODEL_VERSION = 1

    def __init__(self, state):
        if False:
            i = 10
            return i + 15
        self.__proxy__ = _PythonProxy(state)

    @classmethod
    def _native_name(cls):
        if False:
            print('Hello World!')
        return 'dbscan'

    def _get_version(self):
        if False:
            for i in range(10):
                print('nop')
        return self._PYTHON_DBSCAN_MODEL_VERSION

    def _get_native_state(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__proxy__.state

    @classmethod
    def _load_version(self, state, version):
        if False:
            while True:
                i = 10
        '\n        A function to load a previously created DBSCANModel instance.\n\n        Parameters\n        ----------\n        unpickler : GLUnpickler\n            A GLUnpickler file handler.\n\n        version : int\n            Version number maintained by the class writer.\n        '
        state = _PythonProxy(state)
        return DBSCANModel(state)

    def __str__(self):
        if False:
            return 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : str\n            A description of the DBSCANModel.\n        '
        return self.__repr__()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Print a string description of the model when the model name is entered\n        in the terminal.\n        '
        width = 40
        (sections, section_titles) = self._get_summary_struct()
        accessible_fields = {'cluster_id': 'Cluster label for each row in the input dataset.'}
        out = _toolkit_repr_print(self, sections, section_titles, width=width)
        out2 = _summarize_accessible_fields(accessible_fields, width=width)
        return out + '\n' + out2

    def _get_summary_struct(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a structured description of the model. This includes (where\n        relevant) the schema of the training data, description of the training\n        data, training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        model_fields = [('Number of examples', 'num_examples'), ('Number of feature columns', 'num_features'), ('Max distance to a neighbor (radius)', 'radius'), ('Min number of neighbors for core points', 'min_core_neighbors'), ('Number of distance components', 'num_distance_components')]
        training_fields = [('Total training time (seconds)', 'training_time'), ('Number of clusters', 'num_clusters')]
        section_titles = ['Schema', 'Training summary']
        return ([model_fields, training_fields], section_titles)