"""
The Item Content Recommender recommends similar items, where similar is
determined by information about the items rather than the user
interaction patterns.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _turicreate
from turicreate import SFrame as _SFrame
from turicreate.toolkits.recommender.util import _Recommender

def create(item_data, item_id, observation_data=None, user_id=None, target=None, weights='auto', similarity_metrics='auto', item_data_transform='auto', max_item_neighborhood_size=64, verbose=True):
    if False:
        return 10
    'Create a content-based recommender model in which the similarity\n    between the items recommended is determined by the content of\n    those items rather than learned from user interaction data.\n\n    The similarity score between two items is calculated by first\n    computing the similarity between the item data for each column,\n    then taking a weighted average of the per-column similarities to\n    get the final similarity.  The recommendations are generated\n    according to the average similarity of a candidate item to all the\n    items in a user\'s set of rated items.\n\n    Parameters\n    ----------\n\n    item_data : SFrame\n        An SFrame giving the content of the items to use to learn the\n        structure of similar items.  The SFrame must have one column\n        that matches the name of the `item_id`; this gives a unique\n        identifier that can then be used to make recommendations.  The rest\n        of the columns are then used in the distance calculations\n        below.\n\n    item_id : string\n        The name of the column in item_data (and `observation_data`,\n        if given) that represents the item ID.\n\n    observation_data : None (optional)\n        An SFrame giving user and item interaction data.  This\n        information is stored in the model, and the recommender will\n        recommend the items with the most similar content to the\n        items that were present and/or highly rated for that user.\n\n    user_id : None (optional)\n        If observation_data is given, then this specifies the column\n        name corresponding to the user identifier.\n\n    target : None (optional)\n        If observation_data is given, then this specifies the column\n        name corresponding to the target or rating.\n\n    weights : dict or \'auto\' (optional)\n        If given, then weights must be a dictionary of column names\n        present in item_data to weights between the column names.  If\n        \'auto\' is given, the all columns are weighted equally.\n\n    max_item_neighborhood_size : int, 64\n        For each item, we hold this many similar items to use when\n        aggregating models for predictions.  Decreasing this value\n        decreases the memory required by the model and decreases the\n        time required to generate recommendations, but it may also\n        decrease recommendation accuracy.\n\n    verbose : True or False (optional)\n        If set to False, then less information is printed.\n\n    Examples\n    --------\n\n      >>> item_data = tc.SFrame({"my_item_id" : range(4),\n                                 "data_1" : [ [1, 0], [1, 0], [0, 1], [0.5, 0.5] ],\n                                 "data_2" : [ [0, 1], [1, 0], [0, 1], [0.5, 0.5] ] })\n\n      >>> m = tc.recommender.item_content_recommender.create(item_data, "my_item_id")\n      >>> m.recommend_from_interactions([0])\n\n      Columns:\n              my_item_id      int\n              score   float\n              rank    int\n\n      Rows: 3\n\n      Data:\n      +------------+----------------+------+\n      | my_item_id |     score      | rank |\n      +------------+----------------+------+\n      |     3      | 0.707106769085 |  1   |\n      |     1      |      0.5       |  2   |\n      |     2      |      0.5       |  3   |\n      +------------+----------------+------+\n      [3 rows x 3 columns]\n\n      >>> m.recommend_from_interactions([0, 1])\n\n      Columns:\n              my_item_id      int\n              score   float\n              rank    int\n\n      Rows: 2\n\n      Data:\n      +------------+----------------+------+\n      | my_item_id |     score      | rank |\n      +------------+----------------+------+\n      |     3      | 0.707106769085 |  1   |\n      |     2      |      0.25      |  2   |\n      +------------+----------------+------+\n      [2 rows x 3 columns]\n\n    '
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(item_data, _SFrame) or item_data.num_rows() == 0:
        raise TypeError('`item_data` argument must be a non-empty SFrame giving item data to use for similarities.')
    item_columns = set(item_data.column_names())
    if item_id not in item_columns:
        raise ValueError("Item column given as 'item_id = %s', but this is not found in `item_data` SFrame." % item_id)
    item_columns.remove(item_id)
    if weights != 'auto':
        if type(weights) is not dict:
            raise TypeError("`weights` parameter must be 'auto' or a dictionary of column names in `item_data` to weight values.")
        bad_columns = [col_name for col_name in item_columns if col_name not in item_columns]
        if bad_columns:
            raise ValueError('Columns %s given in weights, but these are not found in item_data.' % ', '.join(bad_columns))
        for col_name in item_columns:
            weights.setdefault(col_name, 0)
    if item_data_transform == 'auto':
        item_data_transform = _turicreate.toolkits._feature_engineering.AutoVectorizer(excluded_features=[item_id])
    if not isinstance(item_data_transform, _turicreate.toolkits._feature_engineering.TransformerBase):
        raise TypeError("item_data_transform must be 'auto' or a valid feature_engineering transformer instance.")
    item_data = item_data_transform.fit_transform(item_data)
    gaussian_kernel_metrics = set()
    for c in item_columns:
        if item_data[c].dtype is str:
            item_data[c] = item_data[c].apply(lambda s: {s: 1})
        elif item_data[c].dtype in [float, int]:
            item_data[c] = (item_data[c] - item_data[c].mean()) / max(item_data[c].std(), 1e-08)
            gaussian_kernel_metrics.add(c)
    if verbose:
        print('Applying transform:')
        print(item_data_transform)
    opts = {}
    model_proxy = _turicreate.extensions.item_content_recommender()
    model_proxy.init_options(opts)
    if user_id is None:
        user_id = '__implicit_user__'
    normalization_factor = 1
    if observation_data is None:
        empty_user = _turicreate.SArray([], dtype=str)
        empty_item = _turicreate.SArray([], dtype=item_data[item_id].dtype)
        observation_data = _turicreate.SFrame({user_id: empty_user, item_id: empty_item})
    normalization_factor = 1
    if item_data.num_columns() >= 3:
        if weights == 'auto':
            weights = {col_name: 1 for col_name in item_data.column_names() if col_name != item_id}
        normalization_factor = sum((abs(v) for v in weights.values()))
        if normalization_factor == 0:
            raise ValueError('Weights cannot all be set to 0.')
        distance = [([col_name], 'gaussian_kernel' if col_name in gaussian_kernel_metrics else 'cosine', weight) for (col_name, weight) in weights.items()]
    else:
        distance = 'cosine'
    nn = _turicreate.nearest_neighbors.create(item_data, label=item_id, distance=distance, verbose=verbose)
    graph = nn.query(item_data, label=item_id, k=max_item_neighborhood_size, verbose=verbose)
    graph = graph.rename({'query_label': item_id, 'reference_label': 'similar', 'distance': 'score'}, inplace=True)

    def process_weights(x):
        if False:
            while True:
                i = 10
        return max(-1, min(1, 1 - x / normalization_factor))
    graph['score'] = graph['score'].apply(process_weights)
    opts = {'user_id': user_id, 'item_id': item_id, 'target': target, 'similarity_type': 'cosine', 'max_item_neighborhood_size': max_item_neighborhood_size}
    user_data = _turicreate.SFrame()
    extra_data = {'nearest_items': graph}
    with QuietProgress(verbose):
        model_proxy.train(observation_data, user_data, item_data, opts, extra_data)
    return ItemContentRecommender(model_proxy)

class ItemContentRecommender(_Recommender):
    """A recommender based on the similarity between item content rather
    using user interaction patterns to compute similarity.

    **Creating an ItemContentRecommender**

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.recommender.item_content_recommender.create`
    to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    Notes
    -----
    **Model Definition**

    This model first computes the similarity between items using the
    content of each item. The similarity score between two items is
    calculated by first computing the similarity between the item data
    for each column, then taking a weighted average of the per-column
    similarities to get the final similarity.  The recommendations are
    generated according to the average similarity of a candidate item
    to all the items in a user's set of rated items.

    For more examples, see the associated `create` function.
    """

    def __init__(self, model_proxy):
        if False:
            i = 10
            return i + 15
        '__init__(self)'
        self.__proxy__ = model_proxy

    @classmethod
    def _native_name(cls):
        if False:
            i = 10
            return i + 15
        return 'item_content_recommender'