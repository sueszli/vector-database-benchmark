"""
Models that rank items based on their popularity.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _turicreate
from turicreate.toolkits.recommender.util import _Recommender
from turicreate.data_structures.sframe import SFrame as _SFrame

def create(observation_data, user_id='user_id', item_id='item_id', target=None, user_data=None, item_data=None, random_seed=0, verbose=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a model that makes recommendations using item popularity. When no\n    target column is provided, the popularity is determined by the number of\n    observations involving each item. When a target is provided, popularity\n    is computed using the item\'s mean target value. When the target column\n    contains ratings, for example, the model computes the mean rating for\n    each item and uses this to rank items for recommendations.\n\n    Parameters\n    ----------\n    observation_data : SFrame\n        The dataset to use for training the model. It must contain a column of\n        user ids and a column of item ids. Each row represents an observed\n        interaction between the user and the item.  The (user, item) pairs\n        are stored with the model so that they can later be excluded from\n        recommendations if desired. It can optionally contain a target ratings\n        column. All other columns are interpreted by the underlying model as\n        side features for the observations.\n\n        The user id and item id columns must be of type \'int\' or \'str\'. The\n        target column must be of type \'int\' or \'float\'.\n\n    user_id : string, optional\n        The name of the column in `observation_data` that corresponds to the\n        user id.\n\n    item_id : string, optional\n        The name of the column in `observation_data` that corresponds to the\n        item id.\n\n    target : string, optional\n        The `observation_data` can optionally contain a column of scores\n        representing ratings given by the users. If present, the name of this\n        column may be specified variables `target`.\n\n    user_data : SFrame, optional\n        Side information for the users.  This SFrame must have a column with\n        the same name as what is specified by the `user_id` input parameter.\n        `user_data` can provide any amount of additional user-specific\n        information.\n\n    item_data : SFrame, optional\n        Side information for the items.  This SFrame must have a column with\n        the same name as what is specified by the `item_id` input parameter.\n        `item_data` can provide any amount of additional item-specific\n        information.\n\n    verbose : bool, optional\n        Enables verbose output.\n\n    Examples\n    --------\n    >>> sf = turicreate.SFrame({\'user_id\': ["0", "0", "0", "1", "1", "2", "2", "2"],\n    ...                       \'item_id\': ["a", "b", "c", "a", "b", "b", "c", "d"],\n    ...                       \'rating\': [1, 3, 2, 5, 4, 1, 4, 3]})\n    >>> m = turicreate.popularity_recommender.create(sf, target=\'rating\')\n\n    See Also\n    --------\n    PopularityRecommender\n    '
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(observation_data, _SFrame):
        raise TypeError('observation_data input must be a SFrame')
    opts = {}
    model_proxy = _turicreate.extensions.popularity()
    model_proxy.init_options(opts)
    if user_data is None:
        user_data = _turicreate.SFrame()
    if item_data is None:
        item_data = _turicreate.SFrame()
    opts = {'user_id': user_id, 'item_id': item_id, 'target': target, 'random_seed': 1}
    extra_data = {'nearest_items': _turicreate.SFrame()}
    with QuietProgress(verbose):
        model_proxy.train(observation_data, user_data, item_data, opts, extra_data)
    return PopularityRecommender(model_proxy)

class PopularityRecommender(_Recommender):
    """
    The Popularity Model ranks an item according to its overall popularity.

    When making recommendations, the items are scored by the number of times it
    is seen in the training set. The item scores are the same for all users.
    Hence the recommendations are not tailored for individuals.

    The Popularity Recommender is simple and fast and provides a reasonable baseline.
    It can work well when observation data is sparse. It can be used as a
    "background" model for new users.

    **Creating a PopularityRecommender**

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.recommender.popularity_recommender.create`
    to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    See Also
    --------
    create
    """

    def __init__(self, model_proxy):
        if False:
            print('Hello World!')
        '__init__(self)'
        self.__proxy__ = model_proxy

    @classmethod
    def _native_name(cls):
        if False:
            i = 10
            return i + 15
        return 'popularity'