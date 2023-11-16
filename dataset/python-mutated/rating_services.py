"""System for assigning and displaying ratings of explorations."""
from __future__ import annotations
import datetime
from core import feconf
from core.domain import event_services
from core.domain import exp_fetchers
from core.domain import exp_services
from core.platform import models
from typing import Dict, Optional
MYPY = False
if MYPY:
    from mypy_imports import transaction_services
    from mypy_imports import user_models
(exp_models, user_models) = models.Registry.import_models([models.Names.EXPLORATION, models.Names.USER])
transaction_services = models.Registry.import_transaction_services()
ALLOWED_RATINGS = [1, 2, 3, 4, 5]

def assign_rating_to_exploration(user_id: str, exploration_id: str, new_rating: int) -> None:
    if False:
        print('Hello World!')
    'Records the rating awarded by the user to the exploration in both the\n    user-specific data and exploration summary.\n\n    This function validates the exploration id but not the user id.\n\n    Args:\n        user_id: str. The id of the user assigning the rating.\n        exploration_id: str. The id of the exploration that is\n            assigned a rating.\n        new_rating: int. Value of assigned rating, should be between\n            1 and 5 inclusive.\n\n    Raises:\n        ValueError. The assigned rating is not of type int.\n        ValueError. The assigned rating is lower than 1 or higher than 5.\n        ValueError. The exploration does not exist.\n    '
    if not isinstance(new_rating, int):
        raise ValueError('Expected the rating to be an integer, received %s' % new_rating)
    if new_rating not in ALLOWED_RATINGS:
        raise ValueError('Expected a rating 1-5, received %s.' % new_rating)
    exploration = exp_fetchers.get_exploration_by_id(exploration_id, strict=False)
    if exploration is None:
        raise ValueError('Invalid exploration id %s' % exploration_id)

    @transaction_services.run_in_transaction_wrapper
    def _update_user_rating_transactional() -> Optional[int]:
        if False:
            while True:
                i = 10
        'Updates the user rating of the exploration. Returns the old rating\n        before updation.\n        '
        exp_user_data_model = user_models.ExplorationUserDataModel.get(user_id, exploration_id)
        if exp_user_data_model:
            old_rating: Optional[int] = exp_user_data_model.rating
        else:
            old_rating = None
            exp_user_data_model = user_models.ExplorationUserDataModel.create(user_id, exploration_id)
        exp_user_data_model.rating = new_rating
        exp_user_data_model.rated_on = datetime.datetime.utcnow()
        exp_user_data_model.update_timestamps()
        exp_user_data_model.put()
        return old_rating
    old_rating = _update_user_rating_transactional()
    exploration_summary = exp_fetchers.get_exploration_summary_by_id(exploration_id)
    if not exploration_summary.ratings:
        exploration_summary.ratings = feconf.get_empty_ratings()
    exploration_summary.ratings[str(new_rating)] += 1
    if old_rating:
        exploration_summary.ratings[str(old_rating)] -= 1
    event_services.RateExplorationEventHandler.record(exploration_id, user_id, new_rating, old_rating)
    exploration_summary.scaled_average_rating = exp_services.get_scaled_average_rating(exploration_summary.ratings)
    exp_services.save_exploration_summary(exploration_summary)

def get_user_specific_rating_for_exploration(user_id: str, exploration_id: str) -> Optional[int]:
    if False:
        print('Hello World!')
    'Fetches a rating for the specified exploration from the specified user\n    if one exists.\n\n    Args:\n        user_id: str. The id of the user.\n        exploration_id: str. The id of the exploration.\n\n    Returns:\n        int or None. An integer between 1 and 5 inclusive, or None if the user\n        has not previously rated the exploration.\n    '
    exp_user_data_model = user_models.ExplorationUserDataModel.get(user_id, exploration_id)
    return exp_user_data_model.rating if exp_user_data_model else None

def get_when_exploration_rated(user_id: str, exploration_id: str) -> Optional[datetime.datetime]:
    if False:
        return 10
    'Fetches the datetime the exploration was last rated by this user, or\n    None if no rating has been awarded.\n\n    Currently this function is only used for testing purposes.\n\n    Args:\n        user_id: str. The id of the user.\n        exploration_id: str. The id of the exploration.\n\n    Returns:\n        datetime.datetime or None. When the exploration was last\n        rated by the user, or None if the user has not previously\n        rated the exploration.\n    '
    exp_user_data_model = user_models.ExplorationUserDataModel.get(user_id, exploration_id)
    return exp_user_data_model.rated_on if exp_user_data_model else None

def get_overall_ratings_for_exploration(exploration_id: str) -> Dict[str, int]:
    if False:
        i = 10
        return i + 15
    "Fetches all ratings for an exploration.\n\n    Args:\n        exploration_id: str. The id of the exploration.\n\n    Returns:\n        dict. A dict whose keys are '1', '2', '3', '4', '5' and whose\n        values are nonnegative integers representing the frequency counts\n        of each rating.\n    "
    exp_summary = exp_fetchers.get_exploration_summary_by_id(exploration_id)
    return exp_summary.ratings