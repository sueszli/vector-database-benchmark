"""Beam DoFns and PTransforms to provide validation of feedback models."""
from __future__ import annotations
from core.domain import feedback_services
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.types import feedback_validation_errors
from core.jobs.types import model_property
from core.platform import models
import apache_beam as beam
from typing import Iterator, List, Tuple, Type
MYPY = False
if MYPY:
    from mypy_imports import exp_models
    from mypy_imports import feedback_models
(exp_models, feedback_models) = models.Registry.import_models([models.Names.EXPLORATION, models.Names.FEEDBACK])

@validation_decorators.AuditsExisting(feedback_models.GeneralFeedbackThreadModel)
class ValidateEntityType(beam.DoFn):
    """DoFn to validate the entity type."""

    def process(self, input_model: feedback_models.GeneralFeedbackThreadModel) -> Iterator[feedback_validation_errors.InvalidEntityTypeError]:
        if False:
            i = 10
            return i + 15
        'Function that checks if the entity type is valid\n\n        Args:\n            input_model: feedback_models.GeneralFeedbackThreadModel.\n                Entity to validate.\n\n        Yields:\n            InvalidEntityTypeError. Error for models with invalid entity type.\n        '
        model = job_utils.clone_model(input_model)
        if model.entity_type not in feedback_services.TARGET_TYPE_TO_TARGET_MODEL:
            yield feedback_validation_errors.InvalidEntityTypeError(model)

@validation_decorators.RelationshipsOf(feedback_models.FeedbackAnalyticsModel)
def feedback_analytics_model_relationships(model: Type[feedback_models.FeedbackAnalyticsModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[exp_models.ExplorationModel]]]]:
    if False:
        i = 10
        return i + 15
    'Yields how the properties of the model relates to the ID of others.'
    yield (model.id, [exp_models.ExplorationModel])