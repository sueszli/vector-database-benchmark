"""Beam DoFns and PTransforms to provide validation of improvements models."""
from __future__ import annotations
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.types import improvements_validation_errors
from core.platform import models
import apache_beam as beam
from typing import Iterator
MYPY = False
if MYPY:
    from mypy_imports import improvements_models
(improvements_models,) = models.Registry.import_models([models.Names.IMPROVEMENTS])

@validation_decorators.AuditsExisting(improvements_models.ExplorationStatsTaskEntryModel)
class ValidateCompositeEntityId(beam.DoFn):
    """DoFn to validate the composite entity id."""

    def process(self, input_model: improvements_models.ExplorationStatsTaskEntryModel) -> Iterator[improvements_validation_errors.InvalidCompositeEntityError]:
        if False:
            while True:
                i = 10
        'Function that checks if the composite entity id is valid\n\n        Args:\n            input_model: improvements_models.ExplorationStatsTaskEntryModel.\n                Entity to validate.\n\n        Yields:\n            InvalidCompositeEntityError. Error for models with\n            invalid composite entity.\n        '
        model = job_utils.clone_model(input_model)
        expected_composite_entity_id = improvements_models.ExplorationStatsTaskEntryModel.generate_composite_entity_id(model.entity_type, model.entity_id, model.entity_version)
        if model.composite_entity_id != expected_composite_entity_id:
            yield improvements_validation_errors.InvalidCompositeEntityError(model)