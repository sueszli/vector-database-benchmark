"""Error classes for improvements model."""
from __future__ import annotations
from core.jobs.types import base_validation_errors
from core.platform import models
MYPY = False
if MYPY:
    from mypy_imports import improvements_models
(improvements_models,) = models.Registry.import_models([models.Names.IMPROVEMENTS])

class InvalidCompositeEntityError(base_validation_errors.BaseAuditError):
    """Error class for models that have invalid composite entity id."""

    def __init__(self, model: improvements_models.ExplorationStatsTaskEntryModel) -> None:
        if False:
            i = 10
            return i + 15
        message = 'model has invalid composite entity %s' % model.composite_entity_id
        super().__init__(message, model)