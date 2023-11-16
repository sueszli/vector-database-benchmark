"""Error classes for feedback model audits."""
from __future__ import annotations
from core.jobs.types import base_validation_errors
from core.platform import models
MYPY = False
if MYPY:
    from mypy_imports import feedback_models
(feedback_models,) = models.Registry.import_models([models.Names.FEEDBACK])

class InvalidEntityTypeError(base_validation_errors.BaseAuditError):
    """Error class for models that have invalid entity type."""

    def __init__(self, model: feedback_models.GeneralFeedbackThreadModel) -> None:
        if False:
            i = 10
            return i + 15
        message = 'entity type %s is invalid.' % model.entity_type
        super().__init__(message, model)