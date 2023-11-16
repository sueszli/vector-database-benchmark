"""Error classes for topic model audits."""
from __future__ import annotations
from core.jobs.types import base_validation_errors
from core.platform import models
MYPY = False
if MYPY:
    from mypy_imports import topic_models
(topic_models,) = models.Registry.import_models([models.Names.TOPIC])

class ModelCanonicalNameMismatchError(base_validation_errors.BaseAuditError):
    """Error class for models that have mismatching names."""

    def __init__(self, model: topic_models.TopicModel) -> None:
        if False:
            i = 10
            return i + 15
        message = 'Entity name %s in lowercase does not match canonical name %s' % (model.name, model.canonical_name)
        super().__init__(message, model)