"""Error classes for user model audits."""
from __future__ import annotations
from core.jobs.types import base_validation_errors
from core.platform import models
from typing import List
MYPY = False
if MYPY:
    from mypy_imports import user_models
(base_models, user_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.USER])

class ModelIncorrectKeyError(base_validation_errors.BaseAuditError):
    """Error class for incorrect key in PendingDeletionRequestModel."""

    def __init__(self, model: user_models.PendingDeletionRequestModel, incorrect_keys: List[str]) -> None:
        if False:
            while True:
                i = 10
        message = 'contains keys %s are not allowed' % incorrect_keys
        super().__init__(message, model)

class DraftChangeListLastUpdatedNoneError(base_validation_errors.BaseAuditError):
    """Error class for models with draft change list but draft change list
    last_updated is None.
    """

    def __init__(self, model: user_models.ExplorationUserDataModel) -> None:
        if False:
            print('Hello World!')
        message = 'draft change list %s exists but draft change list last updated is None' % model.draft_change_list
        super().__init__(message, model)

class DraftChangeListLastUpdatedInvalidError(base_validation_errors.BaseAuditError):
    """Error class for models with invalid draft change list last_updated."""

    def __init__(self, model: user_models.ExplorationUserDataModel) -> None:
        if False:
            return 10
        message = 'draft change list last updated %s is greater than the time when job was run' % model.draft_change_list_last_updated
        super().__init__(message, model)

class ArchivedModelNotMarkedDeletedError(base_validation_errors.BaseAuditError):
    """Error class for models which are archived but not deleted."""

    def __init__(self, model: user_models.UserQueryModel) -> None:
        if False:
            i = 10
            return i + 15
        message = 'model is archived but not marked as deleted'
        super().__init__(message, model)