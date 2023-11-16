"""Beam DoFns and PTransforms to provide validation of subtopic models."""
from __future__ import annotations
from core.domain import subtopic_page_domain
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import base_validation
from core.platform import models
from typing import Optional, Type
MYPY = False
if MYPY:
    from mypy_imports import subtopic_models
(subtopic_models,) = models.Registry.import_models([models.Names.SUBTOPIC])

@validation_decorators.AuditsExisting(subtopic_models.SubtopicPageSnapshotMetadataModel)
class ValidateSubtopicPageSnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[subtopic_models.SubtopicPageSnapshotMetadataModel]):
    """Overrides _get_change_domain_class for SubtopicPageSnapshotMetadataModel.
    """

    def _get_change_domain_class(self, unused_input_model: subtopic_models.SubtopicPageSnapshotMetadataModel) -> Type[subtopic_page_domain.SubtopicPageChange]:
        if False:
            print('Hello World!')
        'Returns a change domain class.\n\n        Args:\n            unused_input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            subtopic_page_domain.SubtopicPageChange. A domain object class for\n            the changes made by commit commands of the model.\n        '
        return subtopic_page_domain.SubtopicPageChange

@validation_decorators.AuditsExisting(subtopic_models.SubtopicPageCommitLogEntryModel)
class ValidateSubtopicPageCommitLogEntryModel(base_validation.BaseValidateCommitCmdsSchema[subtopic_models.SubtopicPageCommitLogEntryModel]):
    """Overrides _get_change_domain_class for SubtopicPageCommitLogEntryModel.
    """

    def _get_change_domain_class(self, input_model: subtopic_models.SubtopicPageCommitLogEntryModel) -> Optional[Type[subtopic_page_domain.SubtopicPageChange]]:
        if False:
            print('Hello World!')
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            subtopic_page_domain.SubtopicPageChange. A domain object class for\n            the changes made by commit commands of the model.\n        '
        model = job_utils.clone_model(input_model)
        if model.id.startswith('subtopicpage'):
            return subtopic_page_domain.SubtopicPageChange
        else:
            return None