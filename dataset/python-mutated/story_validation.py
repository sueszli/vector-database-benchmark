"""Beam DoFns and PTransforms to provide validation of story models."""
from __future__ import annotations
from core.domain import story_domain
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import base_validation
from core.platform import models
from typing import Optional, Type
MYPY = False
if MYPY:
    from mypy_imports import story_models
(story_models,) = models.Registry.import_models([models.Names.STORY])

@validation_decorators.AuditsExisting(story_models.StorySnapshotMetadataModel)
class ValidateStorySnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[story_models.StorySnapshotMetadataModel]):
    """Overrides _get_change_domain_class for StorySnapshotMetadataModel."""

    def _get_change_domain_class(self, unused_input_model: story_models.StorySnapshotMetadataModel) -> Type[story_domain.StoryChange]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a change domain class.\n\n        Args:\n            unused_input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            story_domain.StoryChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        return story_domain.StoryChange

@validation_decorators.AuditsExisting(story_models.StoryCommitLogEntryModel)
class ValidateStoryCommitLogEntryModel(base_validation.BaseValidateCommitCmdsSchema[story_models.StoryCommitLogEntryModel]):
    """Overrides _get_change_domain_class for StoryCommitLogEntryModel."""

    def _get_change_domain_class(self, input_model: story_models.StoryCommitLogEntryModel) -> Optional[Type[story_domain.StoryChange]]:
        if False:
            print('Hello World!')
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            story_domain.StoryChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        model = job_utils.clone_model(input_model)
        if model.id.startswith('story'):
            return story_domain.StoryChange
        else:
            return None