"""Beam DoFns and PTransforms to provide validation of skill models."""
from __future__ import annotations
from core.domain import skill_domain
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import base_validation
from core.platform import models
from typing import Optional, Type
MYPY = False
if MYPY:
    from mypy_imports import skill_models
(skill_models,) = models.Registry.import_models([models.Names.SKILL])

@validation_decorators.AuditsExisting(skill_models.SkillSnapshotMetadataModel)
class ValidateSkillSnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[skill_models.SkillSnapshotMetadataModel]):
    """Overrides _get_change_domain_class for SkillSnapshotMetadataModel."""

    def _get_change_domain_class(self, unused_input_model: skill_models.SkillSnapshotMetadataModel) -> Type[skill_domain.SkillChange]:
        if False:
            print('Hello World!')
        'Returns a change domain class.\n\n        Args:\n            unused_input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            skill_domain.SkillChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        return skill_domain.SkillChange

@validation_decorators.AuditsExisting(skill_models.SkillCommitLogEntryModel)
class ValidateSkillCommitLogEntryModel(base_validation.BaseValidateCommitCmdsSchema[skill_models.SkillCommitLogEntryModel]):
    """Overrides _get_change_domain_class for SkillCommitLogEntryModel."""

    def _get_change_domain_class(self, input_model: skill_models.SkillCommitLogEntryModel) -> Optional[Type[skill_domain.SkillChange]]:
        if False:
            while True:
                i = 10
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            skill_domain.SkillChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        model = job_utils.clone_model(input_model)
        if model.id.startswith('skill'):
            return skill_domain.SkillChange
        else:
            return None