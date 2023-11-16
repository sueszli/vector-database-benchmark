"""Beam DoFns and PTransforms to provide validation of question models."""
from __future__ import annotations
from core.domain import question_domain
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import base_validation
from core.jobs.types import model_property
from core.platform import models
from typing import Iterator, List, Optional, Tuple, Type, Union
MYPY = False
if MYPY:
    from mypy_imports import datastore_services
    from mypy_imports import question_models
    from mypy_imports import skill_models
(question_models, skill_models) = models.Registry.import_models([models.Names.QUESTION, models.Names.SKILL])
datastore_services = models.Registry.import_datastore_services()

@validation_decorators.AuditsExisting(question_models.QuestionSnapshotMetadataModel)
class ValidateQuestionSnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[question_models.QuestionSnapshotMetadataModel]):
    """Overrides _get_change_domain_class for QuestionSnapshotMetadataModel."""

    def _get_change_domain_class(self, input_model: question_models.QuestionSnapshotMetadataModel) -> Type[question_domain.QuestionChange]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            question_domain.QuestionChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        return question_domain.QuestionChange

@validation_decorators.RelationshipsOf(question_models.QuestionSkillLinkModel)
def question_skill_link_model_relationships(model: Type[question_models.QuestionSkillLinkModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[Union[question_models.QuestionModel, skill_models.SkillModel]]]]]:
    if False:
        i = 10
        return i + 15
    'Yields how the properties of the model relates to the ID of others.'
    yield (model.id, [question_models.QuestionModel])
    yield (model.skill_id, [skill_models.SkillModel])

@validation_decorators.RelationshipsOf(question_models.QuestionCommitLogEntryModel)
def question_commit_log_entry_model_relationships(model: Type[question_models.QuestionCommitLogEntryModel]) -> Iterator[Tuple[datastore_services.Property, List[Type[question_models.QuestionModel]]]]:
    if False:
        i = 10
        return i + 15
    'Yields how the properties of the model relates to the ID of others.'
    yield (model.question_id, [question_models.QuestionModel])

@validation_decorators.RelationshipsOf(question_models.QuestionSummaryModel)
def question_summary_model_relationships(model: Type[question_models.QuestionSummaryModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[question_models.QuestionModel]]]]:
    if False:
        for i in range(10):
            print('nop')
    'Yields how the properties of the model relates to the ID of others.'
    yield (model.id, [question_models.QuestionModel])

@validation_decorators.AuditsExisting(question_models.QuestionCommitLogEntryModel)
class ValidateQuestionCommitLogEntryModel(base_validation.BaseValidateCommitCmdsSchema[question_models.QuestionCommitLogEntryModel]):
    """Overrides _get_change_domain_class for QuestionCommitLogEntryModel."""

    def _get_change_domain_class(self, input_model: question_models.QuestionCommitLogEntryModel) -> Optional[Type[question_domain.QuestionChange]]:
        if False:
            return 10
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            question_domain.QuestionChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        model = job_utils.clone_model(input_model)
        if model.id.startswith('question'):
            return question_domain.QuestionChange
        else:
            return None