"""Beam DoFns and PTransforms to provide validation of exploration models."""
from __future__ import annotations
from core.domain import exp_domain
from core.domain import rights_domain
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import base_validation
from core.jobs.types import model_property
from core.platform import models
from typing import Iterator, List, Optional, Tuple, Type, Union
MYPY = False
if MYPY:
    from mypy_imports import exp_models
    from mypy_imports import story_models
(exp_models, story_models) = models.Registry.import_models([models.Names.EXPLORATION, models.Names.STORY])

@validation_decorators.AuditsExisting(exp_models.ExplorationSnapshotMetadataModel)
class ValidateExplorationSnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[exp_models.ExplorationSnapshotMetadataModel]):
    """Overrides _get_change_domain_class for exploration models """

    def _get_change_domain_class(self, input_model: exp_models.ExplorationSnapshotMetadataModel) -> Type[exp_domain.ExplorationChange]:
        if False:
            print('Hello World!')
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            exp_domain.ExplorationChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        return exp_domain.ExplorationChange

@validation_decorators.RelationshipsOf(exp_models.ExplorationContextModel)
def exploration_context_model_relationships(model: Type[exp_models.ExplorationContextModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[Union[story_models.StoryModel, exp_models.ExplorationModel]]]]]:
    if False:
        for i in range(10):
            print('nop')
    'Yields how the properties of the model relates to the ID of others.'
    yield (model.story_id, [story_models.StoryModel])
    yield (model.id, [exp_models.ExplorationModel])

@validation_decorators.RelationshipsOf(exp_models.ExpSummaryModel)
def exp_summary_model_relationships(model: Type[exp_models.ExpSummaryModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[Union[exp_models.ExplorationModel, exp_models.ExplorationRightsModel]]]]]:
    if False:
        print('Hello World!')
    'Yields how the properties of the model relates to the ID of others.'
    yield (model.id, [exp_models.ExplorationModel])
    yield (model.id, [exp_models.ExplorationRightsModel])

@validation_decorators.AuditsExisting(exp_models.ExplorationRightsSnapshotMetadataModel)
class ValidateExplorationRightsSnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[exp_models.ExplorationRightsSnapshotMetadataModel]):
    """Overrides _get_change_domain_class for exploration models """

    def _get_change_domain_class(self, input_model: exp_models.ExplorationRightsSnapshotMetadataModel) -> Type[rights_domain.ExplorationRightsChange]:
        if False:
            while True:
                i = 10
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            rights_domain.ExplorationRightsChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        return rights_domain.ExplorationRightsChange

@validation_decorators.AuditsExisting(exp_models.ExplorationCommitLogEntryModel)
class ValidateExplorationCommitLogEntryModel(base_validation.BaseValidateCommitCmdsSchema[exp_models.ExplorationCommitLogEntryModel]):
    """Overrides _get_change_domain_class for exploration models """

    def _get_change_domain_class(self, input_model: exp_models.ExplorationCommitLogEntryModel) -> Optional[Type[Union[rights_domain.ExplorationRightsChange, exp_domain.ExplorationChange]]]:
        if False:
            i = 10
            return i + 15
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            rights_domain.ExplorationRightsChange|exp_domain.ExplorationChange.\n            A domain object class for the changes made by commit commands of\n            the model.\n        '
        model = job_utils.clone_model(input_model)
        if model.id.startswith('rights'):
            return rights_domain.ExplorationRightsChange
        elif model.id.startswith('exploration'):
            return exp_domain.ExplorationChange
        else:
            return None