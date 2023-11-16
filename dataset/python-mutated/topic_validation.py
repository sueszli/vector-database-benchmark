"""Beam DoFns and PTransforms to provide validation of topic models."""
from __future__ import annotations
from core.domain import topic_domain
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import base_validation
from core.jobs.types import model_property
from core.jobs.types import topic_validation_errors
from core.platform import models
import apache_beam as beam
from typing import Iterator, List, Optional, Tuple, Type, Union
MYPY = False
if MYPY:
    from mypy_imports import topic_models
(topic_models,) = models.Registry.import_models([models.Names.TOPIC])

@validation_decorators.AuditsExisting(topic_models.TopicModel)
class ValidateCanonicalNameMatchesNameInLowercase(beam.DoFn):
    """DoFn to validate canonical name matching with lower case name."""

    def process(self, input_model: topic_models.TopicModel) -> Iterator[topic_validation_errors.ModelCanonicalNameMismatchError]:
        if False:
            i = 10
            return i + 15
        'Function that validate that canonical name of the model is same as\n        name of the model in lowercase.\n\n        Args:\n            input_model: datastore_services.Model. TopicModel to validate.\n\n        Yields:\n            ModelCanonicalNameMismatchError. An error class for\n            name mismatched models.\n        '
        model = job_utils.clone_model(input_model)
        name = model.name
        if name.lower() != model.canonical_name:
            yield topic_validation_errors.ModelCanonicalNameMismatchError(model)

@validation_decorators.AuditsExisting(topic_models.TopicSnapshotMetadataModel)
class ValidateTopicSnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[topic_models.TopicSnapshotMetadataModel]):
    """Overrides _get_change_domain_class for TopicSnapshotMetadataModel."""

    def _get_change_domain_class(self, input_model: topic_models.TopicSnapshotMetadataModel) -> Type[topic_domain.TopicChange]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            topic_domain.TopicChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        return topic_domain.TopicChange

@validation_decorators.AuditsExisting(topic_models.TopicRightsSnapshotMetadataModel)
class ValidateTopicRightsSnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[topic_models.TopicRightsSnapshotMetadataModel]):
    """Overrides _get_change_domain_class for TopicRightsSnapshotMetadataModel.
    """

    def _get_change_domain_class(self, input_model: topic_models.TopicRightsSnapshotMetadataModel) -> Type[topic_domain.TopicRightsChange]:
        if False:
            while True:
                i = 10
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            topic_domain.TopicRightsChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        return topic_domain.TopicRightsChange

@validation_decorators.AuditsExisting(topic_models.TopicCommitLogEntryModel)
class ValidateTopicCommitLogEntryModel(base_validation.BaseValidateCommitCmdsSchema[topic_models.TopicCommitLogEntryModel]):
    """Overrides _get_change_domain_class for TopicCommitLogEntryModel.
    """

    def _get_change_domain_class(self, input_model: topic_models.TopicCommitLogEntryModel) -> Optional[Type[Union[topic_domain.TopicRightsChange, topic_domain.TopicChange]]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            topic_domain.TopicRightsChange|topic_domain.TopicChange.\n            A domain object class for the changes made by commit commands of\n            the model.\n        '
        model = job_utils.clone_model(input_model)
        if model.id.startswith('rights'):
            return topic_domain.TopicRightsChange
        elif model.id.startswith('topic'):
            return topic_domain.TopicChange
        else:
            return None

@validation_decorators.RelationshipsOf(topic_models.TopicSummaryModel)
def topic_summary_model_relationships(model: Type[topic_models.TopicSummaryModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[Union[topic_models.TopicModel, topic_models.TopicRightsModel]]]]]:
    if False:
        for i in range(10):
            print('nop')
    'Yields how the properties of the model relates to the ID of others.'
    yield (model.id, [topic_models.TopicModel])
    yield (model.id, [topic_models.TopicRightsModel])