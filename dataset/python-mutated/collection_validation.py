"""Beam DoFns and PTransforms to provide validation of collection models."""
from __future__ import annotations
from core.domain import collection_domain
from core.domain import rights_domain
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import base_validation
from core.jobs.types import model_property
from core.platform import models
from typing import Iterator, List, Optional, Tuple, Type, Union
MYPY = False
if MYPY:
    from mypy_imports import collection_models
(collection_models,) = models.Registry.import_models([models.Names.COLLECTION])

@validation_decorators.AuditsExisting(collection_models.CollectionSnapshotMetadataModel)
class ValidateCollectionSnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[collection_models.CollectionSnapshotMetadataModel]):
    """Overrides _get_change_domain_class for CollectionSnapshotMetadataModel.
    """

    def _get_change_domain_class(self, input_model: collection_models.CollectionSnapshotMetadataModel) -> Type[collection_domain.CollectionChange]:
        if False:
            print('Hello World!')
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            collection_domain.CollectionChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        return collection_domain.CollectionChange

@validation_decorators.RelationshipsOf(collection_models.CollectionSummaryModel)
def collection_summary_model_relationships(model: Type[collection_models.CollectionSummaryModel]) -> Iterator[Tuple[model_property.PropertyType, List[Type[Union[collection_models.CollectionModel, collection_models.CollectionRightsModel]]]]]:
    if False:
        i = 10
        return i + 15
    'Yields how the properties of the model relates to the ID of others.'
    yield (model.id, [collection_models.CollectionModel])
    yield (model.id, [collection_models.CollectionRightsModel])

@validation_decorators.AuditsExisting(collection_models.CollectionRightsSnapshotMetadataModel)
class ValidateCollectionRightsSnapshotMetadataModel(base_validation.BaseValidateCommitCmdsSchema[collection_models.CollectionRightsSnapshotMetadataModel]):
    """Overrides _get_change_domain_class for
    CollectionRightsSnapshotMetadataModel.
    """

    def _get_change_domain_class(self, input_model: collection_models.CollectionRightsSnapshotMetadataModel) -> Type[rights_domain.CollectionRightsChange]:
        if False:
            print('Hello World!')
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            rights_domain.CollectionRightsChange. A domain object class for the\n            changes made by commit commands of the model.\n        '
        return rights_domain.CollectionRightsChange

@validation_decorators.AuditsExisting(collection_models.CollectionCommitLogEntryModel)
class ValidateCollectionCommitLogEntryModel(base_validation.BaseValidateCommitCmdsSchema[collection_models.CollectionCommitLogEntryModel]):
    """Overrides _get_change_domain_class for CollectionCommitLogEntryModel."""

    def _get_change_domain_class(self, input_model: collection_models.CollectionCommitLogEntryModel) -> Optional[Type[Union[rights_domain.CollectionRightsChange, collection_domain.CollectionChange]]]:
        if False:
            return 10
        'Returns a change domain class.\n\n        Args:\n            input_model: datastore_services.Model. Entity to validate.\n\n        Returns:\n            collection_domain.CollectionChange|\n            rights_domain.CollectionRightsChange.\n            A domain object class for the changes made by commit commands of\n            the model.\n        '
        model = job_utils.clone_model(input_model)
        if model.id.startswith('rights'):
            return rights_domain.CollectionRightsChange
        elif model.id.startswith('collection'):
            return collection_domain.CollectionChange
        else:
            return None