"""Beam DoFns and PTransforms to provide validation of auth models."""
from __future__ import annotations
from core import feconf
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import base_validation
from core.platform import models
from typing import Iterator, List, Tuple, Type, Union
MYPY = False
if MYPY:
    from mypy_imports import auth_models
    from mypy_imports import datastore_services
(auth_models,) = models.Registry.import_models([models.Names.AUTH])
datastore_services = models.Registry.import_datastore_services()

@validation_decorators.AuditsExisting(auth_models.FirebaseSeedModel)
class ValidateFirebaseSeedModelId(base_validation.ValidateBaseModelId):
    """Overrides regex to match the single valid FirebaseSeedModel ID."""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._pattern = auth_models.ONLY_FIREBASE_SEED_MODEL_ID

@validation_decorators.AuditsExisting(auth_models.UserIdByFirebaseAuthIdModel)
class ValidateUserIdByFirebaseAuthIdModelId(base_validation.ValidateBaseModelId):
    """Overrides regex to match the Firebase account ID pattern."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._pattern = feconf.FIREBASE_AUTH_ID_REGEX

@validation_decorators.RelationshipsOf(auth_models.UserAuthDetailsModel)
def user_auth_details_model_relationships(model: Type[auth_models.UserAuthDetailsModel]) -> Iterator[Tuple[datastore_services.Property, List[Type[Union[auth_models.UserIdByFirebaseAuthIdModel, auth_models.UserIdentifiersModel]]]]]:
    if False:
        return 10
    'Yields how the properties of the model relate to the IDs of others.'
    yield (model.firebase_auth_id, [auth_models.UserIdByFirebaseAuthIdModel])
    yield (model.gae_id, [auth_models.UserIdentifiersModel])

@validation_decorators.RelationshipsOf(auth_models.UserIdByFirebaseAuthIdModel)
def user_id_by_firebase_auth_id_model_relationships(model: Type[auth_models.UserIdByFirebaseAuthIdModel]) -> Iterator[Tuple[datastore_services.Property, List[Type[auth_models.UserAuthDetailsModel]]]]:
    if False:
        return 10
    'Yields how the properties of the model relate to the IDs of others.'
    yield (model.user_id, [auth_models.UserAuthDetailsModel])

@validation_decorators.RelationshipsOf(auth_models.UserIdentifiersModel)
def user_identifiers_model_relationships(model: Type[auth_models.UserIdentifiersModel]) -> Iterator[Tuple[datastore_services.Property, List[Type[auth_models.UserAuthDetailsModel]]]]:
    if False:
        print('Hello World!')
    'Yields how the properties of the model relate to the IDs of others.'
    yield (model.user_id, [auth_models.UserAuthDetailsModel])