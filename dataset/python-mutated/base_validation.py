"""Beam functions and transforms to provide validation for models. The
BaseModelValidator is intended to be a class that other validators can inherit
from. It takes in a Beam PCollection of models and returns a PCollection of the
validation errors found in the input. The Beam.DoFn classes are functions that
are called in the BaseModelValidator to perform validations.

When writing subclasses to BaseModelValidator, call the new added
validation functions in the expand function, and then flatten the output
with the result of the super function.
"""
from __future__ import annotations
import datetime
import enum
import re
from core import feconf
from core.domain import change_domain
from core.jobs import job_utils
from core.jobs.decorators import validation_decorators
from core.jobs.types import base_validation_errors
from core.platform import models
import apache_beam as beam
from typing import Any, Final, Generic, Iterator, Type, TypeVar, Union
MYPY = False
if MYPY:
    from mypy_imports import base_models
(base_models,) = models.Registry.import_models([models.Names.BASE_MODEL])
BASE_MODEL_ID_PATTERN: str = '^[A-Za-z0-9-_]{1,%s}$' % base_models.ID_LENGTH
MAX_CLOCK_SKEW_SECS: Final = datetime.timedelta(seconds=1)
ModelInstanceType = TypeVar('ModelInstanceType', bound='base_models.BaseModel')

class ValidationModes(enum.Enum):
    """Enum for validation modes."""
    NEUTRAL = 'neutral'
    STRICT = 'strict'
    NON_STRICT = 'non_strict'

class ValidateDeletedModel(beam.DoFn):
    """DoFn to check whether models marked for deletion are stale.

    Doesn't use the AuditsExisting decorator because it audits deleted models,
    not existing ones.
    """

    def process(self, entity: base_models.BaseModel) -> Iterator[base_validation_errors.ModelExpiredError]:
        if False:
            i = 10
            return i + 15
        'Yields audit errors that are discovered in the input model.\n\n        Args:\n            entity: datastore_services.Model. Entity to validate.\n\n        Yields:\n            ModelExpiredError. An error class for expired models.\n        '
        cloned_entity = job_utils.clone_model(entity)
        expiration_date = datetime.datetime.utcnow() - feconf.PERIOD_TO_HARD_DELETE_MODELS_MARKED_AS_DELETED
        if cloned_entity.last_updated < expiration_date:
            yield base_validation_errors.ModelExpiredError(cloned_entity)

@validation_decorators.AuditsExisting(base_models.BaseModel)
class ValidateBaseModelId(beam.DoFn):
    """DoFn to validate model ids.

    IMPORTANT: Models with special ID checks should derive from this class and
    override __init__() to assign a different value to self._regex, or replace
    the process() method entirely. Be sure to decorate the new class with that
    specific model type.
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._pattern = BASE_MODEL_ID_PATTERN

    def process(self, entity: base_models.BaseModel) -> Iterator[base_validation_errors.ModelIdRegexError]:
        if False:
            print('Hello World!')
        'Function that defines how to process each entity in a pipeline of\n        models.\n\n        Args:\n            entity: datastore_services.Model. Entity to validate.\n\n        Yields:\n            ModelIdRegexError. An error class for models with invalid IDs.\n        '
        cloned_entity = job_utils.clone_model(entity)
        if not re.match(self._pattern, cloned_entity.id):
            yield base_validation_errors.ModelIdRegexError(cloned_entity, self._pattern)

@validation_decorators.AuditsExisting(base_models.BaseCommitLogEntryModel)
class ValidatePostCommitStatus(beam.DoFn):
    """DoFn to validate post_commit_status."""

    def process(self, entity: base_models.BaseCommitLogEntryModel) -> Iterator[base_validation_errors.InvalidCommitStatusError]:
        if False:
            i = 10
            return i + 15
        'Function validates that post_commit_status is either public or\n        private\n\n        Args:\n            entity: base_models.BaseCommitLogEntryModel. Entity to validate.\n\n        Yields:\n            InvalidCommitStatusError. Error for commit_type validation.\n        '
        cloned_entity = job_utils.clone_model(entity)
        if cloned_entity.post_commit_status not in [feconf.POST_COMMIT_STATUS_PUBLIC, feconf.POST_COMMIT_STATUS_PRIVATE]:
            yield base_validation_errors.InvalidCommitStatusError(cloned_entity)

@validation_decorators.AuditsExisting(base_models.BaseCommitLogEntryModel)
class ValidatePostCommitIsPrivate(beam.DoFn):
    """DoFn to check if post_commit_status is private when
    post_commit_is_private is true and vice-versa.
    """

    def process(self, entity: base_models.BaseCommitLogEntryModel) -> Iterator[base_validation_errors.InvalidPrivateCommitStatusError]:
        if False:
            return 10
        'Function validates that post_commit_is_private is true iff\n        post_commit_status is private\n\n        Args:\n            entity: base_models.BaseCommitLogEntryModel.\n                Entity to validate.\n\n        Yields:\n            InvalidPrivateCommitStatusError. Error for private commit_type\n            validation.\n        '
        cloned_entity = job_utils.clone_model(entity)
        expected_post_commit_is_private = cloned_entity.post_commit_status == feconf.POST_COMMIT_STATUS_PRIVATE
        if cloned_entity.post_commit_is_private != expected_post_commit_is_private:
            yield base_validation_errors.InvalidPrivateCommitStatusError(cloned_entity)

@validation_decorators.AuditsExisting(base_models.BaseCommitLogEntryModel)
class ValidatePostCommitIsPublic(beam.DoFn):
    """DoFn to check if post_commit_status is public when
    post_commit_is_public is true and vice-versa.
    """

    def process(self, entity: base_models.BaseCommitLogEntryModel) -> Iterator[base_validation_errors.InvalidPublicCommitStatusError]:
        if False:
            i = 10
            return i + 15
        'Function validates that post_commit_is_public is true iff\n        post_commit_status is public.\n\n        Args:\n            entity: base_models.BaseCommitLogEntryModel. Entity to validate.\n\n        Yields:\n            InvalidPublicCommitStatusError. Error for public commit_type\n            validation.\n        '
        cloned_entity = job_utils.clone_model(entity)
        expected_post_commit_is_public = cloned_entity.post_commit_status == feconf.POST_COMMIT_STATUS_PUBLIC
        if cloned_entity.post_commit_community_owned != expected_post_commit_is_public:
            yield base_validation_errors.InvalidPublicCommitStatusError(cloned_entity)

@validation_decorators.AuditsExisting(base_models.BaseModel)
class ValidateModelTimestamps(beam.DoFn):
    """DoFn to check whether created_on and last_updated timestamps are valid.
    """

    def process(self, entity: base_models.BaseModel) -> Iterator[Union[base_validation_errors.InconsistentTimestampsError, base_validation_errors.ModelMutatedDuringJobError]]:
        if False:
            while True:
                i = 10
        'Function that defines how to process each entity in a pipeline of\n        models.\n\n        Args:\n            entity: datastore_services.Model. Entity to validate.\n\n        Yields:\n            ModelMutatedDuringJobError. Error for models mutated during the job.\n            InconsistentTimestampsError. Error for models with inconsistent\n            timestamps.\n        '
        cloned_entity = job_utils.clone_model(entity)
        last_updated_corrected = cloned_entity.last_updated + MAX_CLOCK_SKEW_SECS
        if cloned_entity.created_on > last_updated_corrected:
            yield base_validation_errors.InconsistentTimestampsError(cloned_entity)
        current_datetime = datetime.datetime.utcnow()
        last_updated_corrected = cloned_entity.last_updated - MAX_CLOCK_SKEW_SECS
        if last_updated_corrected > current_datetime:
            yield base_validation_errors.ModelMutatedDuringJobError(cloned_entity)

@validation_decorators.AuditsExisting(base_models.BaseModel)
class ValidateModelDomainObjectInstances(beam.DoFn, Generic[ModelInstanceType]):
    """DoFn to check whether the model instance passes the validation of the
    domain object for model.
    """

    def _get_model_domain_object_instance(self, unused_item: ModelInstanceType) -> Any:
        if False:
            i = 10
            return i + 15
        'Returns a domain object instance created from the model.\n\n        This method can be overridden by subclasses, if needed.\n\n        Args:\n            unused_item: datastore_services.Model. Entity to validate.\n\n        Returns:\n            *. A domain object to validate.\n        '
        return None

    def _get_domain_object_validation_type(self, unused_item: ModelInstanceType) -> ValidationModes:
        if False:
            return 10
        'Returns the type of domain object validation to be performed.\n\n        Some of the storage models support a strict/non strict mode depending\n        on whether the model is published or not. Currently the models which\n        provide this feature are collection, exploration and topic models.\n\n        Other models do not support any strict/non strict validation. So,\n        this function returns neutral mode in the base class. It can be\n        overridden by subclasses to enable strict/non strict mode, if needed.\n\n        Args:\n            unused_item: datastore_services.Model. Entity to validate.\n\n        Returns:\n            str. The type of validation mode: neutral, strict or non strict.\n        '
        return ValidationModes.NEUTRAL

    def process(self, entity: ModelInstanceType) -> Iterator[base_validation_errors.ModelDomainObjectValidateError]:
        if False:
            while True:
                i = 10
        'Function that defines how to process each entity in a pipeline of\n        models.\n\n        Args:\n            entity: datastore_services.Model. A domain object to validate.\n\n        Yields:\n            ModelDomainObjectValidateError. Error for domain object validation.\n        '
        try:
            domain_object = self._get_model_domain_object_instance(entity)
            validation_type = self._get_domain_object_validation_type(entity)
            if domain_object is None:
                return
            if validation_type == ValidationModes.NEUTRAL:
                domain_object.validate()
            elif validation_type == ValidationModes.STRICT:
                domain_object.validate(strict=True)
            elif validation_type == ValidationModes.NON_STRICT:
                domain_object.validate(strict=False)
            else:
                raise Exception('Invalid validation type for domain object: %s' % validation_type)
        except Exception as e:
            yield base_validation_errors.ModelDomainObjectValidateError(entity, str(e))

class BaseValidateCommitCmdsSchema(beam.DoFn, Generic[ModelInstanceType]):
    """DoFn to validate schema of commit commands in commit_cmds dict.

    Decorators are not required here as _get_change_domain_class is not
    implemented. This class is used as a parent class in other places.
    """

    def _get_change_domain_class(self, unused_item: ModelInstanceType) -> Type[change_domain.BaseChange]:
        if False:
            while True:
                i = 10
        'Returns a Change domain class.\n\n        This should be implemented by subclasses.\n\n        Args:\n            unused_item: datastore_services.Model. Entity to validate.\n\n        Returns:\n            change_domain.BaseChange. A domain object class for the\n            changes made by commit commands of the model.\n\n        Raises:\n            NotImplementedError. This function has not yet been implemented.\n        '
        raise NotImplementedError('The _get_change_domain_class() method is missing from the derived class. It should be implemented in the derived class.')

    def process(self, entity: ModelInstanceType) -> Iterator[Union[base_validation_errors.CommitCmdsNoneError, base_validation_errors.CommitCmdsValidateError]]:
        if False:
            return 10
        'Validates schema of commit commands in commit_cmds dict.\n\n        Args:\n            entity: datastore_services.Model. Entity to validate.\n\n        Yields:\n            CommitCmdsNoneError. Error for invalid commit cmds id.\n            CommitCmdsValidateError. Error for wrong commit cmds.\n        '
        change_domain_object = self._get_change_domain_class(entity)
        if change_domain_object is None:
            yield base_validation_errors.CommitCmdsNoneError(entity)
            return
        assert isinstance(entity, (base_models.BaseSnapshotMetadataModel, base_models.BaseCommitLogEntryModel))
        for commit_cmd_dict in entity.commit_cmds:
            if not commit_cmd_dict:
                continue
            try:
                change_domain_object(commit_cmd_dict)
            except Exception as e:
                yield base_validation_errors.CommitCmdsValidateError(entity, commit_cmd_dict, str(e))

@validation_decorators.AuditsExisting(base_models.BaseCommitLogEntryModel, base_models.BaseSnapshotMetadataModel)
class ValidateCommitType(beam.DoFn):
    """DoFn to check whether commit type is valid."""

    def process(self, entity: Union[base_models.BaseCommitLogEntryModel, base_models.BaseSnapshotMetadataModel]) -> Iterator[base_validation_errors.InvalidCommitTypeError]:
        if False:
            return 10
        'Function that defines how to process each entity in a pipeline of\n        models.\n\n        Args:\n            entity: datastore_services.Model. Entity to validate.\n\n        Yields:\n            ModelCommitTypeError. Error for commit_type validation.\n        '
        cloned_entity = job_utils.clone_model(entity)
        if cloned_entity.commit_type not in base_models.VersionedModel.COMMIT_TYPE_CHOICES:
            yield base_validation_errors.InvalidCommitTypeError(cloned_entity)