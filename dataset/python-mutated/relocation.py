from __future__ import annotations
from enum import Enum, unique
from uuid import uuid4
from django.db import models
from sentry.backup.scopes import RelocationScope
from sentry.db.models import BoundedBigIntegerField, region_silo_only_model
from sentry.db.models.base import DefaultFieldsModel, sane_repr
from sentry.db.models.fields.foreignkey import FlexibleForeignKey
from sentry.db.models.fields.uuid import UUIDField

def default_guid():
    if False:
        while True:
            i = 10
    return uuid4().hex

@region_silo_only_model
class Relocation(DefaultFieldsModel):
    """
    Represents a single relocation instance. The relocation may be attempted multiple times, but we
    keep a mapping of 1 `Relocation` model per file upload.
    """
    __relocation_scope__ = RelocationScope.Excluded
    __relocation_dependencies__ = {'sentry.User'}

    class Step(Enum):
        UNKNOWN = 0
        UPLOADING = 1
        PREPROCESSING = 2
        VALIDATING = 3
        IMPORTING = 4
        POSTPROCESSING = 5
        NOTIFYING = 6
        COMPLETED = 7

        @classmethod
        def get_choices(cls) -> list[tuple[int, str]]:
            if False:
                while True:
                    i = 10
            return [(key.value, key.name) for key in cls]

    class Status(Enum):
        IN_PROGRESS = 0
        FAILURE = 1
        SUCCESS = 2

        @classmethod
        def get_choices(cls) -> list[tuple[int, str]]:
            if False:
                i = 10
                return i + 15
            return [(key.value, key.name) for key in cls]
    creator_id = BoundedBigIntegerField()
    owner_id = BoundedBigIntegerField()
    uuid = UUIDField(db_index=True, unique=True, default=default_guid)
    step = models.SmallIntegerField(choices=Step.get_choices(), default=None)
    status = models.SmallIntegerField(choices=Status.get_choices(), default=Status.IN_PROGRESS.value)
    want_org_slugs = models.JSONField(default=list)
    want_usernames = models.JSONField(null=True)
    latest_notified = models.SmallIntegerField(choices=Step.get_choices(), null=True, default=None)
    latest_task = models.CharField(max_length=64, default='')
    latest_task_attempts = models.SmallIntegerField(default=0)
    failure_reason = models.CharField(max_length=256, null=True, default=None)
    __repr__ = sane_repr('owner', 'uuid')

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_relocation'

@region_silo_only_model
class RelocationFile(DefaultFieldsModel):
    """
    A `RelocationFile` is an association between a `Relocation` and a `File`.

    This model should be created in an atomic transaction with the `Relocation` and `File` it points
    to.
    """
    __relocation_scope__ = RelocationScope.Excluded

    class Kind(Enum):
        UNKNOWN = 0
        RAW_USER_DATA = 1
        NORMALIZED_USER_DATA = 2
        BASELINE_CONFIG_VALIDATION_DATA = 3
        COLLIDING_USERS_VALIDATION_DATA = 4

        @classmethod
        def get_choices(cls) -> list[tuple[int, str]]:
            if False:
                i = 10
                return i + 15
            return [(key.value, key.name) for key in cls]

        def __str__(self):
            if False:
                i = 10
                return i + 15
            if self.name == 'RAW_USER_DATA':
                return 'raw-relocation-data'
            elif self.name == 'NORMALIZED_USER_DATA':
                return 'normalized-relocation-data'
            elif self.name == 'BASELINE_CONFIG_VALIDATION_DATA':
                return 'baseline-config'
            elif self.name == 'COLLIDING_USERS_VALIDATION_DATA':
                return 'colliding-users'
            else:
                raise ValueError('Cannot extract a filename from `RelocationFile.Kind.UNKNOWN`.')

        def to_filename(self, ext: str):
            if False:
                for i in range(10):
                    print('nop')
            return str(self) + '.' + ext
    relocation = FlexibleForeignKey('sentry.Relocation')
    file = FlexibleForeignKey('sentry.File')
    kind = models.SmallIntegerField(choices=Kind.get_choices())
    __repr__ = sane_repr('relocation', 'file')

    class Meta:
        unique_together = (('relocation', 'file'),)
        app_label = 'sentry'
        db_table = 'sentry_relocationfile'

@unique
class ValidationStatus(Enum):
    """
    The statuses here are ordered numerically by completeness: `TIMEOUT` is more definite than
    `IN_PROGRESS`, `FAILURE` is more definite than `TIMEOUT`, and so on. If a
    `RelocationValidationAttempt` resolves with a `ValidationStatus` greater than the one already on
    its owning `RelocationValidation`, the new `ValidationStatus` should replace the old.
    """
    IN_PROGRESS = 0
    TIMEOUT = 1
    FAILURE = 2
    INVALID = 3
    VALID = 4

    @classmethod
    def get_choices(cls) -> list[tuple[int, str]]:
        if False:
            while True:
                i = 10
        return [(key.value, key.name) for key in cls]

@region_silo_only_model
class RelocationValidation(DefaultFieldsModel):
    """
    Stores general information about whether or not the associated `Relocation` passed its
    validation run.

    This model essentially unifies the possibly multiple `RelocationValidationAttempt`s that
    represent individual validation runs.
    """
    __relocation_scope__ = RelocationScope.Excluded
    relocation = FlexibleForeignKey('sentry.Relocation')
    status = status = models.SmallIntegerField(choices=ValidationStatus.get_choices(), default=ValidationStatus.IN_PROGRESS.value)
    attempts = models.SmallIntegerField(default=0)

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_relocationvalidation'

@region_silo_only_model
class RelocationValidationAttempt(DefaultFieldsModel):
    """
    Represents a single Google CloudBuild validation run invocation, and tracks it over its
    lifetime.
    """
    __relocation_scope__ = RelocationScope.Excluded
    relocation = FlexibleForeignKey('sentry.Relocation')
    relocation_validation = FlexibleForeignKey('sentry.RelocationValidation')
    status = status = models.SmallIntegerField(choices=ValidationStatus.get_choices(), default=ValidationStatus.IN_PROGRESS.value)
    build_id = UUIDField(db_index=True, unique=True)

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_relocationvalidationattempt'