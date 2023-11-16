"""Unit tests for jobs.transforms.auth_validation."""
from __future__ import annotations
from core import feconf
from core.jobs import job_test_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import auth_validation
from core.jobs.types import base_validation_errors
from core.platform import models
from core.tests import test_utils
import apache_beam as beam
MYPY = False
if MYPY:
    from mypy_imports import auth_models
(auth_models,) = models.Registry.import_models([models.Names.AUTH])

class ValidateFirebaseSeedModelIdTests(job_test_utils.PipelinedTestBase):

    def test_reports_error_for_invalid_id(self) -> None:
        if False:
            i = 10
            return i + 15
        model_with_invalid_id = auth_models.FirebaseSeedModel(id='2', created_on=self.NOW, last_updated=self.NOW)
        output = self.pipeline | beam.Create([model_with_invalid_id]) | beam.ParDo(auth_validation.ValidateFirebaseSeedModelId())
        self.assert_pcoll_equal(output, [base_validation_errors.ModelIdRegexError(model_with_invalid_id, auth_models.ONLY_FIREBASE_SEED_MODEL_ID)])

    def test_reports_nothing_for_valid_id(self) -> None:
        if False:
            return 10
        model_with_valid_id = auth_models.FirebaseSeedModel(id=auth_models.ONLY_FIREBASE_SEED_MODEL_ID, created_on=self.NOW, last_updated=self.NOW)
        output = self.pipeline | beam.Create([model_with_valid_id]) | beam.ParDo(auth_validation.ValidateFirebaseSeedModelId())
        self.assert_pcoll_equal(output, [])

class ValidateUserIdByFirebaseAuthIdModelIdTests(job_test_utils.PipelinedTestBase):

    def test_reports_error_for_invalid_id(self) -> None:
        if False:
            print('Hello World!')
        model_with_invalid_id = auth_models.UserIdByFirebaseAuthIdModel(id='-!\'"', user_id='1', created_on=self.NOW, last_updated=self.NOW)
        output = self.pipeline | beam.Create([model_with_invalid_id]) | beam.ParDo(auth_validation.ValidateUserIdByFirebaseAuthIdModelId())
        self.assert_pcoll_equal(output, [base_validation_errors.ModelIdRegexError(model_with_invalid_id, feconf.FIREBASE_AUTH_ID_REGEX)])

    def test_reports_nothing_for_valid_id(self) -> None:
        if False:
            while True:
                i = 10
        model_with_valid_id = auth_models.UserIdByFirebaseAuthIdModel(id='123', user_id='1', created_on=self.NOW, last_updated=self.NOW)
        output = self.pipeline | beam.Create([model_with_valid_id]) | beam.ParDo(auth_validation.ValidateUserIdByFirebaseAuthIdModelId())
        self.assert_pcoll_equal(output, [])

class RelationshipsOfTests(test_utils.TestBase):

    def test_user_auth_details_model_relationships(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('UserAuthDetailsModel', 'firebase_auth_id'), ['UserIdByFirebaseAuthIdModel'])
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('UserAuthDetailsModel', 'gae_id'), ['UserIdentifiersModel'])

    def test_user_id_by_firebase_auth_id_model_relationships(self) -> None:
        if False:
            print('Hello World!')
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('UserIdByFirebaseAuthIdModel', 'user_id'), ['UserAuthDetailsModel'])

    def test_user_identifiers_model_relationships(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('UserIdentifiersModel', 'user_id'), ['UserAuthDetailsModel'])