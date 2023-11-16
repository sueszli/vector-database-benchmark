"""Unit tests for user model validator errors."""
from __future__ import annotations
import datetime
from core import feconf
from core.jobs.types import base_validation_errors
from core.jobs.types import base_validation_errors_test
from core.jobs.types import user_validation_errors
from core.platform import models
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import user_models
(base_models, user_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.USER])
datastore_services = models.Registry.import_datastore_services()

class ModelIncorrectKeyErrorTests(base_validation_errors_test.AuditErrorsTestBase):

    def test_message(self) -> None:
        if False:
            i = 10
            return i + 15
        model = user_models.PendingDeletionRequestModel(id='test')
        incorrect_keys = ['incorrect key']
        error = user_validation_errors.ModelIncorrectKeyError(model, incorrect_keys)
        self.assertEqual(error.stderr, 'ModelIncorrectKeyError in PendingDeletionRequestModel(id="test"): contains keys %s are not allowed' % incorrect_keys)

class ModelIdRegexErrorTests(base_validation_errors_test.AuditErrorsTestBase):

    def test_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        model = base_models.BaseModel(id='?!"', created_on=self.YEAR_AGO, last_updated=self.NOW)
        error = base_validation_errors.ModelIdRegexError(model, '[abc]{3}')
        self.assertEqual(error.stderr, 'ModelIdRegexError in BaseModel(id="?!\\""): id does not match the expected regex="[abc]{3}"')

class DraftChangeListLastUpdatedNoneErrorTests(base_validation_errors_test.AuditErrorsTestBase):

    def test_message(self) -> None:
        if False:
            return 10
        draft_change_list = [{'cmd': 'edit_exploration_property', 'property_name': 'objective', 'new_value': 'the objective'}]
        model = user_models.ExplorationUserDataModel(id='123', user_id='test', exploration_id='exploration_id', draft_change_list=draft_change_list, draft_change_list_last_updated=None, created_on=self.YEAR_AGO, last_updated=self.YEAR_AGO)
        error = user_validation_errors.DraftChangeListLastUpdatedNoneError(model)
        self.assertEqual(error.stderr, 'DraftChangeListLastUpdatedNoneError in ExplorationUserDataModel(id="123"): draft change list %s exists but draft change list last updated is None' % draft_change_list)

class DraftChangeListLastUpdatedInvalidErrorTests(base_validation_errors_test.AuditErrorsTestBase):

    def test_message(self) -> None:
        if False:
            return 10
        draft_change_list = [{'cmd': 'edit_exploration_property', 'property_name': 'objective', 'new_value': 'the objective'}]
        last_updated = self.NOW + datetime.timedelta(days=5)
        model = user_models.ExplorationUserDataModel(id='123', user_id='test', exploration_id='exploration_id', draft_change_list=draft_change_list, draft_change_list_last_updated=last_updated, created_on=self.YEAR_AGO, last_updated=self.NOW)
        error = user_validation_errors.DraftChangeListLastUpdatedInvalidError(model)
        self.assertEqual(error.stderr, 'DraftChangeListLastUpdatedInvalidError in ExplorationUserDataModel(id="123"): draft change list last updated %s is greater than the time when job was run' % last_updated)

class ArchivedModelNotMarkedDeletedErrorTests(base_validation_errors_test.AuditErrorsTestBase):

    def test_message(self) -> None:
        if False:
            while True:
                i = 10
        model = user_models.UserQueryModel(id='test', submitter_id='submitter', created_on=self.NOW, last_updated=self.NOW, query_status=feconf.USER_QUERY_STATUS_ARCHIVED)
        error = user_validation_errors.ArchivedModelNotMarkedDeletedError(model)
        self.assertEqual(error.stderr, 'ArchivedModelNotMarkedDeletedError in UserQueryModel(id="test"): model is archived but not marked as deleted')