"""Tests for the domain objects relating to the user queries."""
from __future__ import annotations
import datetime
from core import feconf
from core import utils
from core.constants import constants
from core.domain import user_query_domain
from core.tests import test_utils

class UserQueryParamsAttributeTests(test_utils.GenericTestBase):
    """Test for ensuring matching values for UserQueryParams attributes between
    predefined and dynamically fetched fields from assets/constants.ts
    """

    def test_user_query_params_attributes_against_dynamic_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check to see if the list of attributes of UserQueryParams\n        is similar to the one we get during runtime from assets/constants.ts.\n        '
        attribute_names_predefined = list(user_query_domain.UserQueryParams._fields)
        attribute_names = [predicate['backend_attr'] for predicate in constants.EMAIL_DASHBOARD_PREDICATE_DEFINITION]
        attribute_names_predefined.sort()
        attribute_names.sort()
        self.assertEqual(attribute_names_predefined, attribute_names)

class UserQueryTests(test_utils.GenericTestBase):
    """Test for the UserQuery."""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.signup(self.CURRICULUM_ADMIN_EMAIL, self.CURRICULUM_ADMIN_USERNAME)
        self.user_id = self.get_user_id_from_email(self.CURRICULUM_ADMIN_EMAIL)
        self.user_query_params = user_query_domain.UserQueryParams(inactive_in_last_n_days=20)
        self.user_query = user_query_domain.UserQuery(query_id='user_query_id', query_params=self.user_query_params, submitter_id=self.user_id, query_status=feconf.USER_QUERY_STATUS_PROCESSING, user_ids=[], sent_email_model_id=None, created_on=datetime.datetime.utcnow())
        self.user_query.validate()

    def test_validate_query_with_invalid_user_id_submitter_id_raises(self) -> None:
        if False:
            print('Hello World!')
        self.user_query.submitter_id = 'aaabbc'
        with self.assertRaisesRegex(utils.ValidationError, 'Expected submitter ID to be a valid user ID'):
            self.user_query.validate()

    def test_validate_query_with_invalid_status_raises(self) -> None:
        if False:
            while True:
                i = 10
        self.user_query.status = 'a'
        with self.assertRaisesRegex(utils.ValidationError, 'Invalid status: a'):
            self.user_query.validate()

    def test_validate_query_with_non_user_id_values_in_user_ids_raises(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_query.user_ids = ['aaa']
        with self.assertRaisesRegex(utils.ValidationError, 'Expected user ID in user_ids to be a valid user ID'):
            self.user_query.validate()

    def test_create_default_returns_correct_user_query(self) -> None:
        if False:
            return 10
        default_user_query = user_query_domain.UserQuery.create_default('id', self.user_query_params, self.user_id)
        self.assertEqual(default_user_query.params, self.user_query_params)
        self.assertEqual(default_user_query.submitter_id, self.user_id)
        self.assertEqual(default_user_query.status, feconf.USER_QUERY_STATUS_PROCESSING)
        self.assertEqual(default_user_query.user_ids, [])

    def test_archive_returns_correct_dict(self) -> None:
        if False:
            return 10
        self.user_query.archive(sent_email_model_id='sent_email_model_id')
        self.assertEqual(self.user_query.sent_email_model_id, 'sent_email_model_id')
        self.assertEqual(self.user_query.status, feconf.USER_QUERY_STATUS_ARCHIVED)
        self.assertTrue(self.user_query.deleted)