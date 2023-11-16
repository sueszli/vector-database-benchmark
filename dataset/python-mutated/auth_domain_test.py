"""Tests for core.domain.auth_domain"""
from __future__ import annotations
from core import utils
from core.domain import auth_domain
from core.domain import auth_services
from core.platform import models
from core.tests import test_utils
MYPY = False
if MYPY:
    from mypy_imports import auth_models
    from mypy_imports import user_models
(auth_models, user_models) = models.Registry.import_models([models.Names.AUTH, models.Names.USER])

class AuthIdUserIdPairTests(test_utils.TestBase):

    def test_unpacking(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (auth_id, user_id) = auth_domain.AuthIdUserIdPair('aid', 'uid')
        self.assertEqual(auth_id, 'aid')
        self.assertEqual(user_id, 'uid')

class AuthClaimsTests(test_utils.TestBase):

    def test_rejects_empty_auth_id(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(Exception, 'auth_id must not be empty'):
            auth_domain.AuthClaims('', None, False)
        with self.assertRaisesRegex(Exception, 'auth_id must not be empty'):
            auth_domain.AuthClaims('', None, True)

    def test_attributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        auth = auth_domain.AuthClaims('sub', 'email@test.com', True)
        self.assertEqual(auth.auth_id, 'sub')
        self.assertEqual(auth.email, 'email@test.com')
        self.assertTrue(auth.role_is_super_admin)

    def test_repr(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(repr(auth_domain.AuthClaims('sub', 'email@test.com', False)), 'AuthClaims(auth_id=%r, email=%r, role_is_super_admin=%r)' % ('sub', 'email@test.com', False))
        self.assertEqual(repr(auth_domain.AuthClaims('tub', None, True)), 'AuthClaims(auth_id=%r, email=%r, role_is_super_admin=%r)' % ('tub', None, True))

    def test_comparison(self) -> None:
        if False:
            while True:
                i = 10
        auth = auth_domain.AuthClaims('sub', 'email@test.com', False)
        self.assertEqual(auth, auth_domain.AuthClaims('sub', 'email@test.com', False))
        self.assertNotEqual(auth, auth_domain.AuthClaims('tub', None, False))

    def test_hash(self) -> None:
        if False:
            while True:
                i = 10
        a = auth_domain.AuthClaims('a', 'a@a.com', False)
        b = auth_domain.AuthClaims('b', 'b@b.com', True)
        auth_set = set([a, b])
        self.assertIn(auth_domain.AuthClaims('a', 'a@a.com', False), auth_set)
        self.assertIn(auth_domain.AuthClaims('b', 'b@b.com', True), auth_set)
        self.assertNotIn(auth_domain.AuthClaims('a', 'a@a.com', True), auth_set)
        self.assertNotIn(auth_domain.AuthClaims('c', 'c@c.com', True), auth_set)

class UserAuthDetailsTests(test_utils.GenericTestBase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.user_auth_details_model = auth_models.UserAuthDetailsModel.get(self.owner_id)
        self.user_auth_details = auth_services.get_user_auth_details_from_model(self.user_auth_details_model)
        self.auth_id = self.get_auth_id_from_email(self.OWNER_EMAIL)
        self.user_auth_details.validate()

    def test_repr(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(repr(auth_domain.UserAuthDetails('uid', 'g_auth_id', 'f_auth_id', 'pid', True)), 'UserAuthDetails(user_id=%r, gae_id=%r, firebase_auth_id=%r, parent_user_id=%r, deleted=%r)' % ('uid', 'g_auth_id', 'f_auth_id', 'pid', True))

    def test_to_dict(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(auth_domain.UserAuthDetails('uid', 'g_auth_id', 'f_auth_id', 'pid', True).to_dict(), {'gae_id': 'g_auth_id', 'firebase_auth_id': 'f_auth_id', 'parent_user_id': 'pid', 'deleted': True})

    def test_validate_non_str_user_id(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_auth_details.user_id = 123
        with self.assertRaisesRegex(utils.ValidationError, 'user_id must be a string'):
            self.user_auth_details.validate()

    def test_validate_user_id_enforces_all_lowercase_letters(self) -> None:
        if False:
            i = 10
            return i + 15
        self.user_auth_details.user_id = 'uid_%s%s' % ('a' * 31, 'A')
        with self.assertRaisesRegex(utils.ValidationError, 'wrong format'):
            self.user_auth_details.validate()

    def test_validate_user_id_enforces_length_to_be_at_least_36(self) -> None:
        if False:
            while True:
                i = 10
        self.user_auth_details.user_id = 'uid_%s' % ('a' * 31)
        with self.assertRaisesRegex(utils.ValidationError, 'wrong format'):
            self.user_auth_details.validate()

    def test_validate_user_id_enforces_uid_prefix(self) -> None:
        if False:
            return 10
        self.user_auth_details.user_id = 'a' * 36
        with self.assertRaisesRegex(utils.ValidationError, 'wrong format'):
            self.user_auth_details.validate()

    def test_validate_empty_user_id(self) -> None:
        if False:
            return 10
        self.user_auth_details.user_id = ''
        with self.assertRaisesRegex(utils.ValidationError, 'No user_id specified'):
            self.user_auth_details.validate()

    def test_validate_parent_user_id_enforces_all_lowercase_letters(self) -> None:
        if False:
            i = 10
            return i + 15
        self.user_auth_details.parent_user_id = 'uid_%s%s' % ('a' * 31, 'A')
        with self.assertRaisesRegex(utils.ValidationError, 'wrong format'):
            self.user_auth_details.validate()

    def test_validate_parent_user_id_enforces_length_to_be_at_least_36(self) -> None:
        if False:
            i = 10
            return i + 15
        self.user_auth_details.parent_user_id = 'uid_%s' % ('a' * 31)
        with self.assertRaisesRegex(utils.ValidationError, 'wrong format'):
            self.user_auth_details.validate()

    def test_validate_parent_user_id_enforces_uid_prefix(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_auth_details.parent_user_id = 'a' * 36
        with self.assertRaisesRegex(utils.ValidationError, 'wrong format'):
            self.user_auth_details.validate()

    def test_validate_non_str_gae_id(self) -> None:
        if False:
            print('Hello World!')
        self.user_auth_details.gae_id = 123
        with self.assertRaisesRegex(utils.ValidationError, 'gae_id must be a string'):
            self.user_auth_details.validate()

    def test_validate_non_str_firebase_auth_id(self) -> None:
        if False:
            i = 10
            return i + 15
        self.user_auth_details.firebase_auth_id = 123
        with self.assertRaisesRegex(utils.ValidationError, 'firebase_auth_id must be a string'):
            self.user_auth_details.validate()

    def test_parent_user_id_and_gae_id_together_raises_error(self) -> None:
        if False:
            print('Hello World!')
        self.user_auth_details.parent_user_id = user_models.UserSettingsModel.get_new_id('')
        self.user_auth_details.gae_id = self.auth_id
        self.user_auth_details.firebase_auth_id = None
        with self.assertRaisesRegex(utils.ValidationError, 'parent_user_id must not be set for a full user'):
            self.user_auth_details.validate()

    def test_parent_user_id_and_firebase_auth_id_together_raises_error(self) -> None:
        if False:
            return 10
        self.user_auth_details.parent_user_id = user_models.UserSettingsModel.get_new_id('')
        self.user_auth_details.gae_id = None
        self.user_auth_details.firebase_auth_id = self.auth_id
        with self.assertRaisesRegex(utils.ValidationError, 'parent_user_id must not be set for a full user'):
            self.user_auth_details.validate()

    def test_both_parent_user_id_and_auth_id_none_raises_error(self) -> None:
        if False:
            return 10
        self.user_auth_details.parent_user_id = None
        self.user_auth_details.gae_id = None
        self.user_auth_details.firebase_auth_id = None
        with self.assertRaisesRegex(utils.ValidationError, 'parent_user_id must be set for a profile user'):
            self.user_auth_details.validate()