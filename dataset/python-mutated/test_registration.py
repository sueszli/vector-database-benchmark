from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import UserTypes
from synapse.api.errors import ThreepidValidationError
from synapse.server import HomeServer
from synapse.types import JsonDict, UserID, UserInfo
from synapse.util import Clock
from tests.unittest import HomeserverTestCase, override_config

class RegistrationStoreTestCase(HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.store = hs.get_datastores().main
        self.user_id = '@my-user:test'
        self.tokens = ['AbCdEfGhIjKlMnOpQrStUvWxYz', 'BcDeFgHiJkLmNoPqRsTuVwXyZa']
        self.pwhash = '{xx1}123456789'
        self.device_id = 'akgjhdjklgshg'

    def test_register(self) -> None:
        if False:
            i = 10
            return i + 15
        self.get_success(self.store.register_user(self.user_id, self.pwhash))
        self.assertEqual(UserInfo(user_id=UserID.from_string(self.user_id), is_admin=False, is_guest=False, consent_server_notice_sent=None, consent_ts=None, consent_version=None, appservice_id=None, creation_ts=0, user_type=None, is_deactivated=False, locked=False, is_shadow_banned=False, approved=True), self.get_success(self.store.get_user_by_id(self.user_id)))

    def test_consent(self) -> None:
        if False:
            return 10
        self.get_success(self.store.register_user(self.user_id, self.pwhash))
        before_consent = self.clock.time_msec()
        self.reactor.advance(5)
        self.get_success(self.store.user_set_consent_version(self.user_id, '1'))
        self.reactor.advance(5)
        user = self.get_success(self.store.get_user_by_id(self.user_id))
        assert user
        self.assertEqual(user.consent_version, '1')
        self.assertIsNotNone(user.consent_ts)
        assert user.consent_ts is not None
        self.assertGreater(user.consent_ts, before_consent)
        self.assertLess(user.consent_ts, self.clock.time_msec())

    def test_add_tokens(self) -> None:
        if False:
            while True:
                i = 10
        self.get_success(self.store.register_user(self.user_id, self.pwhash))
        self.get_success(self.store.add_access_token_to_user(self.user_id, self.tokens[1], self.device_id, valid_until_ms=None))
        result = self.get_success(self.store.get_user_by_access_token(self.tokens[1]))
        assert result
        self.assertEqual(result.user_id, self.user_id)
        self.assertEqual(result.device_id, self.device_id)
        self.assertIsNotNone(result.token_id)

    def test_user_delete_access_tokens(self) -> None:
        if False:
            while True:
                i = 10
        self.get_success(self.store.register_user(self.user_id, self.pwhash))
        self.get_success(self.store.add_access_token_to_user(self.user_id, self.tokens[0], device_id=None, valid_until_ms=None))
        self.get_success(self.store.add_access_token_to_user(self.user_id, self.tokens[1], self.device_id, valid_until_ms=None))
        self.get_success(self.store.user_delete_access_tokens(self.user_id, device_id=self.device_id))
        user = self.get_success(self.store.get_user_by_access_token(self.tokens[1]))
        self.assertIsNone(user, 'access token was not deleted by device_id')
        user = self.get_success(self.store.get_user_by_access_token(self.tokens[0]))
        assert user
        self.assertEqual(self.user_id, user.user_id)
        self.get_success(self.store.user_delete_access_tokens(self.user_id))
        user = self.get_success(self.store.get_user_by_access_token(self.tokens[0]))
        self.assertIsNone(user, 'access token was not deleted without device_id')

    def test_is_support_user(self) -> None:
        if False:
            i = 10
            return i + 15
        TEST_USER = '@test:test'
        SUPPORT_USER = '@support:test'
        res = self.get_success(self.store.is_support_user(None))
        self.assertFalse(res)
        self.get_success(self.store.register_user(user_id=TEST_USER, password_hash=None))
        res = self.get_success(self.store.is_support_user(TEST_USER))
        self.assertFalse(res)
        self.get_success(self.store.register_user(user_id=SUPPORT_USER, password_hash=None, user_type=UserTypes.SUPPORT))
        res = self.get_success(self.store.is_support_user(SUPPORT_USER))
        self.assertTrue(res)

    def test_3pid_inhibit_invalid_validation_session_error(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that enabling the configuration option to inhibit 3PID errors on\n        /requestToken also inhibits validation errors caused by an unknown session ID.\n        '
        e = self.get_failure(self.store.validate_threepid_session('fake_sid', 'fake_client_secret', 'fake_token', 0), ThreepidValidationError)
        self.assertEqual(e.value.msg, 'Unknown session_id', e)
        self.store._ignore_unknown_session_error = True
        e = self.get_failure(self.store.validate_threepid_session('fake_sid', 'fake_client_secret', 'fake_token', 0), ThreepidValidationError)
        self.assertEqual(e.value.msg, 'Validation token not found or has expired', e)

class ApprovalRequiredRegistrationTestCase(HomeserverTestCase):

    def default_config(self) -> JsonDict:
        if False:
            return 10
        config = super().default_config()
        msc3866_config = config.get('experimental_features', {}).get('msc3866')
        if msc3866_config is not None:
            return config
        config['experimental_features'] = {'msc3866': {'enabled': True, 'require_approval_for_new_accounts': True}}
        return config

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.store = hs.get_datastores().main
        self.user_id = '@my-user:test'
        self.pwhash = '{xx1}123456789'

    @override_config({'experimental_features': {'msc3866': {'enabled': True, 'require_approval_for_new_accounts': False}}})
    def test_approval_not_required(self) -> None:
        if False:
            print('Hello World!')
        "Tests that if we don't require approval for new accounts, newly created\n        accounts are automatically marked as approved.\n        "
        self.get_success(self.store.register_user(self.user_id, self.pwhash))
        user = self.get_success(self.store.get_user_by_id(self.user_id))
        assert user is not None
        self.assertTrue(user.approved)
        approved = self.get_success(self.store.is_user_approved(self.user_id))
        self.assertTrue(approved)

    def test_approval_required(self) -> None:
        if False:
            return 10
        'Tests that if we require approval for new accounts, newly created accounts\n        are not automatically marked as approved.\n        '
        self.get_success(self.store.register_user(self.user_id, self.pwhash))
        user = self.get_success(self.store.get_user_by_id(self.user_id))
        assert user is not None
        self.assertFalse(user.approved)
        approved = self.get_success(self.store.is_user_approved(self.user_id))
        self.assertFalse(approved)

    def test_override(self) -> None:
        if False:
            print('Hello World!')
        "Tests that if we require approval for new accounts, but we explicitly say the\n        new user should be considered approved, they're marked as approved.\n        "
        self.get_success(self.store.register_user(self.user_id, self.pwhash, approved=True))
        user = self.get_success(self.store.get_user_by_id(self.user_id))
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.approved, 1)
        approved = self.get_success(self.store.is_user_approved(self.user_id))
        self.assertTrue(approved)

    def test_approve_user(self) -> None:
        if False:
            while True:
                i = 10
        'Tests that approving the user updates their approval status.'
        self.get_success(self.store.register_user(self.user_id, self.pwhash))
        approved = self.get_success(self.store.is_user_approved(self.user_id))
        self.assertFalse(approved)
        self.get_success(self.store.update_user_approval_status(UserID.from_string(self.user_id), True))
        approved = self.get_success(self.store.is_user_approved(self.user_id))
        self.assertTrue(approved)