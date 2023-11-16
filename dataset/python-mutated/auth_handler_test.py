"""Tests for certbot._internal.auth_handler."""
import datetime
import logging
import sys
import unittest
from unittest import mock
from josepy import b64encode
import pytest
from acme import challenges
from acme import client as acme_client
from acme import errors as acme_errors
from acme import messages
from certbot import achallenges
from certbot import errors
from certbot._internal.display import obj as display_obj
from certbot.plugins import common as plugin_common
from certbot.tests import acme_util
from certbot.tests import util as test_util

class ChallengeFactoryTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        from certbot._internal.auth_handler import AuthHandler
        self.handler = AuthHandler(None, None, mock.Mock(key='mock_key'), [])
        self.authzr = acme_util.gen_authzr(messages.STATUS_PENDING, 'test', acme_util.CHALLENGES, [messages.STATUS_PENDING] * 6)

    def test_all(self):
        if False:
            for i in range(10):
                print('nop')
        achalls = self.handler._challenge_factory(self.authzr, range(0, len(acme_util.CHALLENGES)))
        assert [achall.chall for achall in achalls] == acme_util.CHALLENGES

    def test_one_http(self):
        if False:
            while True:
                i = 10
        achalls = self.handler._challenge_factory(self.authzr, [0])
        assert [achall.chall for achall in achalls] == [acme_util.HTTP01]

    def test_unrecognized(self):
        if False:
            while True:
                i = 10
        authzr = acme_util.gen_authzr(messages.STATUS_PENDING, 'test', [mock.Mock(chall='chall', typ='unrecognized')], [messages.STATUS_PENDING])
        achalls = self.handler._challenge_factory(authzr, [0])
        assert type(achalls[0]) == achallenges.Other

class HandleAuthorizationsTest(unittest.TestCase):
    """handle_authorizations test.

    This tests everything except for all functions under _poll_challenges.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        from certbot._internal.auth_handler import AuthHandler
        self.mock_display = mock.Mock()
        self.mock_config = mock.Mock(debug_challenges=False)
        display_obj.set_display(self.mock_display)
        self.mock_auth = mock.MagicMock(name='Authenticator')
        self.mock_auth.get_chall_pref.return_value = [challenges.HTTP01]
        self.mock_auth.perform.side_effect = gen_auth_resp
        self.mock_account = mock.MagicMock()
        self.mock_net = mock.MagicMock(spec=acme_client.ClientV2)
        self.mock_net.retry_after.side_effect = acme_client.ClientV2.retry_after
        self.handler = AuthHandler(self.mock_auth, self.mock_net, self.mock_account, [])
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        if False:
            while True:
                i = 10
        logging.disable(logging.NOTSET)

    def _test_name1_http_01_1_common(self):
        if False:
            for i in range(10):
                print('nop')
        authzr = gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)
        mock_order = mock.MagicMock(authorizations=[authzr])
        self.mock_net.poll.side_effect = _gen_mock_on_poll(retry=1, wait_value=30)
        with mock.patch('certbot._internal.auth_handler.time') as mock_time:
            authzr = self.handler.handle_authorizations(mock_order, self.mock_config)
            assert self.mock_net.answer_challenge.call_count == 1
            assert self.mock_net.poll.call_count == 2
            assert mock_time.sleep.call_count == 2
            assert mock_time.sleep.call_args_list[1][0][0] <= 30
            assert mock_time.sleep.call_args_list[1][0][0] > 3
            assert self.mock_auth.cleanup.call_count == 1
            assert self.mock_auth.cleanup.call_args[0][0][0].typ == 'http-01'
            assert len(authzr) == 1

    def test_name1_http_01_1_acme_2(self):
        if False:
            return 10
        self._test_name1_http_01_1_common()

    def test_name1_http_01_1_dns_1_acme_2(self):
        if False:
            i = 10
            return i + 15
        self.mock_net.poll.side_effect = _gen_mock_on_poll()
        self.mock_auth.get_chall_pref.return_value.append(challenges.DNS01)
        authzr = gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)
        mock_order = mock.MagicMock(authorizations=[authzr])
        authzr = self.handler.handle_authorizations(mock_order, self.mock_config)
        assert self.mock_net.answer_challenge.call_count == 1
        assert self.mock_net.poll.call_count == 1
        assert self.mock_auth.cleanup.call_count == 1
        cleaned_up_achalls = self.mock_auth.cleanup.call_args[0][0]
        assert len(cleaned_up_achalls) == 1
        assert cleaned_up_achalls[0].typ == 'http-01'
        assert len(authzr) == 1

    def test_name3_http_01_3_common_acme_2(self):
        if False:
            print('Hello World!')
        authzrs = [gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES), gen_dom_authzr(domain='1', challs=acme_util.CHALLENGES), gen_dom_authzr(domain='2', challs=acme_util.CHALLENGES)]
        mock_order = mock.MagicMock(authorizations=authzrs)
        self.mock_net.poll.side_effect = _gen_mock_on_poll()
        authzr = self.handler.handle_authorizations(mock_order, self.mock_config)
        assert self.mock_net.answer_challenge.call_count == 3
        assert self.mock_net.poll.call_count == 3
        assert self.mock_auth.cleanup.call_count == 1
        assert len(authzr) == 3

    def test_debug_challenges(self):
        if False:
            for i in range(10):
                print('nop')
        config = mock.Mock(debug_challenges=True, verbose_count=0)
        authzrs = [gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)]
        mock_order = mock.MagicMock(authorizations=authzrs)
        account_key_thumbprint = b'foobarbaz'
        self.mock_account.key.thumbprint.return_value = account_key_thumbprint
        self.mock_net.poll.side_effect = _gen_mock_on_poll()
        self.handler.handle_authorizations(mock_order, config)
        assert self.mock_net.answer_challenge.call_count == 1
        assert self.mock_display.notification.call_count == 1
        assert 'Pass "-v" for more info' in self.mock_display.notification.call_args[0][0]
        assert f'http://{authzrs[0].body.identifier.value}/.well-known/acme-challenge/' + b64encode(authzrs[0].body.challenges[0].chall.token).decode() not in self.mock_display.notification.call_args[0][0]
        assert b64encode(account_key_thumbprint).decode() not in self.mock_display.notification.call_args[0][0]

    def test_debug_challenges_verbose(self):
        if False:
            return 10
        config = mock.Mock(debug_challenges=True, verbose_count=1)
        authzrs = [gen_dom_authzr(domain='0', challs=[acme_util.HTTP01]), gen_dom_authzr(domain='1', challs=[acme_util.DNS01])]
        mock_order = mock.MagicMock(authorizations=authzrs)
        account_key_thumbprint = b'foobarbaz'
        self.mock_account.key.thumbprint.return_value = account_key_thumbprint
        self.mock_net.poll.side_effect = _gen_mock_on_poll()
        self.mock_auth.get_chall_pref.return_value = [challenges.HTTP01, challenges.DNS01]
        self.handler.handle_authorizations(mock_order, config)
        assert self.mock_net.answer_challenge.call_count == 2
        assert self.mock_display.notification.call_count == 1
        assert 'Pass "-v" for more info' not in self.mock_display.notification.call_args[0][0]
        assert f'http://{authzrs[0].body.identifier.value}/.well-known/acme-challenge/' + b64encode(authzrs[0].body.challenges[0].chall.token).decode() in self.mock_display.notification.call_args[0][0]
        assert b64encode(account_key_thumbprint).decode() in self.mock_display.notification.call_args[0][0]
        assert f'_acme-challenge.{authzrs[1].body.identifier.value}' in self.mock_display.notification.call_args[0][0]
        assert authzrs[1].body.challenges[0].validation(self.mock_account.key) in self.mock_display.notification.call_args[0][0]

    def test_perform_failure(self):
        if False:
            return 10
        authzrs = [gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)]
        mock_order = mock.MagicMock(authorizations=authzrs)
        self.mock_auth.perform.side_effect = errors.AuthorizationError
        with pytest.raises(errors.AuthorizationError):
            self.handler.handle_authorizations(mock_order, self.mock_config)

    def test_max_retries_exceeded(self):
        if False:
            for i in range(10):
                print('nop')
        authzrs = [gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)]
        mock_order = mock.MagicMock(authorizations=authzrs)
        self.mock_net.poll.side_effect = _gen_mock_on_poll(retry=2)
        with pytest.raises(errors.AuthorizationError, match='All authorizations were not finalized by the CA.'):
            self.handler.handle_authorizations(mock_order, self.mock_config, False, 1)

    @mock.patch('certbot._internal.auth_handler.time.sleep')
    def test_deadline_exceeded(self, mock_sleep):
        if False:
            return 10
        authzrs = [gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)]
        mock_order = mock.MagicMock(authorizations=authzrs)
        orig_now = datetime.datetime.now
        state = {'time_slept': 0}

        def mock_sleep_effect(secs):
            if False:
                return 10
            state['time_slept'] += secs
        mock_sleep.side_effect = mock_sleep_effect

        def mock_now_effect():
            if False:
                print('Hello World!')
            return orig_now() + datetime.timedelta(seconds=state['time_slept'])
        interval = datetime.timedelta(minutes=20).seconds
        self.mock_net.poll.side_effect = _gen_mock_on_poll(status=messages.STATUS_PENDING, wait_value=interval)
        with pytest.raises(errors.AuthorizationError, match='All authorizations were not finalized by the CA.'):
            with mock.patch('certbot._internal.auth_handler.datetime.datetime') as mock_dt:
                mock_dt.now.side_effect = mock_now_effect
                self.handler.handle_authorizations(mock_order, self.mock_config, False)
        assert mock_sleep.call_count == 3
        assert mock_sleep.call_args_list[0][0][0] == 1
        assert abs(mock_sleep.call_args_list[1][0][0] - (interval - 1)) <= 1
        assert abs(mock_sleep.call_args_list[2][0][0] - (interval / 2 - 1)) <= 1

    def test_no_domains(self):
        if False:
            print('Hello World!')
        mock_order = mock.MagicMock(authorizations=[])
        with pytest.raises(errors.AuthorizationError):
            self.handler.handle_authorizations(mock_order, self.mock_config)

    def test_preferred_challenge_choice_common_acme_2(self):
        if False:
            for i in range(10):
                print('nop')
        authzrs = [gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)]
        mock_order = mock.MagicMock(authorizations=authzrs)
        self.mock_auth.get_chall_pref.return_value.append(challenges.HTTP01)
        self.handler.pref_challs.extend((challenges.HTTP01.typ, challenges.DNS01.typ))
        self.mock_net.poll.side_effect = _gen_mock_on_poll()
        self.handler.handle_authorizations(mock_order, self.mock_config)
        assert self.mock_auth.cleanup.call_count == 1
        assert self.mock_auth.cleanup.call_args[0][0][0].typ == 'http-01'

    def test_preferred_challenges_not_supported_acme_2(self):
        if False:
            while True:
                i = 10
        authzrs = [gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)]
        mock_order = mock.MagicMock(authorizations=authzrs)
        self.handler.pref_challs.append(challenges.DNS01.typ)
        with pytest.raises(errors.AuthorizationError):
            self.handler.handle_authorizations(mock_order, self.mock_config)

    def test_dns_only_challenge_not_supported(self):
        if False:
            print('Hello World!')
        authzrs = [gen_dom_authzr(domain='0', challs=[acme_util.DNS01])]
        mock_order = mock.MagicMock(authorizations=authzrs)
        with pytest.raises(errors.AuthorizationError):
            self.handler.handle_authorizations(mock_order, self.mock_config)

    def test_perform_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.mock_auth.perform.side_effect = errors.AuthorizationError
        authzr = gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)
        mock_order = mock.MagicMock(authorizations=[authzr])
        with pytest.raises(errors.AuthorizationError):
            self.handler.handle_authorizations(mock_order, self.mock_config)
        assert self.mock_auth.cleanup.call_count == 1
        assert self.mock_auth.cleanup.call_args[0][0][0].typ == 'http-01'

    def test_answer_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.mock_net.answer_challenge.side_effect = errors.AuthorizationError
        authzrs = [gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)]
        mock_order = mock.MagicMock(authorizations=authzrs)
        with pytest.raises(errors.AuthorizationError):
            self.handler.handle_authorizations(mock_order, self.mock_config)
        assert self.mock_auth.cleanup.call_count == 1
        assert self.mock_auth.cleanup.call_args[0][0][0].typ == 'http-01'

    def test_incomplete_authzr_error(self):
        if False:
            for i in range(10):
                print('nop')
        authzrs = [gen_dom_authzr(domain='0', challs=acme_util.CHALLENGES)]
        mock_order = mock.MagicMock(authorizations=authzrs)
        self.mock_net.poll.side_effect = _gen_mock_on_poll(status=messages.STATUS_INVALID)
        with test_util.patch_display_util():
            with pytest.raises(errors.AuthorizationError, match='Some challenges have failed.'):
                self.handler.handle_authorizations(mock_order, self.mock_config, False)
        assert self.mock_auth.cleanup.call_count == 1
        assert self.mock_auth.cleanup.call_args[0][0][0].typ == 'http-01'

    def test_best_effort(self):
        if False:
            print('Hello World!')

        def _conditional_mock_on_poll(authzr):
            if False:
                return 10
            'This mock will invalidate one authzr, and invalidate the other one'
            valid_mock = _gen_mock_on_poll(messages.STATUS_VALID)
            invalid_mock = _gen_mock_on_poll(messages.STATUS_INVALID)
            if authzr.body.identifier.value == 'will-be-invalid':
                return invalid_mock(authzr)
            return valid_mock(authzr)
        authzrs = [gen_dom_authzr(domain='will-be-valid', challs=acme_util.CHALLENGES), gen_dom_authzr(domain='will-be-invalid', challs=acme_util.CHALLENGES)]
        self.mock_net.poll.side_effect = _conditional_mock_on_poll
        mock_order = mock.MagicMock(authorizations=authzrs)
        with mock.patch('certbot._internal.auth_handler.AuthHandler._report_failed_authzrs') as mock_report:
            valid_authzr = self.handler.handle_authorizations(mock_order, self.mock_config, True)
        assert len(valid_authzr) == 1
        assert mock_report.call_count == 1
        self.mock_net.poll.side_effect = _gen_mock_on_poll(status=messages.STATUS_INVALID)
        with test_util.patch_display_util():
            with pytest.raises(errors.AuthorizationError, match='All challenges have failed.'):
                self.handler.handle_authorizations(mock_order, self.mock_config, True)

    def test_validated_challenge_not_rerun(self):
        if False:
            i = 10
            return i + 15
        authzr = acme_util.gen_authzr(messages.STATUS_PENDING, '0', [acme_util.DNS01], [messages.STATUS_PENDING])
        mock_order = mock.MagicMock(authorizations=[authzr])
        with pytest.raises(errors.AuthorizationError):
            self.handler.handle_authorizations(mock_order, self.mock_config)
        authzr = acme_util.gen_authzr(messages.STATUS_VALID, '0', [acme_util.DNS01], [messages.STATUS_VALID])
        mock_order = mock.MagicMock(authorizations=[authzr])
        self.handler.handle_authorizations(mock_order, self.mock_config)

    def test_valid_authzrs_deactivated(self):
        if False:
            return 10
        'When we deactivate valid authzrs in an orderr, we expect them to become deactivated\n        and to receive a list of deactivated authzrs in return.'

        def _mock_deactivate(authzr):
            if False:
                print('Hello World!')
            if authzr.body.status == messages.STATUS_VALID:
                if authzr.body.identifier.value == 'is_valid_but_will_fail':
                    raise acme_errors.Error('Mock deactivation ACME error')
                authzb = authzr.body.update(status=messages.STATUS_DEACTIVATED)
                authzr = messages.AuthorizationResource(body=authzb)
            else:
                raise errors.Error("Can't deactivate non-valid authz")
            return authzr
        to_deactivate = [('is_valid', messages.STATUS_VALID), ('is_pending', messages.STATUS_PENDING), ('is_valid_but_will_fail', messages.STATUS_VALID)]
        to_deactivate = [acme_util.gen_authzr(a[1], a[0], [acme_util.HTTP01], [a[1]]) for a in to_deactivate]
        orderr = mock.MagicMock(authorizations=to_deactivate)
        self.mock_net.deactivate_authorization.side_effect = _mock_deactivate
        (authzrs, failed) = self.handler.deactivate_valid_authorizations(orderr)
        assert self.mock_net.deactivate_authorization.call_count == 2
        assert len(authzrs) == 1
        assert len(failed) == 1
        assert authzrs[0].body.identifier.value == 'is_valid'
        assert authzrs[0].body.status == messages.STATUS_DEACTIVATED
        assert failed[0].body.identifier.value == 'is_valid_but_will_fail'
        assert failed[0].body.status == messages.STATUS_VALID

def _gen_mock_on_poll(status=messages.STATUS_VALID, retry=0, wait_value=1):
    if False:
        return 10
    state = {'count': retry}

    def _mock(authzr):
        if False:
            while True:
                i = 10
        state['count'] = state['count'] - 1
        effective_status = status if state['count'] < 0 else messages.STATUS_PENDING
        updated_azr = acme_util.gen_authzr(effective_status, authzr.body.identifier.value, [challb.chall for challb in authzr.body.challenges], [effective_status] * len(authzr.body.challenges))
        return (updated_azr, mock.MagicMock(headers={'Retry-After': str(wait_value)}))
    return _mock

class ChallbToAchallTest(unittest.TestCase):
    """Tests for certbot._internal.auth_handler.challb_to_achall."""

    def _call(self, challb):
        if False:
            while True:
                i = 10
        from certbot._internal.auth_handler import challb_to_achall
        return challb_to_achall(challb, 'account_key', 'domain')

    def test_it(self):
        if False:
            print('Hello World!')
        assert self._call(acme_util.HTTP01_P) == achallenges.KeyAuthorizationAnnotatedChallenge(challb=acme_util.HTTP01_P, account_key='account_key', domain='domain')

class GenChallengePathTest(unittest.TestCase):
    """Tests for certbot._internal.auth_handler.gen_challenge_path.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        logging.disable(logging.FATAL)

    def tearDown(self):
        if False:
            print('Hello World!')
        logging.disable(logging.NOTSET)

    @classmethod
    def _call(cls, challbs, preferences):
        if False:
            print('Hello World!')
        from certbot._internal.auth_handler import gen_challenge_path
        return gen_challenge_path(challbs, preferences)

    def test_common_case(self):
        if False:
            while True:
                i = 10
        'Given DNS01 and HTTP01 with appropriate combos.'
        challbs = (acme_util.DNS01_P, acme_util.HTTP01_P)
        prefs = [challenges.DNS01, challenges.HTTP01]
        assert self._call(challbs, prefs) == (0,)
        assert self._call(challbs[::-1], prefs) == (1,)

    def test_not_supported(self):
        if False:
            for i in range(10):
                print('nop')
        challbs = (acme_util.DNS01_P,)
        prefs = [challenges.HTTP01]
        with pytest.raises(errors.AuthorizationError):
            self._call(challbs, prefs)

class ReportFailedAuthzrsTest(unittest.TestCase):
    """Tests for certbot._internal.auth_handler.AuthHandler._report_failed_authzrs."""

    def setUp(self):
        if False:
            while True:
                i = 10
        from certbot._internal.auth_handler import AuthHandler
        self.mock_auth = mock.MagicMock(spec=plugin_common.Plugin, name='buzz')
        self.mock_auth.name = 'buzz'
        self.mock_auth.auth_hint.return_value = 'the buzz hint'
        self.handler = AuthHandler(self.mock_auth, mock.MagicMock(), mock.MagicMock(), [])
        kwargs = {'chall': acme_util.HTTP01, 'uri': 'uri', 'status': messages.STATUS_INVALID, 'error': messages.Error.with_code('tls', detail='detail')}
        assert kwargs['error'].description is not None
        http_01 = messages.ChallengeBody(**kwargs)
        kwargs['chall'] = acme_util.HTTP01
        http_01 = messages.ChallengeBody(**kwargs)
        self.authzr1 = mock.MagicMock()
        self.authzr1.body.identifier.value = 'example.com'
        self.authzr1.body.challenges = [http_01, http_01]
        kwargs['error'] = messages.Error.with_code('dnssec', detail='detail')
        http_01_diff = messages.ChallengeBody(**kwargs)
        self.authzr2 = mock.MagicMock()
        self.authzr2.body.identifier.value = 'foo.bar'
        self.authzr2.body.challenges = [http_01_diff]

    @mock.patch('certbot._internal.auth_handler.display_util.notify')
    def test_same_error_and_domain(self, mock_notify):
        if False:
            print('Hello World!')
        self.handler._report_failed_authzrs([self.authzr1])
        mock_notify.assert_called_with('\nCertbot failed to authenticate some domains (authenticator: buzz). The Certificate Authority reported these problems:\n  Domain: example.com\n  Type:   tls\n  Detail: detail\n\n  Domain: example.com\n  Type:   tls\n  Detail: detail\n\nHint: the buzz hint\n')

    @mock.patch('certbot._internal.auth_handler.display_util.notify')
    def test_different_errors_and_domains(self, mock_notify):
        if False:
            while True:
                i = 10
        self.mock_auth.name = 'quux'
        self.mock_auth.auth_hint.return_value = 'quuuuuux'
        self.handler._report_failed_authzrs([self.authzr1, self.authzr2])
        mock_notify.assert_called_with('\nCertbot failed to authenticate some domains (authenticator: quux). The Certificate Authority reported these problems:\n  Domain: foo.bar\n  Type:   dnssec\n  Detail: detail\n\n  Domain: example.com\n  Type:   tls\n  Detail: detail\n\n  Domain: example.com\n  Type:   tls\n  Detail: detail\n\nHint: quuuuuux\n')

    @mock.patch('certbot._internal.auth_handler.display_util.notify')
    def test_non_subclassed_authenticator(self, mock_notify):
        if False:
            return 10
        "If authenticator not derived from common.Plugin, we shouldn't call .auth_hint"
        from certbot._internal.auth_handler import AuthHandler
        self.mock_auth = mock.MagicMock(name='quuz')
        self.mock_auth.name = 'quuz'
        self.mock_auth.auth_hint.side_effect = Exception
        self.handler = AuthHandler(self.mock_auth, mock.MagicMock(), mock.MagicMock(), [])
        self.handler._report_failed_authzrs([self.authzr1])
        assert mock_notify.call_count == 1

def gen_auth_resp(chall_list):
    if False:
        print('Hello World!')
    'Generate a dummy authorization response.'
    return ['%s%s' % (chall.__class__.__name__, chall.domain) for chall in chall_list]

def gen_dom_authzr(domain, challs):
    if False:
        i = 10
        return i + 15
    'Generates new authzr for domains.'
    return acme_util.gen_authzr(messages.STATUS_PENDING, domain, challs, [messages.STATUS_PENDING] * len(challs))
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))