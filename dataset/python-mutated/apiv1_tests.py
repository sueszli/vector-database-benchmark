import contextlib
import unittest
import json
from mock import patch, MagicMock
from r2.lib import signing
from r2.tests import RedditControllerTestCase
from r2.lib.validator import VThrottledLogin, VUname
from common import LoginRegBase

class APIV1LoginTests(LoginRegBase, RedditControllerTestCase):
    CONTROLLER = 'apiv1login'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(APIV1LoginTests, self).setUp()
        self.device_id = 'dead-beef'

    def make_ua_signature(self, platform='test', version=1):
        if False:
            print('Hello World!')
        payload = 'User-Agent:{}|Client-Vendor-ID:{}'.format(self.user_agent, self.device_id)
        return self.sign(payload, platform, version)

    def sign(self, payload, platform='test', version=1):
        if False:
            for i in range(10):
                print('nop')
        return signing.sign_v1_message(payload, platform, version)

    def additional_headers(self, headers, body):
        if False:
            return 10
        return {signing.SIGNATURE_UA_HEADER: self.make_ua_signature(), signing.SIGNATURE_BODY_HEADER: self.sign('Body:' + body)}

    def assert_success(self, res):
        if False:
            i = 10
            return i + 15
        self.assertEqual(res.status, 200)
        body = res.body
        body = json.loads(body)
        self.assertTrue('json' in body)
        errors = body['json'].get('errors')
        self.assertEqual(len(errors), 0)
        data = body['json'].get('data')
        self.assertTrue(bool(data))
        self.assertTrue('modhash' in data)
        self.assertTrue('cookie' in data)

    def assert_failure(self, res, code=None):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(res.status, 200)
        body = res.body
        body = json.loads(body)
        self.assertTrue('json' in body)
        errors = body['json'].get('errors')
        self.assertTrue(code in [x[0] for x in errors])
        data = body['json'].get('data')
        self.assertFalse(bool(data))

    def assert_403_response(self, res, calling):
        if False:
            print('Hello World!')
        self.assertEqual(res.status, 403)
        self.simple_event.assert_any_call(calling)
        self.assert_headers(res, 'content-type', 'application/json; charset=UTF-8')

    def test_nosigning_login(self):
        if False:
            while True:
                i = 10
        res = self.do_login(headers={signing.SIGNATURE_UA_HEADER: None, signing.SIGNATURE_BODY_HEADER: None}, expect_errors=True)
        self.assert_403_response(res, 'signing.ua.invalid.invalid_format')

    def test_no_body_signing_login(self):
        if False:
            for i in range(10):
                print('nop')
        res = self.do_login(headers={signing.SIGNATURE_BODY_HEADER: None}, expect_errors=True)
        self.assert_403_response(res, 'signing.body.invalid.invalid_format')

    def test_nosigning_register(self):
        if False:
            for i in range(10):
                print('nop')
        res = self.do_register(headers={signing.SIGNATURE_UA_HEADER: None, signing.SIGNATURE_BODY_HEADER: None}, expect_errors=True)
        self.assert_403_response(res, 'signing.ua.invalid.invalid_format')

    def test_no_body_signing_register(self):
        if False:
            while True:
                i = 10
        res = self.do_login(headers={signing.SIGNATURE_BODY_HEADER: None}, expect_errors=True)
        self.assert_403_response(res, 'signing.body.invalid.invalid_format')

    @unittest.skip('registration captcha is unfinished')
    def test_captcha_blocking(self):
        if False:
            for i in range(10):
                print('nop')
        with contextlib.nested(self.mock_register(), self.failed_captcha()):
            res = self.do_register()
            self.assert_success(res)