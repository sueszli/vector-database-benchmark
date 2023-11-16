from typing import List, Optional
from unittest.mock import Mock, patch
from synapse._scripts.register_new_matrix_user import request_registration
from synapse.types import JsonDict
from tests.unittest import TestCase

class RegisterTestCase(TestCase):

    def test_success(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The script will fetch a nonce, and then generate a MAC with it, and then\n        post that MAC.\n        '

        def get(url: str, verify: Optional[bool]=None) -> Mock:
            if False:
                while True:
                    i = 10
            r = Mock()
            r.status_code = 200
            r.json = lambda : {'nonce': 'a'}
            return r

        def post(url: str, json: Optional[JsonDict]=None, verify: Optional[bool]=None) -> Mock:
            if False:
                return 10
            assert json is not None
            self.assertEqual(json['username'], 'user')
            self.assertEqual(json['password'], 'pass')
            self.assertEqual(json['nonce'], 'a')
            self.assertEqual(len(json['mac']), 40)
            r = Mock()
            r.status_code = 200
            return r
        requests = Mock()
        requests.get = get
        requests.post = post
        out: List[str] = []
        err_code: List[int] = []
        with patch('synapse._scripts.register_new_matrix_user.requests', requests):
            request_registration('user', 'pass', 'matrix.org', 'shared', admin=False, _print=out.append, exit=err_code.append)
        self.assertIn('Success!', out)
        self.assertEqual(err_code, [])

    def test_failure_nonce(self) -> None:
        if False:
            while True:
                i = 10
        '\n        If the script fails to fetch a nonce, it throws an error and quits.\n        '

        def get(url: str, verify: Optional[bool]=None) -> Mock:
            if False:
                while True:
                    i = 10
            r = Mock()
            r.status_code = 404
            r.reason = 'Not Found'
            r.json = lambda : {'not': 'error'}
            return r
        requests = Mock()
        requests.get = get
        out: List[str] = []
        err_code: List[int] = []
        with patch('synapse._scripts.register_new_matrix_user.requests', requests):
            request_registration('user', 'pass', 'matrix.org', 'shared', admin=False, _print=out.append, exit=err_code.append)
        self.assertEqual(err_code, [1])
        self.assertIn('ERROR! Received 404 Not Found', out)
        self.assertNotIn('Success!', out)

    def test_failure_post(self) -> None:
        if False:
            return 10
        '\n        The script will fetch a nonce, and then if the final POST fails, will\n        report an error and quit.\n        '

        def get(url: str, verify: Optional[bool]=None) -> Mock:
            if False:
                i = 10
                return i + 15
            r = Mock()
            r.status_code = 200
            r.json = lambda : {'nonce': 'a'}
            return r

        def post(url: str, json: Optional[JsonDict]=None, verify: Optional[bool]=None) -> Mock:
            if False:
                i = 10
                return i + 15
            assert json is not None
            self.assertEqual(json['username'], 'user')
            self.assertEqual(json['password'], 'pass')
            self.assertEqual(json['nonce'], 'a')
            self.assertEqual(len(json['mac']), 40)
            r = Mock()
            r.status_code = 500
            r.reason = 'Broken'
            return r
        requests = Mock()
        requests.get = get
        requests.post = post
        out: List[str] = []
        err_code: List[int] = []
        with patch('synapse._scripts.register_new_matrix_user.requests', requests):
            request_registration('user', 'pass', 'matrix.org', 'shared', admin=False, _print=out.append, exit=err_code.append)
        self.assertEqual(err_code, [1])
        self.assertIn('ERROR! Received 500 Broken', out)
        self.assertNotIn('Success!', out)