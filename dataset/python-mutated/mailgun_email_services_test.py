"""Tests for the Mailgun API wrapper."""
from __future__ import annotations
import urllib
from core import feconf
from core import utils
from core.platform import models
from core.platform.email import mailgun_email_services
from core.tests import test_utils
from typing import Dict, Tuple
secrets_services = models.Registry.import_secrets_services()
MailgunQueryType = Tuple[str, bytes, Dict[str, str]]

class EmailTests(test_utils.GenericTestBase):
    """Tests for sending emails."""

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        self.swapped_request = lambda *args: args
        self.swap_api_key_secrets_return_none = self.swap_to_always_return(secrets_services, 'get_secret', None)
        self.swap_api_key_secrets_return_secret = self.swap_with_checks(secrets_services, 'get_secret', lambda _: 'key', expected_args=[('MAILGUN_API_KEY',)])

    class Response:
        """Class to mock utils.url_open responses."""

        def __init__(self, url: MailgunQueryType, expected_url: MailgunQueryType) -> None:
            if False:
                i = 10
                return i + 15
            self.url = url
            self.expected_url = expected_url

        def getcode(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            'Gets the status code of this url_open mock.\n\n            Returns:\n                int. 200 to signify status is OK. 500 otherwise.\n            '
            return 200 if self.url == self.expected_url else 500

    def test_send_email_to_mailgun_without_bcc_reply_to_and_recipients(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test for sending HTTP POST request.'
        expected_query_url: MailgunQueryType = ('https://api.mailgun.net/v3/domain/messages', b'from=a%40a.com&subject=Hola+%F0%9F%98%82+-+invitation+to+collaborate&text=plaintext_body+%F0%9F%98%82&html=Hi+abc%2C%3Cbr%3E+%F0%9F%98%82&to=b%40b.com&recipient_variables=%7B%7D', {'Authorization': 'Basic YXBpOmtleQ=='})
        swapped_urlopen = lambda x: self.Response(x, expected_query_url)
        swap_urlopen_context = self.swap(utils, 'url_open', swapped_urlopen)
        swap_request_context = self.swap(urllib.request, 'Request', self.swapped_request)
        swap_domain = self.swap(feconf, 'MAILGUN_DOMAIN_NAME', 'domain')
        with self.swap_api_key_secrets_return_secret, swap_urlopen_context:
            with swap_request_context, swap_domain:
                resp = mailgun_email_services.send_email_to_recipients('a@a.com', ['b@b.com'], 'Hola 😂 - invitation to collaborate', 'plaintext_body 😂', 'Hi abc,<br> 😂')
                self.assertTrue(resp)

    def test_send_email_to_mailgun_with_bcc_and_recipient(self) -> None:
        if False:
            return 10
        expected_query_url = ('https://api.mailgun.net/v3/domain/messages', b'from=a%40a.com&subject=Hola+%F0%9F%98%82+-+invitation+to+collaborate&text=plaintext_body+%F0%9F%98%82&html=Hi+abc%2C%3Cbr%3E+%F0%9F%98%82&to=b%40b.com&bcc=c%40c.com&h%3AReply-To=abc&recipient_variables=%7B%27b%40b.com%27%3A+%7B%27first%27%3A+%27Bob%27%2C+%27id%27%3A+1%7D%7D', {'Authorization': 'Basic YXBpOmtleQ=='})
        swapped_urlopen = lambda x: self.Response(x, expected_query_url)
        swap_urlopen_context = self.swap(utils, 'url_open', swapped_urlopen)
        swap_request_context = self.swap(urllib.request, 'Request', self.swapped_request)
        swap_domain = self.swap(feconf, 'MAILGUN_DOMAIN_NAME', 'domain')
        with self.swap_api_key_secrets_return_secret, swap_urlopen_context:
            with swap_request_context, swap_domain:
                resp = mailgun_email_services.send_email_to_recipients('a@a.com', ['b@b.com'], 'Hola 😂 - invitation to collaborate', 'plaintext_body 😂', 'Hi abc,<br> 😂', bcc=['c@c.com'], reply_to='abc', recipient_variables={'b@b.com': {'first': 'Bob', 'id': 1}})
                self.assertTrue(resp)

    def test_send_email_to_mailgun_with_bcc_and_recipients(self) -> None:
        if False:
            i = 10
            return i + 15
        expected_query_url = ('https://api.mailgun.net/v3/domain/messages', b'from=a%40a.com&subject=Hola+%F0%9F%98%82+-+invitation+to+collaborate&text=plaintext_body+%F0%9F%98%82&html=Hi+abc%2C%3Cbr%3E+%F0%9F%98%82&to=b%40b.com&bcc=%5B%27c%40c.com%27%2C+%27d%40d.com%27%5D&h%3AReply-To=abc&recipient_variables=%7B%27b%40b.com%27%3A+%7B%27first%27%3A+%27Bob%27%2C+%27id%27%3A+1%7D%7D', {'Authorization': 'Basic YXBpOmtleQ=='})
        swapped_urlopen = lambda x: self.Response(x, expected_query_url)
        swap_urlopen_context = self.swap(utils, 'url_open', swapped_urlopen)
        swap_request_context = self.swap(urllib.request, 'Request', self.swapped_request)
        swap_domain = self.swap(feconf, 'MAILGUN_DOMAIN_NAME', 'domain')
        with self.swap_api_key_secrets_return_secret, swap_urlopen_context:
            with swap_request_context, swap_domain:
                resp = mailgun_email_services.send_email_to_recipients('a@a.com', ['b@b.com'], 'Hola 😂 - invitation to collaborate', 'plaintext_body 😂', 'Hi abc,<br> 😂', bcc=['c@c.com', 'd@d.com'], reply_to='abc', recipient_variables={'b@b.com': {'first': 'Bob', 'id': 1}})
                self.assertTrue(resp)

    def test_batch_send_to_mailgun(self) -> None:
        if False:
            print('Hello World!')
        'Test for sending HTTP POST request.'
        expected_query_url: MailgunQueryType = ('https://api.mailgun.net/v3/domain/messages', b'from=a%40a.com&subject=Hola+%F0%9F%98%82+-+invitation+to+collaborate&text=plaintext_body+%F0%9F%98%82&html=Hi+abc%2C%3Cbr%3E+%F0%9F%98%82&to=%5B%27b%40b.com%27%2C+%27c%40c.com%27%2C+%27d%40d.com%27%5D&recipient_variables=%7B%7D', {'Authorization': 'Basic YXBpOmtleQ=='})
        swapped_urlopen = lambda x: self.Response(x, expected_query_url)
        swapped_request = lambda *args: args
        swap_urlopen_context = self.swap(utils, 'url_open', swapped_urlopen)
        swap_request_context = self.swap(urllib.request, 'Request', swapped_request)
        swap_domain = self.swap(feconf, 'MAILGUN_DOMAIN_NAME', 'domain')
        with self.swap_api_key_secrets_return_secret, swap_urlopen_context:
            with swap_request_context, swap_domain:
                resp = mailgun_email_services.send_email_to_recipients('a@a.com', ['b@b.com', 'c@c.com', 'd@d.com'], 'Hola 😂 - invitation to collaborate', 'plaintext_body 😂', 'Hi abc,<br> 😂')
                self.assertTrue(resp)

    def test_mailgun_key_not_set_raises_exception(self) -> None:
        if False:
            return 10
        'Test that exceptions are raised when API key or domain name are\n        unset.\n        '
        mailgun_exception = self.assertRaisesRegex(Exception, 'Mailgun API key is not available.')
        with self.swap_api_key_secrets_return_none, mailgun_exception:
            with self.capture_logging() as logs:
                mailgun_email_services.send_email_to_recipients('a@a.com', ['b@b.com', 'c@c.com', 'd@d.com'], 'Hola 😂 - invitation to collaborate', 'plaintext_body 😂', 'Hi abc,<br> 😂')
                self.assertIn('Cloud Secret Manager is not able to get MAILGUN_API_KEY.', logs)

    def test_mailgun_domain_name_not_set_raises_exception(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        mailgun_exception = self.assertRaisesRegex(Exception, 'Mailgun domain name is not set.')
        with self.swap_api_key_secrets_return_secret, mailgun_exception:
            with self.capture_logging() as logs:
                mailgun_email_services.send_email_to_recipients('a@a.com', ['b@b.com', 'c@c.com', 'd@d.com'], 'Hola 😂 - invitation to collaborate', 'plaintext_body 😂', 'Hi abc,<br> 😂')
                self.assertIn('Cloud Secret Manager is not able to get MAILGUN_API_KEY.', logs)

    def test_invalid_status_code_returns_false(self) -> None:
        if False:
            while True:
                i = 10
        expected_query_url: MailgunQueryType = ('https://api.mailgun.net/v3/domain/messages', b'from=a%40a.com&subject=Hola+%F0%9F%98%82+-+invitation+to+collaborate&text=plaintext_body+%F0%9F%98%82&html=Hi+abc%2C%3Cbr%3E+%F0%9F%98%82&to=%5B%27b%40b.com%27%2C+%27c%40c.com%27%2C+%27d%40d.com%27%5D&recipient_variables=%7B%7D', {'Authorization': 'Basic'})
        swapped_request = lambda *args: args
        swapped_urlopen = lambda x: self.Response(x, expected_query_url)
        swap_urlopen_context = self.swap(utils, 'url_open', swapped_urlopen)
        swap_request_context = self.swap(urllib.request, 'Request', swapped_request)
        swap_domain = self.swap(feconf, 'MAILGUN_DOMAIN_NAME', 'domain')
        with self.swap_api_key_secrets_return_secret, swap_urlopen_context:
            with swap_request_context, swap_domain:
                resp = mailgun_email_services.send_email_to_recipients('a@a.com', ['b@b.com'], 'Hola 😂 - invitation to collaborate', 'plaintext_body 😂', 'Hi abc,<br> 😂', bcc=['c@c.com', 'd@d.com'], reply_to='abc', recipient_variables={'b@b.com': {'first': 'Bob', 'id': 1}})
                self.assertFalse(resp)