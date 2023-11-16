from django.http import HttpRequest
from django.template.context_processors import csrf
from django.test import SimpleTestCase
from .tests import CsrfFunctionTestMixin

class TestContextProcessor(CsrfFunctionTestMixin, SimpleTestCase):

    def test_force_token_to_string(self):
        if False:
            while True:
                i = 10
        request = HttpRequest()
        test_secret = 32 * 'a'
        request.META['CSRF_COOKIE'] = test_secret
        token = csrf(request).get('csrf_token')
        self.assertMaskedSecretCorrect(token, test_secret)