import os
from unittest import mock
from django.core.checks.async_checks import E001, check_async_unsafe
from django.test import SimpleTestCase

class AsyncCheckTests(SimpleTestCase):

    @mock.patch.dict(os.environ, {'DJANGO_ALLOW_ASYNC_UNSAFE': ''})
    def test_no_allowed_async_unsafe(self):
        if False:
            print('Hello World!')
        self.assertEqual(check_async_unsafe(None), [])

    @mock.patch.dict(os.environ, {'DJANGO_ALLOW_ASYNC_UNSAFE': 'true'})
    def test_allowed_async_unsafe_set(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(check_async_unsafe(None), [E001])