"""Tests for various static pages (like the About page)."""
from __future__ import annotations
from core import feconf
from core.tests import test_utils

class NoninteractivePagesTests(test_utils.GenericTestBase):

    def test_redirect_forum(self) -> None:
        if False:
            return 10
        response = self.get_html_response('/forum', expected_status_int=302)
        self.assertIn(feconf.GOOGLE_GROUP_URL, response.headers['location'])

    def test_redirect_about(self) -> None:
        if False:
            return 10
        response = self.get_html_response('/credits', expected_status_int=302)
        self.assertIn('about', response.headers['location'])

    def test_redirect_foundation(self) -> None:
        if False:
            print('Hello World!')
        response = self.get_html_response('/foundation', expected_status_int=302)
        self.assertIn('about-foundation', response.headers['location'])

    def test_redirect_teach(self) -> None:
        if False:
            i = 10
            return i + 15
        response = self.get_html_response('/participate', expected_status_int=302)
        self.assertIn('teach', response.headers['location'])