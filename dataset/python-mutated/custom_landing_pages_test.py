"""Tests for custom landing pages."""
from __future__ import annotations
from core import feconf
from core.tests import test_utils

class FractionLandingRedirectPageTest(test_utils.GenericTestBase):
    """Test for redirecting landing page for fractions."""

    def test_old_fractions_landing_url_without_viewer_type(self) -> None:
        if False:
            return 10
        'Test to validate the old Fractions landing url without viewerType\n        redirects to the new Fractions landing url.\n        '
        response = self.get_html_response(feconf.FRACTIONS_LANDING_PAGE_URL, expected_status_int=302)
        self.assertEqual('http://localhost/math/fractions', response.headers['location'])

    def test_old_fraction_landing_url_with_viewer_type(self) -> None:
        if False:
            while True:
                i = 10
        'Test to validate the old Fractions landing url with viewerType\n        redirects to the new Fractions landing url.\n        '
        response = self.get_html_response('%s?viewerType=student' % feconf.FRACTIONS_LANDING_PAGE_URL, expected_status_int=302)
        self.assertEqual('http://localhost/math/fractions', response.headers['location'])

class TopicLandingRedirectPageTest(test_utils.GenericTestBase):
    """Test for redirecting the old landing page URL to the new one."""

    def test_old_topic_url_redirect(self) -> None:
        if False:
            return 10
        response = self.get_html_response('/learn/maths/fractions', expected_status_int=302)
        self.assertEqual('http://localhost/math/fractions', response.headers['location'])