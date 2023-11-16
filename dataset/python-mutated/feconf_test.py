"""Unit tests for core/feconf.py."""
from __future__ import annotations
import datetime
import os
from core import feconf
from core.tests import test_utils
import bs4

class FeconfTests(test_utils.GenericTestBase):
    """Unit tests for core/feconf.py."""

    def test_dev_mode_in_production_throws_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def mock_getenv(env: str) -> str:
            if False:
                for i in range(10):
                    print('nop')
            if env == 'SERVER_SOFTWARE':
                return 'Production'
            return 'Development'
        swap_getenv = self.swap(os, 'getenv', mock_getenv)
        with swap_getenv, self.assertRaisesRegex(Exception, "DEV_MODE can't be true on production."):
            feconf.check_dev_mode_is_true()

    def test_dev_mode_in_development_passes_succcessfully(self) -> None:
        if False:
            return 10

        def mock_getenv(*unused_args: str) -> str:
            if False:
                i = 10
                return i + 15
            return 'Development'
        swap_getenv = self.swap(os, 'getenv', mock_getenv)
        with swap_getenv:
            feconf.check_dev_mode_is_true()

    def test_get_empty_ratings(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(feconf.get_empty_ratings(), {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0})

    def test_callable_variables_return_correctly(self) -> None:
        if False:
            return 10
        recipient_username = 'Anshuman'
        self.assertEqual(feconf.DEFAULT_SALUTATION_HTML_FN(recipient_username), 'Hi %s,' % recipient_username)
        sender_username = 'Ezio'
        self.assertEqual(feconf.DEFAULT_SIGNOFF_HTML_FN(sender_username), 'Thanks!<br>%s (Oppia moderator)' % sender_username)
        exploration_title = 'Test'
        self.assertEqual(feconf.DEFAULT_EMAIL_SUBJECT_FN(exploration_title), 'Your Oppia exploration "Test" has been unpublished')
        self.assertEqual(feconf.VALID_MODERATOR_ACTIONS['unpublish_exploration']['email_config'], 'unpublish_exploration_email_html_body')
        self.assertEqual(feconf.VALID_MODERATOR_ACTIONS['unpublish_exploration']['email_intent'], 'unpublish_exploration')

    def test_terms_page_last_updated_is_in_sync_with_terms_page(self) -> None:
        if False:
            while True:
                i = 10
        with open('core/templates/pages/terms-page/terms-page.component.html', 'r', encoding='utf-8') as f:
            terms_page_contents = f.read()
            terms_page_parsed_html = bs4.BeautifulSoup(terms_page_contents, 'html.parser')
            max_date = max((datetime.datetime.strptime(element.get_text().split(':')[0], '%d %b %Y') for element in terms_page_parsed_html.find('ul', class_='e2e-test-changelog').find_all('li')))
        self.assertEqual(feconf.TERMS_PAGE_LAST_UPDATED_UTC, max_date)