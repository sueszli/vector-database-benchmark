from __future__ import unicode_literals
from __future__ import print_function
import mock
from tests.compat import unittest
from gitsome.github import GitHub
from tests.data.markdown import formatted_markdown, raw_markdown, ssl_error

class WebViewerTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.github = GitHub()

    def test_format_markdown(self):
        if False:
            while True:
                i = 10
        result = self.github.web_viewer.format_markdown(raw_markdown)
        assert result == formatted_markdown

    @mock.patch('gitsome.github.click.echo_via_pager')
    def test_view_url(self, mock_click_echo_via_pager):
        if False:
            return 10
        url = 'https://www.github.com/donnemartin/gitsome'
        self.github.web_viewer.view_url(url)
        assert mock_click_echo_via_pager.mock_calls

    @unittest.skip('Skipping test_view_url_ssl_error')
    @mock.patch('gitsome.github.click.echo_via_pager')
    def test_view_url_ssl_error(self, mock_click_echo_via_pager):
        if False:
            for i in range(10):
                print('nop')
        'Temp skipping this test due to a change [undocumented?] in the way\n        the requests ssl error sample website is handled:\n            http://docs.python-requests.org/en/master/user/advanced/#ssl-cert-verification  # NOQA\n        See https://github.com/donnemartin/gitsome/pull/64 for more details.\n        '
        url = 'https://requestb.in'
        self.github.web_viewer.view_url(url)
        mock_click_echo_via_pager.assert_called_with(ssl_error, None)