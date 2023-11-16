import testtools
import bandit
from bandit.core.docs_utils import get_url

class DocsUtilTests(testtools.TestCase):
    """This set of tests exercises bandit.core.docs_util functions."""
    BASE_URL = f'https://bandit.readthedocs.io/en/{bandit.__version__}/'

    def test_overwrite_bib_info(self):
        if False:
            while True:
                i = 10
        expected_url = self.BASE_URL + 'blacklists/blacklist_calls.html#b304-b305-ciphers-and-modes'
        self.assertEqual(get_url('B304'), get_url('B305'))
        self.assertEqual(expected_url, get_url('B304'))

    def test_plugin_call_bib(self):
        if False:
            for i in range(10):
                print('nop')
        expected_url = self.BASE_URL + 'plugins/b101_assert_used.html'
        self.assertEqual(expected_url, get_url('B101'))

    def test_import_call_bib(self):
        if False:
            return 10
        expected_url = self.BASE_URL + 'blacklists/blacklist_imports.html#b413-import-pycrypto'
        self.assertEqual(expected_url, get_url('B413'))