"""
    Test win_chcp
"""
import pytest
from salt.exceptions import CodePageError
from salt.utils import win_chcp
from tests.support.unit import TestCase
pytestmark = [pytest.mark.skip_unless_on_windows]

class CHCPTest(TestCase):
    """
    Test case for salt.utils.win_chcp
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls._chcp_code = win_chcp.get_codepage_id()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls._chcp_code = None

    def setUp(self):
        if False:
            return 10
        win_chcp.set_codepage_id(self._chcp_code)

    def tearDown(self):
        if False:
            while True:
                i = 10
        win_chcp.set_codepage_id(self._chcp_code)

    def test_get_and_set_code_page(self):
        if False:
            for i in range(10):
                print('nop')
        for page in (20424, '20866', 437, 65001, '437'):
            self.assertEqual(win_chcp.set_codepage_id(page), int(page))
            self.assertEqual(win_chcp.get_codepage_id(), int(page))

    def test_bad_page_code(self):
        if False:
            for i in range(10):
                print('nop')
        with win_chcp.chcp(437):
            self.assertEqual(win_chcp.get_codepage_id(), 437)
            bad_codes = ('0', 'bad code', 1234, -34, '437 dogs', '(*&^(*^%&$%&')
            for page in bad_codes:
                self.assertEqual(win_chcp.set_codepage_id(page), -1)
                self.assertEqual(win_chcp.get_codepage_id(), 437)
            for page in bad_codes:
                self.assertRaises(CodePageError, win_chcp.set_codepage_id, page, True)