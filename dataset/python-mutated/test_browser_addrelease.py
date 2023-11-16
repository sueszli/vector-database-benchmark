from test.picardtestcase import PicardTestCase
from picard.browser.addrelease import extract_discnumber
from picard.metadata import Metadata

class BrowserAddreleaseTest(PicardTestCase):

    def test_extract_discnumber(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(1, extract_discnumber(Metadata()))
        self.assertEqual(1, extract_discnumber(Metadata({'discnumber': '1'})))
        self.assertEqual(42, extract_discnumber(Metadata({'discnumber': '42'})))
        self.assertEqual(3, extract_discnumber(Metadata({'discnumber': '3/12'})))
        self.assertEqual(3, extract_discnumber(Metadata({'discnumber': ' 3 / 12 '})))