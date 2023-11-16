from locale import strxfrm
from test.picardtestcase import PicardTestCase
from picard.util import natsort

class NatsortTest(PicardTestCase):

    def test_natkey(self):
        if False:
            return 10
        self.assertTrue(natsort.natkey('foo1bar') < natsort.natkey('foo02bar'))
        self.assertTrue(natsort.natkey('foo1bar') == natsort.natkey('foo01bar'))
        self.assertTrue(natsort.natkey('foo (100)') < natsort.natkey('foo (00200)'))

    def test_natsorted(self):
        if False:
            return 10
        unsorted_list = ['foo11', 'foo0012', 'foo02', 'foo0', 'foo1', 'foo10', 'foo9']
        expected = ['foo0', 'foo1', 'foo02', 'foo9', 'foo10', 'foo11', 'foo0012']
        sorted_list = natsort.natsorted(unsorted_list)
        self.assertEqual(expected, sorted_list)

    def test_natkey_handles_null_char(self):
        if False:
            while True:
                i = 10
        self.assertEqual(natsort.natkey('foo\x00'), natsort.natkey('foo'))

    def test_natkey_handles_numeric_chars(self):
        if False:
            while True:
                i = 10
        self.assertEqual(natsort.natkey('foo0123456789|²³|٠١٢٣٤٥٦٧٨٩|๐๑๒๓๔๕๖๗๘๙|𝟜𝟚bar'), [strxfrm('foo'), 123456789, strxfrm('|²³|'), 123456789, strxfrm('|'), 123456789, strxfrm('|'), 42, strxfrm('bar')])