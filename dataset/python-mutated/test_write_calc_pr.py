import unittest
from io import StringIO
from ...workbook import Workbook

class TestWriteCalcPr(unittest.TestCase):
    """
    Test the Workbook _write_calc_pr() method.

    """

    def setUp(self):
        if False:
            return 10
        self.fh = StringIO()
        self.workbook = Workbook()
        self.workbook._set_filehandle(self.fh)

    def test_write_calc_pr(self):
        if False:
            print('Hello World!')
        'Test the _write_calc_pr() method.'
        self.workbook._write_calc_pr()
        exp = '<calcPr calcId="124519" fullCalcOnLoad="1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_calc_mode_auto_except_tables(self):
        if False:
            print('Hello World!')
        '\n        Test the _write_calc_pr() method with the calculation mode set\n        to auto_except_tables.\n\n        '
        self.workbook.set_calc_mode('auto_except_tables')
        self.workbook._write_calc_pr()
        exp = '<calcPr calcId="124519" calcMode="autoNoTable" fullCalcOnLoad="1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_calc_mode_manual(self):
        if False:
            while True:
                i = 10
        '\n        Test the _write_calc_pr() method with the calculation mode set to\n        manual.\n\n        '
        self.workbook.set_calc_mode('manual')
        self.workbook._write_calc_pr()
        exp = '<calcPr calcId="124519" calcMode="manual" calcOnSave="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_calc_pr2(self):
        if False:
            while True:
                i = 10
        'Test the _write_calc_pr() method with non-default calc id.'
        self.workbook.set_calc_mode('auto', 12345)
        self.workbook._write_calc_pr()
        exp = '<calcPr calcId="12345" fullCalcOnLoad="1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.workbook.fileclosed = 1