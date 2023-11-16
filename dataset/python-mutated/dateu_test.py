"""
@author: ZackZK
"""
from unittest import TestCase
from tushare.util import dateu
from tushare.util.dateu import is_holiday

class Test_Is_holiday(TestCase):

    def test_is_holiday(self):
        if False:
            print('Hello World!')
        dateu.holiday = ['2016-01-04']
        self.assertTrue(is_holiday('2016-01-04'))
        self.assertFalse(is_holiday('2016-01-01'))
        self.assertTrue(is_holiday('2016-01-09'))
        self.assertTrue(is_holiday('2016-01-10'))