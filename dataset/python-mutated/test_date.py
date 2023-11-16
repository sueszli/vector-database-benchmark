import pytest
from visidata import Sheet, date

class TestVisidataDate:

    def test_date(self):
        if False:
            for i in range(10):
                print('nop')
        dt = Sheet().customdate('%d%m%Y')
        assert not date(2021, 7, 1) <= dt('22092017')
        assert date(2021, 7, 1) <= dt('28092021')