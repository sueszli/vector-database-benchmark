"""
Created on 2015/3/14
@author: Jimmy Liu
"""
import unittest
import tushare.stock.macro as fd

class Test(unittest.TestCase):

    def test_get_gdp_year(self):
        if False:
            i = 10
            return i + 15
        print(fd.get_gdp_year())

    def test_get_gdp_quarter(self):
        if False:
            while True:
                i = 10
        print(fd.get_gdp_quarter())

    def test_get_gdp_for(self):
        if False:
            for i in range(10):
                print('nop')
        print(fd.get_gdp_for())

    def test_get_gdp_pull(self):
        if False:
            i = 10
            return i + 15
        print(fd.get_gdp_pull())

    def test_get_gdp_contrib(self):
        if False:
            while True:
                i = 10
        print(fd.get_gdp_contrib())

    def test_get_cpi(self):
        if False:
            i = 10
            return i + 15
        print(fd.get_cpi())

    def test_get_ppi(self):
        if False:
            while True:
                i = 10
        print(fd.get_ppi())

    def test_get_deposit_rate(self):
        if False:
            return 10
        print(fd.get_deposit_rate())

    def test_get_loan_rate(self):
        if False:
            return 10
        print(fd.get_loan_rate())

    def test_get_rrr(self):
        if False:
            while True:
                i = 10
        print(fd.get_rrr())

    def test_get_money_supply(self):
        if False:
            for i in range(10):
                print('nop')
        print(fd.get_money_supply())

    def test_get_money_supply_bal(self):
        if False:
            return 10
        print(fd.get_money_supply_bal())
if __name__ == '__main__':
    unittest.main()