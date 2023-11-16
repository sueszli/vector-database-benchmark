from decimal import Decimal
from sys import float_info
from django.test import SimpleTestCase
from django.utils.numberformat import format as nformat

class TestNumberFormat(SimpleTestCase):

    def test_format_number(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(nformat(1234, '.'), '1234')
        self.assertEqual(nformat(1234.2, '.'), '1234.2')
        self.assertEqual(nformat(1234, '.', decimal_pos=2), '1234.00')
        self.assertEqual(nformat(1234, '.', grouping=2, thousand_sep=','), '1234')
        self.assertEqual(nformat(1234, '.', grouping=2, thousand_sep=',', force_grouping=True), '12,34')
        self.assertEqual(nformat(-1234.33, '.', decimal_pos=1), '-1234.3')
        with self.settings(USE_THOUSAND_SEPARATOR=True):
            self.assertEqual(nformat(1234, '.', grouping=3, thousand_sep=',', use_l10n=False), '1234')
            self.assertEqual(nformat(1234, '.', grouping=3, thousand_sep=',', use_l10n=True), '1,234')

    def test_format_string(self):
        if False:
            return 10
        self.assertEqual(nformat('1234', '.'), '1234')
        self.assertEqual(nformat('1234.2', '.'), '1234.2')
        self.assertEqual(nformat('1234', '.', decimal_pos=2), '1234.00')
        self.assertEqual(nformat('1234', '.', grouping=2, thousand_sep=','), '1234')
        self.assertEqual(nformat('1234', '.', grouping=2, thousand_sep=',', force_grouping=True), '12,34')
        self.assertEqual(nformat('-1234.33', '.', decimal_pos=1), '-1234.3')
        self.assertEqual(nformat('10000', '.', grouping=3, thousand_sep='comma', force_grouping=True), '10comma000')

    def test_large_number(self):
        if False:
            i = 10
            return i + 15
        most_max = '{}17976931348623157081452742373170435679807056752584499659891747680315726078002853876058955863276687817154045895351438246423432132688946418276846754670353751698604991057655128207624549009038932894407586850845513394230458323690322294816580855933212334827479782620414472316873817718091929988125040402618412485836{}'
        most_max2 = '{}359538626972463141629054847463408713596141135051689993197834953606314521560057077521179117265533756343080917907028764928468642653778928365536935093407075033972099821153102564152490980180778657888151737016910267884609166473806445896331617118664246696549595652408289446337476354361838599762500808052368249716736'
        int_max = int(float_info.max)
        self.assertEqual(nformat(int_max, '.'), most_max.format('', '8'))
        self.assertEqual(nformat(int_max + 1, '.'), most_max.format('', '9'))
        self.assertEqual(nformat(int_max * 2, '.'), most_max2.format(''))
        self.assertEqual(nformat(0 - int_max, '.'), most_max.format('-', '8'))
        self.assertEqual(nformat(-1 - int_max, '.'), most_max.format('-', '9'))
        self.assertEqual(nformat(-2 * int_max, '.'), most_max2.format('-'))

    def test_float_numbers(self):
        if False:
            i = 10
            return i + 15
        tests = [(9e-10, 10, '0.0000000009'), (9e-19, 2, '0.00'), (9.9e-13, 0, '0'), (9.9e-13, 13, '0.0000000000009'), (1e+16, None, '10000000000000000'), (1e+16, 2, '10000000000000000.00'), (3.0, None, '3.0')]
        for (value, decimal_pos, expected_value) in tests:
            with self.subTest(value=value, decimal_pos=decimal_pos):
                self.assertEqual(nformat(value, '.', decimal_pos), expected_value)
        self.assertEqual(nformat(1e+16, '.', thousand_sep=',', grouping=3, force_grouping=True), '10,000,000,000,000,000')
        self.assertEqual(nformat(1e+16, '.', decimal_pos=2, thousand_sep=',', grouping=3, force_grouping=True), '10,000,000,000,000,000.00')

    def test_decimal_numbers(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(nformat(Decimal('1234'), '.'), '1234')
        self.assertEqual(nformat(Decimal('1234.2'), '.'), '1234.2')
        self.assertEqual(nformat(Decimal('1234'), '.', decimal_pos=2), '1234.00')
        self.assertEqual(nformat(Decimal('1234'), '.', grouping=2, thousand_sep=','), '1234')
        self.assertEqual(nformat(Decimal('1234'), '.', grouping=2, thousand_sep=',', force_grouping=True), '12,34')
        self.assertEqual(nformat(Decimal('-1234.33'), '.', decimal_pos=1), '-1234.3')
        self.assertEqual(nformat(Decimal('0.00000001'), '.', decimal_pos=8), '0.00000001')
        self.assertEqual(nformat(Decimal('9e-19'), '.', decimal_pos=2), '0.00')
        self.assertEqual(nformat(Decimal('.00000000000099'), '.', decimal_pos=0), '0')
        self.assertEqual(nformat(Decimal('1e16'), '.', thousand_sep=',', grouping=3, force_grouping=True), '10,000,000,000,000,000')
        self.assertEqual(nformat(Decimal('1e16'), '.', decimal_pos=2, thousand_sep=',', grouping=3, force_grouping=True), '10,000,000,000,000,000.00')
        self.assertEqual(nformat(Decimal('3.'), '.'), '3')
        self.assertEqual(nformat(Decimal('3.0'), '.'), '3.0')
        tests = [('9e9999', None, '9e+9999'), ('9e9999', 3, '9.000e+9999'), ('9e201', None, '9e+201'), ('9e200', None, '9e+200'), ('1.2345e999', 2, '1.23e+999'), ('9e-999', None, '9e-999'), ('1e-7', 8, '0.00000010'), ('1e-8', 8, '0.00000001'), ('1e-9', 8, '0.00000000'), ('1e-10', 8, '0.00000000'), ('1e-11', 8, '0.00000000'), ('1' + '0' * 300, 3, '1.000e+300'), ('0.{}1234'.format('0' * 299), 3, '0.000')]
        for (value, decimal_pos, expected_value) in tests:
            with self.subTest(value=value):
                self.assertEqual(nformat(Decimal(value), '.', decimal_pos), expected_value)

    def test_decimal_subclass(self):
        if False:
            i = 10
            return i + 15

        class EuroDecimal(Decimal):
            """
            Wrapper for Decimal which prefixes each amount with the € symbol.
            """

            def __format__(self, specifier, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                amount = super().__format__(specifier, **kwargs)
                return '€ {}'.format(amount)
        price = EuroDecimal('1.23')
        self.assertEqual(nformat(price, ','), '€ 1,23')

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(nformat('', '.'), '')
        self.assertEqual(nformat(None, '.'), 'None')