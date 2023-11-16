import os.path
from test.picardtestcase import PicardTestCase
from picard.util import bytes2human

class Testbytes2human(PicardTestCase):

    def test_00(self):
        if False:
            i = 10
            return i + 15
        self.run_test()
        self.assertEqual(bytes2human.binary(45682), '44.6 KiB')
        self.assertEqual(bytes2human.binary(-45682), '-44.6 KiB')
        self.assertEqual(bytes2human.binary(-45682, 2), '-44.61 KiB')
        self.assertEqual(bytes2human.decimal(45682), '45.7 kB')
        self.assertEqual(bytes2human.decimal(45682, 2), '45.68 kB')
        self.assertEqual(bytes2human.decimal(9223372036854775807), '9223.4 PB')
        self.assertEqual(bytes2human.decimal(9223372036854775807, 3), '9223.372 PB')
        self.assertEqual(bytes2human.decimal(123.6), '123 B')
        self.assertRaises(ValueError, bytes2human.decimal, 'xxx')
        self.assertRaises(ValueError, bytes2human.decimal, '123.6')
        self.assertRaises(ValueError, bytes2human.binary, 'yyy')
        self.assertRaises(ValueError, bytes2human.binary, '456yyy')
        try:
            bytes2human.decimal('123')
        except Exception as e:
            self.fail('Unexpected exception: %s' % e)

    def test_calc_unit_raises_value_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, bytes2human.calc_unit, 1, None)
        self.assertRaises(ValueError, bytes2human.calc_unit, 1, 100)
        self.assertRaises(ValueError, bytes2human.calc_unit, 1, 999)
        self.assertRaises(ValueError, bytes2human.calc_unit, 1, 1023)
        self.assertRaises(ValueError, bytes2human.calc_unit, 1, 1025)
        self.assertEqual((1.0, 'B'), bytes2human.calc_unit(1, 1024))
        self.assertEqual((1.0, 'B'), bytes2human.calc_unit(1, 1000))

    def run_test(self, lang='C', create_test_data=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compare data generated with sample files\n        Setting create_test_data to True will generated sample files\n        from code execution (developer-only, check carefully)\n        '
        filename = os.path.join('test', 'data', 'b2h_test_%s.dat' % lang)
        testlist = self._create_testlist()
        if create_test_data:
            self._save_expected_to(filename, testlist)
        expected = self._read_expected_from(filename)
        self.assertEqual(testlist, expected)
        if create_test_data:
            self.fail('!!! UNSET create_test_data mode !!! (%s)' % filename)

    @staticmethod
    def _create_testlist():
        if False:
            return 10
        values = [0, 1]
        for n in (1000, 1024):
            p = 1
            for e in range(0, 6):
                p *= n
                for x in (0.1, 0.5, 0.99, 0.9999, 1, 1.5):
                    values.append(int(p * x))
        list = []
        for x in sorted(values):
            list.append(';'.join([str(x), bytes2human.decimal(x), bytes2human.binary(x), bytes2human.short_string(x, 1024, 2)]))
        return list

    @staticmethod
    def _save_expected_to(path, a_list):
        if False:
            return 10
        with open(path, 'wb') as f:
            f.writelines([line + '\n' for line in a_list])
            f.close()

    @staticmethod
    def _read_expected_from(path):
        if False:
            print('Hello World!')
        with open(path, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
            f.close()
            return lines

    def test_calc_unit(self):
        if False:
            return 10
        self.assertEqual(bytes2human.calc_unit(12456, 1024), (12.1640625, 'KiB'))
        self.assertEqual(bytes2human.calc_unit(-12456, 1000), (-12.456, 'kB'))
        self.assertRaises(ValueError, bytes2human.calc_unit, 0, 1001)