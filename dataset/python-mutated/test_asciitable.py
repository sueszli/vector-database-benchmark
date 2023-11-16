import os
import unittest
from jc.exceptions import ParseError
import jc.parsers.asciitable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):

    def test_asciitable_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'asciitable' with no data\n        "
        self.assertEqual(jc.parsers.asciitable.parse('', quiet=True), [])

    def test_asciitable_m_pure_ascii(self):
        if False:
            print('Hello World!')
        "\n        Test 'asciitable' with a pure ASCII table\n        "
        input = '\n+========+========+========+========+========+========+========+\n| type   | tota   | used   | fr ee  | shar   | buff   | avai   |\n\n+========+========+========+========+========+========+========+\n| Mem    | 3861   | 2228   | 3364   | 1183   | 2743   | 3389   |\n+--------+--------+--------+--------+--------+--------+--------+\n|        |        |        |        | test 2 |        |        |\n+--------+--------+--------+--------+--------+--------+--------+\n| last   | last   | last   | ab cde |        |        | final  |\n+========+========+========+========+========+========+========+\n        '
        expected = [{'type': 'Mem', 'tota': '3861', 'used': '2228', 'fr_ee': '3364', 'shar': '1183', 'buff': '2743', 'avai': '3389'}, {'type': None, 'tota': None, 'used': None, 'fr_ee': None, 'shar': 'test 2', 'buff': None, 'avai': None}, {'type': 'last', 'tota': 'last', 'used': 'last', 'fr_ee': 'ab cde', 'shar': None, 'buff': None, 'avai': 'final'}]
        self.assertEqual(jc.parsers.asciitable.parse(input, quiet=True), expected)

    def test_asciitable_m_unicode(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'asciitable' with a unicode table\n        "
        input = '\n╒════════╤════════╤════════╤════════╤════════╤════════╤════════╕\n│ type   │ total  │ used   │ fr ee  │ shar   │ buff   │ avai   │\n╞════════╪════════╪════════╪════════╪════════╪════════╪════════╡\n│ Mem    │ 3861   │ 2228   │ 3364   │ 1183   │ 2743   │ 3389   │\n├────────┼────────┼────────┼────────┼────────┼────────┼────────┤\n│ Swap   │ 2097   │ 0      │ 2097   │        │        │        │\n├────────┼────────┼────────┼────────┼────────┼────────┼────────┤\n│ last   │ last   │ last   │ ab cde │        │        │ final  │\n╘════════╧════════╧════════╧════════╧════════╧════════╧════════╛\n        '
        expected = [{'type': 'Mem', 'total': '3861', 'used': '2228', 'fr_ee': '3364', 'shar': '1183', 'buff': '2743', 'avai': '3389'}, {'type': 'Swap', 'total': '2097', 'used': '0', 'fr_ee': '2097', 'shar': None, 'buff': None, 'avai': None}, {'type': 'last', 'total': 'last', 'used': 'last', 'fr_ee': 'ab cde', 'shar': None, 'buff': None, 'avai': 'final'}]
        self.assertEqual(jc.parsers.asciitable.parse(input, quiet=True), expected)

    def test_asciitable_pure_ascii_extra_spaces(self):
        if False:
            return 10
        "\n        Test 'asciitable' with a pure ASCII table that has heading and\n        trailing spaces and newlines.\n        "
        input = '\n    \n      \n    +========+========+========+========+========+========+========+\n    | type   | total  | used   | fr ee  | shar   | buff   | avai        \n    +========+========+========+========+========+========+========+   \n    | Mem    | 3861   | 2228   | 3364   | 1183   | 2743   | 3389     \n    +--------+--------+--------+--------+--------+--------+--------+\n             |        |        |        | test 2 |        |\n    +--------+--------+--------+--------+--------+--------+--------+       \n    | last   | last   | last   | ab cde |        |        | final  |\n    +========+========+========+========+========+========+========+    \n     \n  \n        '
        expected = [{'type': 'Mem', 'total': '3861', 'used': '2228', 'fr_ee': '3364', 'shar': '1183', 'buff': '2743', 'avai': '3389'}, {'type': None, 'total': None, 'used': None, 'fr_ee': None, 'shar': 'test 2', 'buff': None, 'avai': None}, {'type': 'last', 'total': 'last', 'used': 'last', 'fr_ee': 'ab cde', 'shar': None, 'buff': None, 'avai': 'final'}]
        self.assertEqual(jc.parsers.asciitable.parse(input, quiet=True), expected)

    def test_asciitable_unicode_extra_spaces(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'asciitable' with a pure ASCII table that has heading and\n        trailing spaces and newlines.\n        "
        input = '\n    \n  \n        ╒════════╤════════╤════════╤════════╤════════╤════════╤════════╕\n          type   │ total  │ used   │ free   │ shar   │ buff   │ avai\n        ╞════════╪════════╪════════╪════════╪════════╪════════╪════════╡      \n          Mem    │ 3861   │ 2228   │ 3364   │ 1183   │ 2743   │ 3389     \n        ├────────┼────────┼────────┼────────┼────────┼────────┼────────┤  \n          Swap   │ 2097   │ 0      │ 2097   │        │        │\n        ╘════════╧════════╧════════╧════════╧════════╧════════╧════════╛\n   \n \n        '
        expected = [{'type': 'Mem', 'total': '3861', 'used': '2228', 'free': '3364', 'shar': '1183', 'buff': '2743', 'avai': '3389'}, {'type': 'Swap', 'total': '2097', 'used': '0', 'free': '2097', 'shar': None, 'buff': None, 'avai': None}]
        self.assertEqual(jc.parsers.asciitable.parse(input, quiet=True), expected)

    def test_asciitable_markdown(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'asciitable' with a markdown table\n        "
        input = '\n        | type   |   total |   used |    free |   shared |   buff cache |   available |\n        |--------|---------|--------|---------|----------|--------------|-------------|\n        | Mem    | 3861332 | 222820 | 3364176 |    11832 |       274336 |     3389588 |\n        | Swap   | 2097148 |      0 | 2097148 |          |              |             |\n        '
        expected = [{'type': 'Mem', 'total': '3861332', 'used': '222820', 'free': '3364176', 'shared': '11832', 'buff_cache': '274336', 'available': '3389588'}, {'type': 'Swap', 'total': '2097148', 'used': '0', 'free': '2097148', 'shared': None, 'buff_cache': None, 'available': None}]
        self.assertEqual(jc.parsers.asciitable.parse(input, quiet=True), expected)

    def test_asciitable_simple(self):
        if False:
            print('Hello World!')
        "\n        Test 'asciitable' with a simple table\n        "
        input = '\n        type      total    used     free    shared    buff cache    available\n        ------  -------  ------  -------  --------  ------------  -----------\n        Mem     3861332  222820  3364176     11832        274336      3389588\n        Swap    2097148       0  2097148\n        '
        expected = [{'type': 'Mem', 'total': '3861332', 'used': '222820', 'free': '3364176', 'shared': '11832', 'buff_cache': '274336', 'available': '3389588'}, {'type': 'Swap', 'total': '2097148', 'used': '0', 'free': '2097148', 'shared': None, 'buff_cache': None, 'available': None}]
        self.assertEqual(jc.parsers.asciitable.parse(input, quiet=True), expected)

    def test_asciitable_pretty_ansi(self):
        if False:
            print('Hello World!')
        "\n        Test 'asciitable' with a pretty table with ANSI codes\n        "
        input = '┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓                                   \n                                   ┃\x1b[1m \x1b[0m\x1b[1mReleased    \x1b[0m\x1b[1m \x1b[0m┃\x1b[1m \x1b[0m\x1b[1mTitle                            \x1b[0m\x1b[1m \x1b[0m┃\x1b[1m \x1b[0m\x1b[1m    Box Office\x1b[0m\x1b[1m \x1b[0m┃                                   \n                                   ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩                                   \n                                   │\x1b[36m \x1b[0m\x1b[36mDec 20, 2019\x1b[0m\x1b[36m \x1b[0m│\x1b[35m \x1b[0m\x1b[35mStar Wars: The Rise of Skywalker \x1b[0m\x1b[35m \x1b[0m│\x1b[32m \x1b[0m\x1b[32m  $952,110,690\x1b[0m\x1b[32m \x1b[0m│                                   \n                                   │\x1b[36m \x1b[0m\x1b[36mMay 25, 2018\x1b[0m\x1b[36m \x1b[0m│\x1b[35m \x1b[0m\x1b[35mSolo: A Star Wars Story          \x1b[0m\x1b[35m \x1b[0m│\x1b[32m \x1b[0m\x1b[32m  $393,151,347\x1b[0m\x1b[32m \x1b[0m│                                   \n                                   │\x1b[36m \x1b[0m\x1b[36mDec 15, 2017\x1b[0m\x1b[36m \x1b[0m│\x1b[35m \x1b[0m\x1b[35mStar Wars Ep. V111: The Last Jedi\x1b[0m\x1b[35m \x1b[0m│\x1b[32m \x1b[0m\x1b[32m$1,332,539,889\x1b[0m\x1b[32m \x1b[0m│                                   \n                                   │\x1b[36m \x1b[0m\x1b[36mDec 16, 2016\x1b[0m\x1b[36m \x1b[0m│\x1b[35m \x1b[0m\x1b[35mRogue One: A Star Wars Story     \x1b[0m\x1b[35m \x1b[0m│\x1b[32m \x1b[0m\x1b[32m$1,332,439,889\x1b[0m\x1b[32m \x1b[0m│                                   \n                                   └──────────────┴───────────────────────────────────┴────────────────┘                                   \n'
        expected = [{'released': 'Dec 20, 2019', 'title': 'Star Wars: The Rise of Skywalker', 'box_office': '$952,110,690'}, {'released': 'May 25, 2018', 'title': 'Solo: A Star Wars Story', 'box_office': '$393,151,347'}, {'released': 'Dec 15, 2017', 'title': 'Star Wars Ep. V111: The Last Jedi', 'box_office': '$1,332,539,889'}, {'released': 'Dec 16, 2016', 'title': 'Rogue One: A Star Wars Story', 'box_office': '$1,332,439,889'}]
        self.assertEqual(jc.parsers.asciitable.parse(input, quiet=True), expected)

    def test_asciitable_special_chars_in_header(self):
        if False:
            print('Hello World!')
        "\n        Test 'asciitable' with a pure ASCII table that has special\n        characters in the header. These should be converted to underscores\n        and no trailing or consecutive underscores should end up in the\n        resulting key names.\n        "
        input = '\nProtocol  Address     Age (min)  Hardware Addr   Type   Interface\nInternet  10.12.13.1        98   0950.5785.5cd1  ARPA   FastEthernet2.13\nInternet  10.12.13.3       131   0150.7685.14d5  ARPA   GigabitEthernet2.13\nInternet  10.12.13.4       198   0950.5C8A.5c41  ARPA   GigabitEthernet2.17\n        '
        expected = [{'protocol': 'Internet', 'address': '10.12.13.1', 'age_min': '98', 'hardware_addr': '0950.5785.5cd1', 'type': 'ARPA', 'interface': 'FastEthernet2.13'}, {'protocol': 'Internet', 'address': '10.12.13.3', 'age_min': '131', 'hardware_addr': '0150.7685.14d5', 'type': 'ARPA', 'interface': 'GigabitEthernet2.13'}, {'protocol': 'Internet', 'address': '10.12.13.4', 'age_min': '198', 'hardware_addr': '0950.5C8A.5c41', 'type': 'ARPA', 'interface': 'GigabitEthernet2.17'}]
        self.assertEqual(jc.parsers.asciitable.parse(input, quiet=True), expected)

    def test_asciitable_no_lower_raw(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'asciitable' with a pure ASCII table that has special\n        characters and mixed case in the header. These should be converted to underscores\n        and no trailing or consecutive underscores should end up in the\n        resulting key names. Using `raw` in this test to preserve case. (no lower)\n        "
        input = '\nProtocol  Address     Age (min)  Hardware Addr   Type   Interface\nInternet  10.12.13.1        98   0950.5785.5cd1  ARPA   FastEthernet2.13\nInternet  10.12.13.3       131   0150.7685.14d5  ARPA   GigabitEthernet2.13\nInternet  10.12.13.4       198   0950.5C8A.5c41  ARPA   GigabitEthernet2.17\n        '
        expected = [{'Protocol': 'Internet', 'Address': '10.12.13.1', 'Age_min': '98', 'Hardware_Addr': '0950.5785.5cd1', 'Type': 'ARPA', 'Interface': 'FastEthernet2.13'}, {'Protocol': 'Internet', 'Address': '10.12.13.3', 'Age_min': '131', 'Hardware_Addr': '0150.7685.14d5', 'Type': 'ARPA', 'Interface': 'GigabitEthernet2.13'}, {'Protocol': 'Internet', 'Address': '10.12.13.4', 'Age_min': '198', 'Hardware_Addr': '0950.5C8A.5c41', 'Type': 'ARPA', 'Interface': 'GigabitEthernet2.17'}]
        self.assertEqual(jc.parsers.asciitable.parse(input, raw=True, quiet=True), expected)

    def test_asciitable_centered_col_header(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'asciitable' with long centered column header which can break\n        column alignment\n        "
        input = '\n            +---------+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+------------------+\n            | fdc_id  |        data_type         |                                                                                                                           description                                                                                                                           | food_category_id | publication_date |\n            +---------+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+------------------+\n            | 167512  | sr_legacy_food           | Pillsbury Golden Layer Buttermilk Biscuits, Artificial Flavor, refrigerated dough                                                                                                                                                                               |                  | 2019-04-01       |\n            | 167513  | sr_legacy_food           | Pillsbury, Cinnamon Rolls with Icing, refrigerated dough                                                                                                                                                                                                        |                  | 2019-04-01       |\n            | 167514  | sr_legacy_food           | Kraft Foods, Shake N Bake Original Recipe, Coating for Pork, dry                                                                                                                                                                                                |                  | 2019-04-01       |\n            | 167515  | sr_legacy_food           | George Weston Bakeries, Thomas English Muffins                                                                                                                                                                                                                  |                  | 2019-04-01       |\n            | 167516  | sr_legacy_food           | Waffles, buttermilk, frozen, ready-to-heat                                                                                                                                                                                                                      |                  | 2019-04-01       |\n            | 167517  | sr_legacy_food           | Waffle, buttermilk, frozen, ready-to-heat, toasted                                                                                                                                                                                                              |                  | 2019-04-01       |\n            | 167518  | sr_legacy_food           | Waffle, buttermilk, frozen, ready-to-heat, microwaved                                                                                                                                                                                                           |                  | 2019-04-01       |\n        '
        expected = [{'fdc_id': '167512', 'data_type': 'sr_legacy_food', 'description': 'Pillsbury Golden Layer Buttermilk Biscuits, Artificial Flavor, refrigerated dough', 'food_category_id': None, 'publication_date': '2019-04-01'}, {'fdc_id': '167513', 'data_type': 'sr_legacy_food', 'description': 'Pillsbury, Cinnamon Rolls with Icing, refrigerated dough', 'food_category_id': None, 'publication_date': '2019-04-01'}, {'fdc_id': '167514', 'data_type': 'sr_legacy_food', 'description': 'Kraft Foods, Shake N Bake Original Recipe, Coating for Pork, dry', 'food_category_id': None, 'publication_date': '2019-04-01'}, {'fdc_id': '167515', 'data_type': 'sr_legacy_food', 'description': 'George Weston Bakeries, Thomas English Muffins', 'food_category_id': None, 'publication_date': '2019-04-01'}, {'fdc_id': '167516', 'data_type': 'sr_legacy_food', 'description': 'Waffles, buttermilk, frozen, ready-to-heat', 'food_category_id': None, 'publication_date': '2019-04-01'}, {'fdc_id': '167517', 'data_type': 'sr_legacy_food', 'description': 'Waffle, buttermilk, frozen, ready-to-heat, toasted', 'food_category_id': None, 'publication_date': '2019-04-01'}, {'fdc_id': '167518', 'data_type': 'sr_legacy_food', 'description': 'Waffle, buttermilk, frozen, ready-to-heat, microwaved', 'food_category_id': None, 'publication_date': '2019-04-01'}]
        self.assertEqual(jc.parsers.asciitable.parse(input, quiet=True), expected)
if __name__ == '__main__':
    unittest.main()