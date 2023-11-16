import os
import unittest
from jc.exceptions import ParseError
import jc.parsers.asciitable_m
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):

    def test_asciitable_m_nodata(self):
        if False:
            return 10
        "\n        Test 'asciitable_m' with no data\n        "
        self.assertEqual(jc.parsers.asciitable_m.parse('', quiet=True), [])

    def test_asciitable_m_pure_ascii(self):
        if False:
            return 10
        "\n        Test 'asciitable_m' with a pure ASCII table\n        "
        input = '\n+========+========+========+========+========+========+========+\n| type   | tota   | used   | fr ee  | shar   | buff   | avai   |\n|        | l      |        |        | ed     | _cac   | labl   |\n|        |        |        |        |        | he     | e      |\n+========+========+========+========+========+========+========+\n| Mem    | 3861   | 2228   | 3364   | 1183   | 2743   | 3389   |\n|        | 332    | 20     | 176    | 2      | 36     | 588    |\n+--------+--------+--------+--------+--------+--------+--------+\n|        |        |        |        |        |        |        |\n|        |        |        |        | test 2 |        |        |\n+--------+--------+--------+--------+--------+--------+--------+\n| last   | last   | last   | ab cde |        |        | final  |\n+========+========+========+========+========+========+========+\n        '
        expected = [{'type': 'Mem', 'tota_l': '3861\n332', 'used': '2228\n20', 'fr_ee': '3364\n176', 'shar_ed': '1183\n2', 'buff_cac_he': '2743\n36', 'avai_labl_e': '3389\n588'}, {'type': None, 'tota_l': None, 'used': None, 'fr_ee': None, 'shar_ed': 'test 2', 'buff_cac_he': None, 'avai_labl_e': None}, {'type': 'last', 'tota_l': 'last', 'used': 'last', 'fr_ee': 'ab cde', 'shar_ed': None, 'buff_cac_he': None, 'avai_labl_e': 'final'}]
        self.assertEqual(jc.parsers.asciitable_m.parse(input, quiet=True), expected)

    def test_asciitable_m_unicode(self):
        if False:
            print('Hello World!')
        "\n        Test 'asciitable_m' with a unicode table\n        "
        input = '\n╒════════╤════════╤════════╤════════╤════════╤════════╤════════╕\n│ type   │ tota   │ used   │ fr ee  │ shar   │ buff   │ avai   │\n│        │ l      │        │        │ ed     │ _cac   │ labl   │\n│        │        │        │        │        │ he     │ e      │\n╞════════╪════════╪════════╪════════╪════════╪════════╪════════╡\n│ Mem    │ 3861   │ 2228   │ 3364   │ 1183   │ 2743   │ 3389   │\n│        │ 332    │ 20     │ 176    │ 2      │ 36     │ 588    │\n├────────┼────────┼────────┼────────┼────────┼────────┼────────┤\n│ Swap   │ 2097   │ 0      │ 2097   │        │        │        │\n│        │ 148    │        │ 148    │        │        │        │\n│        │        │        │ kb     │        │        │        │\n├────────┼────────┼────────┼────────┼────────┼────────┼────────┤\n│ last   │ last   │ last   │ ab cde │        │        │ final  │\n╘════════╧════════╧════════╧════════╧════════╧════════╧════════╛\n        '
        expected = [{'type': 'Mem', 'tota_l': '3861\n332', 'used': '2228\n20', 'fr_ee': '3364\n176', 'shar_ed': '1183\n2', 'buff_cac_he': '2743\n36', 'avai_labl_e': '3389\n588'}, {'type': 'Swap', 'tota_l': '2097\n148', 'used': '0', 'fr_ee': '2097\n148\nkb', 'shar_ed': None, 'buff_cac_he': None, 'avai_labl_e': None}, {'type': 'last', 'tota_l': 'last', 'used': 'last', 'fr_ee': 'ab cde', 'shar_ed': None, 'buff_cac_he': None, 'avai_labl_e': 'final'}]
        self.assertEqual(jc.parsers.asciitable_m.parse(input, quiet=True), expected)

    def test_asciitable_m_pure_ascii_extra_spaces(self):
        if False:
            return 10
        "\n        Test 'asciitable_m' with a pure ASCII table that has heading and\n        trailing spaces and newlines.\n        "
        input = '\n    \n      \n    +========+========+========+========+========+========+========+\n    | type   | tota   | used   | fr ee  | shar   | buff   | avai  \n    |        | l      |        |        | ed     | _cac   | labl         \n    |        |        |        |        |        | he     | e      |\n    +========+========+========+========+========+========+========+   \n    | Mem    | 3861   | 2228   | 3364   | 1183   | 2743   | 3389   |\n    |        | 332    | 20     | 176    | 2      | 36     | 588    |\n    +--------+--------+--------+--------+--------+--------+--------+\n    |        |        |        |        |        |        |        |\n    |        |        |        |        | test 2 |        |        |     \n    +--------+--------+--------+--------+--------+--------+--------+\n    | last   | last   | last   | ab cde |        |        | final     \n    +========+========+========+========+========+========+========+    \n     \n  \n        '
        expected = [{'type': 'Mem', 'tota_l': '3861\n332', 'used': '2228\n20', 'fr_ee': '3364\n176', 'shar_ed': '1183\n2', 'buff_cac_he': '2743\n36', 'avai_labl_e': '3389\n588'}, {'type': None, 'tota_l': None, 'used': None, 'fr_ee': None, 'shar_ed': 'test 2', 'buff_cac_he': None, 'avai_labl_e': None}, {'type': 'last', 'tota_l': 'last', 'used': 'last', 'fr_ee': 'ab cde', 'shar_ed': None, 'buff_cac_he': None, 'avai_labl_e': 'final'}]
        self.assertEqual(jc.parsers.asciitable_m.parse(input, quiet=True), expected)

    def test_asciitable_m_unicode_extra_spaces(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'asciitable_m' with a pure ASCII table that has heading and\n        trailing spaces and newlines.\n        "
        input = '\n    \n  \n        ╒════════╤════════╤════════╤════════╤════════╤════════╤════════╕\n          type   │ tota   │ used   │ free   │ shar   │ buff   │ avai   \n                 │ l      │        │        │ ed     │ _cac   │ labl   \n                 │        │        │        │        │ he     │ e        \n        ╞════════╪════════╪════════╪════════╪════════╪════════╪════════╡      \n          Mem    │ 3861   │ 2228   │ 3364   │ 1183   │ 2743   │ 3389   \n                 │ 332    │ 20     │ 176    │ 2      │ 36     │ 588  \n        ├────────┼────────┼────────┼────────┼────────┼────────┼────────┤  \n          Swap   │ 2097   │ 0      │ 2097   │        │        │            \n                 │ 148    │        │ 148    │        │        │        \n        ╘════════╧════════╧════════╧════════╧════════╧════════╧════════╛\n   \n \n        '
        expected = [{'type': 'Mem', 'tota_l': '3861\n332', 'used': '2228\n20', 'free': '3364\n176', 'shar_ed': '1183\n2', 'buff_cac_he': '2743\n36', 'avai_labl_e': '3389\n588'}, {'type': 'Swap', 'tota_l': '2097\n148', 'used': '0', 'free': '2097\n148', 'shar_ed': None, 'buff_cac_he': None, 'avai_labl_e': None}]
        self.assertEqual(jc.parsers.asciitable_m.parse(input, quiet=True), expected)

    def test_asciitable_m_pretty_ansi(self):
        if False:
            while True:
                i = 10
        "\n        Test 'asciitable-m' with a pretty table with ANSI codes\n        "
        input = '\n┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓                                   \n┃\x1b[1m \x1b[0m\x1b[1mReleased    \x1b[0m\x1b[1m \x1b[0m┃\x1b[1m \x1b[0m\x1b[1mTitle                            \x1b[0m\x1b[1m \x1b[0m┃\x1b[1m \x1b[0m\x1b[1m    Box Office\x1b[0m\x1b[1m \x1b[0m┃                                   \n┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩                                   \n│\x1b[36m \x1b[0m\x1b[36mDec 20, 2019\x1b[0m\x1b[36m \x1b[0m│\x1b[35m \x1b[0m\x1b[35mStar Wars: The Rise of Skywalker \x1b[0m\x1b[35m \x1b[0m│\x1b[32m \x1b[0m\x1b[32m  $952,110,690\x1b[0m\x1b[32m \x1b[0m│                                   \n│\x1b[36m \x1b[0m\x1b[36mMay 25, 2018\x1b[0m\x1b[36m \x1b[0m│\x1b[35m \x1b[0m\x1b[35mSolo: A Star Wars Story          \x1b[0m\x1b[35m \x1b[0m│\x1b[32m \x1b[0m\x1b[32m  $393,151,347\x1b[0m\x1b[32m \x1b[0m│                                   \n│\x1b[36m \x1b[0m\x1b[36mDec 15, 2017\x1b[0m\x1b[36m \x1b[0m│\x1b[35m \x1b[0m\x1b[35mStar Wars Ep. V111: The Last Jedi\x1b[0m\x1b[35m \x1b[0m│\x1b[32m \x1b[0m\x1b[32m$1,332,539,889\x1b[0m\x1b[32m \x1b[0m│                                   \n│\x1b[36m \x1b[0m\x1b[36mDec 16, 2016\x1b[0m\x1b[36m \x1b[0m│\x1b[35m \x1b[0m\x1b[35mRogue One: A Star Wars Story     \x1b[0m\x1b[35m \x1b[0m│\x1b[32m \x1b[0m\x1b[32m$1,332,439,889\x1b[0m\x1b[32m \x1b[0m│                                   \n└──────────────┴───────────────────────────────────┴────────────────┘                                   \n'
        expected = [{'released': 'Dec 20, 2019\nMay 25, 2018\nDec 15, 2017\nDec 16, 2016', 'title': 'Star Wars: The Rise of Skywalker\nSolo: A Star Wars Story\nStar Wars Ep. V111: The Last Jedi\nRogue One: A Star Wars Story', 'box_office': '$952,110,690\n$393,151,347\n$1,332,539,889\n$1,332,439,889'}]
        self.assertEqual(jc.parsers.asciitable_m.parse(input, quiet=True), expected)

    def test_asciitable_m_special_chars_in_header(self):
        if False:
            while True:
                i = 10
        "\n        Test 'asciitable_m' with a pure ASCII table that has special\n        characters in the header. These should be converted to underscores\n        and no trailing or consecutive underscores should end up in the\n        resulting key names.\n        "
        input = '\n+----------+------------+-----------+----------------+-------+--------------------+\n| Protocol | Address    | Age (min) | Hardware Addr  | Type  | Interface          |\n|          |            | of int    |                |       |                    |\n+----------+------------+-----------+----------------+-------+--------------------+\n| Internet | 10.12.13.1 |       98  | 0950.5785.5cd1 | ARPA  | FastEthernet2.13   |\n+----------+------------+-----------+----------------+-------+--------------------+\n        '
        expected = [{'protocol': 'Internet', 'address': '10.12.13.1', 'age_min_of_int': '98', 'hardware_addr': '0950.5785.5cd1', 'type': 'ARPA', 'interface': 'FastEthernet2.13'}]
        self.assertEqual(jc.parsers.asciitable_m.parse(input, quiet=True), expected)

    def test_asciitable_no_lower_raw(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'asciitable_m' with a pure ASCII table that has special\n        characters and mixed case in the header. These should be converted to underscores\n        and no trailing or consecutive underscores should end up in the\n        resulting key names. Using `raw` in this test to preserve case. (no lower)\n        "
        input = '\n+----------+------------+-----------+----------------+-------+--------------------+\n| Protocol | Address    | Age (min) | Hardware Addr  | Type  | Interface          |\n|          |            | of int    |                |       |                    |\n+----------+------------+-----------+----------------+-------+--------------------+\n| Internet | 10.12.13.1 |       98  | 0950.5785.5cd1 | ARPA  | FastEthernet2.13   |\n+----------+------------+-----------+----------------+-------+--------------------+\n        '
        expected = [{'Protocol': 'Internet', 'Address': '10.12.13.1', 'Age_min_of_int': '98', 'Hardware_Addr': '0950.5785.5cd1', 'Type': 'ARPA', 'Interface': 'FastEthernet2.13'}]
        self.assertEqual(jc.parsers.asciitable_m.parse(input, raw=True, quiet=True), expected)

    def test_asciitable_m_sep_char_in_cell(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'asciitable_m' with a column separator character inside the data\n        "
        input = '\n| Author          | yada        | yada2           | yada3           | yada4           | yada5           | yada6      | yada7           |\n├─────────────────┼─────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼────────────┼─────────────────┤\n│ Kelly Brazil    │             │ a76d46f9ecb1eff │ kellyjonbrazil@ │ Fri Feb 4 12:14 │ refactor ignore │ 1644005656 │                 │\n│                 │             │ 4d6cc7ad633c97c │ gmail.com       │ :16 2022 -0800  │ _exceptions     │            │                 │\n│                 │             │ ec0e99001a      │                 │                 │                 │            │                 │\n├─────────────────┼─────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼────────────┼─────────────────┤\n│ Kevin Lyter     │             │ 6b069a82d0fa19c │ lyterk@sent.com │ Thu Feb 3 18:13 │ Add xrandr to l │ 1643940838 │                 │\n│                 │             │ 8d83b19b934bace │                 │ :58 2022 -0800  │ ib.py           │            │                 │\n│                 │             │ 556cb758d7      │                 │                 │                 │            │                 │\n├─────────────────┼─────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼────────────┼─────────────────┤\n│ Kevin Lyter     │             │ 6b793d052147406 │ lyterk@sent.com │ Thu Feb 3 18:13 │ Clean up types  │ 1643940791 │                 │\n│                 │             │ f388c4d5dc04f50 │                 │ :11 2022 -0800  │                 │            │                 │\n│                 │             │ 6a3456f409      │                 │                 │                 │            │                 │\n│                 │             │                 │                 │                 │ * | operator =  │            │                 │\n│                 │             │                 │                 │                 │ > Union[]       │            │                 │\n│                 │             │                 │                 │                 │ * Rem           │            │                 │\n│                 │             │                 │                 │                 │ ove unused impo │            │                 │\n│                 │             │                 │                 │                 │ rt Iterator     │            │                 │\n│                 │             │                 │                 │                 │ * R             │            │                 │\n│                 │             │                 │                 │                 │ emove comment   │            │                 │\n├─────────────────┼─────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼────────────┼─────────────────┤\n│ Kevin Lyter     │             │ ce9103f7cc66689 │ lyterk@sent.com │ Thu Feb 3 18:12 │ Delete old file │ 1643940766 │                 │\n│                 │             │ 5dc7840d32797d8 │                 │ :46 2022 -0800  │ s in template f │            │                 │\n│                 │             │ c7274cf1b8      │                 │                 │ older           │            │                 │\n├─────────────────┼─────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼────────────┼─────────────────┤\n        '
        expected = [{'author': 'Kelly Brazil', 'yada': None, 'yada2': 'a76d46f9ecb1eff\n4d6cc7ad633c97c\nec0e99001a', 'yada3': 'kellyjonbrazil@\ngmail.com', 'yada4': 'Fri Feb 4 12:14\n:16 2022 -0800', 'yada5': 'refactor ignore\n_exceptions', 'yada6': '1644005656', 'yada7': None}, {'author': 'Kevin Lyter', 'yada': None, 'yada2': '6b069a82d0fa19c\n8d83b19b934bace\n556cb758d7', 'yada3': 'lyterk@sent.com', 'yada4': 'Thu Feb 3 18:13\n:58 2022 -0800', 'yada5': 'Add xrandr to l\nib.py', 'yada6': '1643940838', 'yada7': None}, {'author': 'Kevin Lyter', 'yada': None, 'yada2': 'ce9103f7cc66689\n5dc7840d32797d8\nc7274cf1b8', 'yada3': 'lyterk@sent.com', 'yada4': 'Thu Feb 3 18:12\n:46 2022 -0800', 'yada5': 'Delete old file\ns in template f\nolder', 'yada6': '1643940766', 'yada7': None}]
        self.assertEqual(jc.parsers.asciitable_m.parse(input, quiet=True), expected)

    def test_asciitable_m_markdown(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'asciitable_m' with a markdown table. Should raise a ParseError\n        "
        input = '\n        | type   |   total |   used |    free |   shared |   buff cache |   available |\n        |--------|---------|--------|---------|----------|--------------|-------------|\n        | Mem    | 3861332 | 222820 | 3364176 |    11832 |       274336 |     3389588 |\n        | Swap   | 2097148 |      0 | 2097148 |          |              |             |\n        '
        self.assertRaises(ParseError, jc.parsers.asciitable_m.parse, input, quiet=True)

    def test_asciitable_m_simple(self):
        if False:
            return 10
        "\n        Test 'asciitable_m' with a simple table. Should raise a ParseError\n        "
        input = '\n        type      total    used     free    shared    buff cache    available\n        ------  -------  ------  -------  --------  ------------  -----------\n        Mem     3861332  222820  3364176     11832        274336      3389588\n        Swap    2097148       0  2097148\n        '
        self.assertRaises(ParseError, jc.parsers.asciitable_m.parse, input, quiet=True)
if __name__ == '__main__':
    unittest.main()