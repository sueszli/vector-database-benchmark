from runner.koan import *
import re

class AboutRegex(Koan):
    """
        These koans are based on Ben's book: Regular Expressions in 10
        minutes. I found this book very useful, so I decided to write
        a koan file in order to practice everything it taught me.
        http://www.forta.com/books/0672325667/
    """

    def test_matching_literal_text(self):
        if False:
            while True:
                i = 10
        '\n            Lesson 1 Matching Literal String\n        '
        string = 'Hello, my name is Felix and these koans are based ' + "on Ben's book: Regular Expressions in 10 minutes."
        m = re.search(__, string)
        self.assertTrue(m and m.group(0) and (m.group(0) == 'Felix'), 'I want my name')

    def test_matching_literal_text_how_many(self):
        if False:
            for i in range(10):
                print('nop')
        '\n            Lesson 1 -- How many matches?\n\n            The default behaviour of most regular expression engines is\n            to return just the first match. In python you have the\n            following options:\n\n                match()    -->  Determine if the RE matches at the\n                                beginning of the string.\n                search()   -->  Scan through a string, looking for any\n                                location where this RE matches.\n                findall()  -->  Find all substrings where the RE\n                                matches, and return them as a list.\n                finditer() -->  Find all substrings where the RE\n                                matches, and return them as an iterator.\n        '
        string = 'Hello, my name is Felix and these koans are based ' + "on Ben's book: Regular Expressions in 10 minutes. " + 'Repeat My name is Felix'
        m = re.match('Felix', string)
        self.assertEqual(m, __)

    def test_matching_literal_text_not_case_sensitivity(self):
        if False:
            i = 10
            return i + 15
        "\n            Lesson 1 -- Matching Literal String non case sensitivity.\n            Most regex implementations also support matches that are not\n            case sensitive. In python you can use re.IGNORECASE, in\n            Javascript you can specify the optional i flag. In Ben's\n            book you can see more languages.\n\n        "
        string = 'Hello, my name is Felix or felix and this koan ' + "is based on Ben's book: Regular Expressions in 10 minutes."
        self.assertEqual(re.findall('felix', string), __)
        self.assertEqual(re.findall('felix', string, re.IGNORECASE), __)

    def test_matching_any_character(self):
        if False:
            print('Hello World!')
        '\n            Lesson 1: Matching any character\n\n            `.` matches any character: alphabetic characters, digits,\n            and punctuation.\n        '
        string = 'pecks.xlx\n' + 'orders1.xls\n' + 'apec1.xls\n' + 'na1.xls\n' + 'na2.xls\n' + 'sa1.xls'
        change_this_search_string = 'a..xlx'
        self.assertEquals(len(re.findall(change_this_search_string, string)), 3)

    def test_matching_set_character(self):
        if False:
            return 10
        '\n            Lesson 2 -- Matching sets of characters\n\n            A set of characters is defined using the metacharacters\n            `[` and `]`. Everything between them is part of the set, and\n            any single one of the set members will match.\n        '
        string = 'sales.xlx\n' + 'sales1.xls\n' + 'orders3.xls\n' + 'apac1.xls\n' + 'sales2.xls\n' + 'na1.xls\n' + 'na2.xls\n' + 'sa1.xls\n' + 'ca1.xls'
        change_this_search_string = '[nsc]a[2-9].xls'
        self.assertEquals(len(re.findall(change_this_search_string, string)), 3)

    def test_anything_but_matching(self):
        if False:
            while True:
                i = 10
        "\n            Lesson 2 -- Using character set ranges\n            Occasionally, you'll have a list of characters that you don't\n            want to match. Character sets can be negated using the ^\n            metacharacter.\n\n        "
        string = 'sales.xlx\n' + 'sales1.xls\n' + 'orders3.xls\n' + 'apac1.xls\n' + 'sales2.xls\n' + 'sales3.xls\n' + 'europe2.xls\n' + 'sam.xls\n' + 'na1.xls\n' + 'na2.xls\n' + 'sa1.xls\n' + 'ca1.xls'
        change_this_search_string = '[^nc]am'
        self.assertEquals(re.findall(change_this_search_string, string), ['sam.xls'])