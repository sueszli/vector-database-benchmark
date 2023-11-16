import os
import tempfile
import unittest
from collections import OrderedDict
import logging
from coalib.parsing.ConfParser import ConfParser
from coalib.settings.Section import Section

class ConfParserTest(unittest.TestCase):
    example_file = 'setting = without_section\n    [foo]\n    to be ignored\n    a_default, another = val\n    TEST = tobeignored  # do you know that thats a comment\n    test = push\n    t =\n    escaped_\\=equal = escaped_\\#hash\n    escaped_\\\\backslash = escaped_\\ space\n    escaped_\\,comma = escaped_\\.dot\n    [MakeFiles]\n     j  , another = a\n                   multiline\n                   value\n    # just a comment\n    # just a comment\n    nokey. = value\n    foo.test = content\n    makefiles.lastone = val\n    append += key\n\n    [EMPTY_ELEM_STRIP]\n    A = a, b, c\n    B = a, ,, d\n    C = ,,,\n\n    [name]\n    key1 = value1\n    key2 = value1\n    key1 = value2\n    '

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tempdir = tempfile.gettempdir()
        self.file = os.path.join(self.tempdir, '.coafile')
        self.nonexistentfile = os.path.join(self.tempdir, 'e81k7bd98t')
        with open(self.file, 'w') as file:
            file.write(self.example_file)
        self.uut = ConfParser()
        try:
            os.remove(self.nonexistentfile)
        except FileNotFoundError:
            pass
        logger = logging.getLogger()
        with self.assertLogs(logger, 'WARNING') as self.cm:
            self.sections = self.uut.parse(self.file)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        os.remove(self.file)

    def test_warning_typo(self):
        if False:
            return 10
        logger = logging.getLogger()
        with self.assertLogs(logger, 'WARNING') as cm:
            newConf = ConfParser(comment_seperators=('#',))
            self.assertEquals(cm.output[0], 'WARNING:root:The setting `comment_seperators` is deprecated. Please use `comment_separators` instead.')

    def test_parse_nonexisting_file(self):
        if False:
            return 10
        self.assertRaises(FileNotFoundError, self.uut.parse, self.nonexistentfile)
        self.assertNotEqual(self.uut.parse(self.file, True), self.sections)

    def test_parse_nonexisting_section(self):
        if False:
            while True:
                i = 10
        self.assertRaises(IndexError, self.uut.get_section, 'non-existent section')

    def test_parse_default_section_deprecated(self):
        if False:
            while True:
                i = 10
        default_should = OrderedDict([('setting', 'without_section')])
        (key, val) = self.sections.popitem(last=False)
        self.assertTrue(isinstance(val, Section))
        self.assertEqual(key, 'default')
        is_dict = OrderedDict()
        for k in val:
            is_dict[k] = str(val[k])
        self.assertEqual(is_dict, default_should)
        self.assertRegex(self.cm.output[0], 'A setting does not have a section.')
        line_num = val.contents['setting'].line_number
        self.assertEqual(line_num, 1)

    def test_parse_foo_section(self):
        if False:
            print('Hello World!')
        foo_should = OrderedDict([('a_default', 'val'), ('another', 'val'), ('comment0', '# do you know that thats a comment'), ('test', 'content'), ('t', ''), ('escaped_=equal', 'escaped_#hash'), ('escaped_\\backslash', 'escaped_ space'), ('escaped_,comma', 'escaped_.dot')])
        self.sections.popitem(last=False)
        (key, val) = self.sections.popitem(last=False)
        self.assertTrue(isinstance(val, Section))
        self.assertEqual(key, 'foo')
        is_dict = OrderedDict()
        for k in val:
            is_dict[k] = str(val[k])
        self.assertEqual(is_dict, foo_should)

    def test_parse_makefiles_section(self):
        if False:
            print('Hello World!')
        makefiles_should = OrderedDict([('j', 'a\nmultiline\nvalue'), ('another', 'a\nmultiline\nvalue'), ('comment1', '# just a comment'), ('comment2', '# just a comment'), ('lastone', 'val'), ('append', 'key'), ('comment3', '')])
        self.sections.popitem(last=False)
        self.sections.popitem(last=False)
        (key, val) = self.sections.popitem(last=False)
        self.assertTrue(isinstance(val, Section))
        self.assertEqual(key, 'makefiles')
        is_dict = OrderedDict()
        for k in val:
            is_dict[k] = str(val[k])
        self.assertEqual(is_dict, makefiles_should)
        self.assertEqual(val['comment1'].key, 'comment1')
        line_num = val.contents['another'].line_number
        self.assertEqual(line_num, 12)
        line_num = val.contents['append'].line_number
        self.assertEqual(line_num, 20)
        line_num = val.contents['another'].end_line_number
        self.assertEqual(line_num, 14)

    def test_parse_empty_elem_strip_section(self):
        if False:
            print('Hello World!')
        empty_elem_strip_should = OrderedDict([('a', 'a, b, c'), ('b', 'a, ,, d'), ('c', ',,,'), ('comment4', '')])
        self.sections.popitem(last=False)
        self.sections.popitem(last=False)
        self.sections.popitem(last=False)
        (key, val) = self.sections.popitem(last=False)
        self.assertTrue(isinstance(val, Section))
        self.assertEqual(key, 'empty_elem_strip')
        is_dict = OrderedDict()
        for k in val:
            is_dict[k] = str(val[k])
        self.assertEqual(is_dict, empty_elem_strip_should)
        line_num = val.contents['b'].line_number
        self.assertEqual(line_num, 24)

    def test_line_number_name_section(self):
        if False:
            return 10
        self.sections.popitem(last=False)
        self.sections.popitem(last=False)
        self.sections.popitem(last=False)
        self.sections.popitem(last=False)
        (key, val) = self.sections.popitem(last=False)
        line_num = val.contents['key1'].line_number
        self.assertEqual(line_num, 30)
        line_num = val.contents['key1'].end_line_number
        self.assertEqual(line_num, 30)

    def test_remove_empty_iter_elements(self):
        if False:
            return 10
        uut = ConfParser(remove_empty_iter_elements=True)
        uut.parse(self.file)
        self.assertEqual(list(uut.get_section('EMPTY_ELEM_STRIP')['A']), ['a', 'b', 'c'])
        self.assertEqual(list(uut.get_section('EMPTY_ELEM_STRIP')['B']), ['a', 'd'])
        self.assertEqual(list(uut.get_section('EMPTY_ELEM_STRIP')['C']), [])
        uut = ConfParser(remove_empty_iter_elements=False)
        uut.parse(self.file)
        self.assertEqual(list(uut.get_section('EMPTY_ELEM_STRIP')['A']), ['a', 'b', 'c'])
        self.assertEqual(list(uut.get_section('EMPTY_ELEM_STRIP')['B']), ['a', '', '', 'd'])
        self.assertEqual(list(uut.get_section('EMPTY_ELEM_STRIP')['C']), ['', '', '', ''])

    def test_config_directory(self):
        if False:
            print('Hello World!')
        self.uut.parse(self.tempdir)

    def test_settings_override_warning(self):
        if False:
            return 10
        self.assertEqual(self.cm.output[1], 'WARNING:root:test setting has already been defined in section foo. The previous setting will be overridden.')
        self.assertEqual(self.cm.output[2], 'WARNING:root:key1 setting has already been defined in section name. The previous setting will be overridden.')