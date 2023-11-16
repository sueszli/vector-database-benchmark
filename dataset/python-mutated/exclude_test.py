import io
from xml.etree import ElementTree as ET
from hscommon.testutil import eq_
from hscommon.plat import ISWINDOWS
from core.tests.base import DupeGuru
from core.exclude import ExcludeList, ExcludeDict, default_regexes, AlreadyThereException
from re import error

class TestCaseListXMLLoading:

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        self.exclude_list = ExcludeList()

    def test_load_non_existant_file(self):
        if False:
            for i in range(10):
                print('nop')
        self.exclude_list.load_from_xml('non_existant.xml')
        eq_(len(default_regexes), len(self.exclude_list))
        eq_(len(default_regexes), self.exclude_list.marked_count)

    def test_save_to_xml(self):
        if False:
            return 10
        f = io.BytesIO()
        self.exclude_list.save_to_xml(f)
        f.seek(0)
        doc = ET.parse(f)
        root = doc.getroot()
        eq_('exclude_list', root.tag)

    def test_save_and_load(self, tmpdir):
        if False:
            while True:
                i = 10
        e1 = ExcludeList()
        e2 = ExcludeList()
        eq_(len(e1), 0)
        e1.add('one')
        e1.mark('one')
        e1.add('two')
        tmpxml = str(tmpdir.join('exclude_testunit.xml'))
        e1.save_to_xml(tmpxml)
        e2.load_from_xml(tmpxml)
        assert 'one' in e2
        assert 'two' in e2
        eq_(len(e2), 2)
        eq_(e2.marked_count, 1)

    def test_load_xml_with_garbage_and_missing_elements(self):
        if False:
            for i in range(10):
                print('nop')
        root = ET.Element('foobar')
        exclude_node = ET.SubElement(root, 'bogus')
        exclude_node.set('regex', 'None')
        exclude_node.set('marked', 'y')
        exclude_node = ET.SubElement(root, 'exclude')
        exclude_node.set('regex', 'one')
        exclude_node.set('markedddd', 'y')
        exclude_node = ET.SubElement(root, 'exclude')
        exclude_node.set('regex', 'two')
        exclude_node = ET.SubElement(root, 'exclude')
        exclude_node.set('regex', 'three')
        exclude_node.set('markedddd', 'pazjbjepo')
        f = io.BytesIO()
        tree = ET.ElementTree(root)
        tree.write(f, encoding='utf-8')
        f.seek(0)
        self.exclude_list.load_from_xml(f)
        print(f'{[x for x in self.exclude_list]}')
        eq_(3, len(self.exclude_list))
        eq_(0, self.exclude_list.marked_count)

class TestCaseDictXMLLoading(TestCaseListXMLLoading):

    def setup_method(self, method):
        if False:
            print('Hello World!')
        self.exclude_list = ExcludeDict()

class TestCaseListEmpty:

    def setup_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        self.app = DupeGuru()
        self.app.exclude_list = ExcludeList(union_regex=False)
        self.exclude_list = self.app.exclude_list

    def test_add_mark_and_remove_regex(self):
        if False:
            return 10
        regex1 = 'one'
        regex2 = 'two'
        self.exclude_list.add(regex1)
        assert regex1 in self.exclude_list
        self.exclude_list.add(regex2)
        self.exclude_list.mark(regex1)
        self.exclude_list.mark(regex2)
        eq_(len(self.exclude_list), 2)
        eq_(len(self.exclude_list.compiled), 2)
        compiled_files = [x for x in self.exclude_list.compiled_files]
        eq_(len(compiled_files), 2)
        self.exclude_list.remove(regex2)
        assert regex2 not in self.exclude_list
        eq_(len(self.exclude_list), 1)

    def test_add_duplicate(self):
        if False:
            return 10
        self.exclude_list.add('one')
        eq_(1, len(self.exclude_list))
        try:
            self.exclude_list.add('one')
        except Exception:
            pass
        eq_(1, len(self.exclude_list))

    def test_add_not_compilable(self):
        if False:
            for i in range(10):
                print('nop')
        regex = 'one))'
        try:
            self.exclude_list.add(regex)
        except Exception as e:
            eq_(type(e), error)
        added = self.exclude_list.mark(regex)
        eq_(added, False)
        eq_(len(self.exclude_list), 0)
        eq_(len(self.exclude_list.compiled), 0)
        compiled_files = [x for x in self.exclude_list.compiled_files]
        eq_(len(compiled_files), 0)

    def test_force_add_not_compilable(self):
        if False:
            for i in range(10):
                print('nop')
        'Used when loading from XML for example'
        regex = 'one))'
        self.exclude_list.add(regex, forced=True)
        marked = self.exclude_list.mark(regex)
        eq_(marked, False)
        eq_(len(self.exclude_list), 1)
        eq_(len(self.exclude_list.compiled), 0)
        compiled_files = [x for x in self.exclude_list.compiled_files]
        eq_(len(compiled_files), 0)
        regex = 'one))'
        try:
            self.exclude_list.add(regex, forced=True)
        except Exception as e:
            assert type(e) is AlreadyThereException
        eq_(len(self.exclude_list), 1)
        eq_(len(self.exclude_list.compiled), 0)

    def test_rename_regex(self):
        if False:
            while True:
                i = 10
        regex = 'one'
        self.exclude_list.add(regex)
        self.exclude_list.mark(regex)
        regex_renamed = 'one))'
        self.exclude_list.rename(regex, regex_renamed)
        assert regex not in self.exclude_list
        assert regex_renamed in self.exclude_list
        eq_(self.exclude_list.is_marked(regex_renamed), False)
        self.exclude_list.mark(regex_renamed)
        eq_(self.exclude_list.is_marked(regex_renamed), False)
        regex_renamed_compilable = 'two'
        self.exclude_list.rename(regex_renamed, regex_renamed_compilable)
        assert regex_renamed_compilable in self.exclude_list
        eq_(self.exclude_list.is_marked(regex_renamed), False)
        self.exclude_list.mark(regex_renamed_compilable)
        eq_(self.exclude_list.is_marked(regex_renamed_compilable), True)
        eq_(len(self.exclude_list), 1)
        regex_compilable = 'three'
        self.exclude_list.rename(regex_renamed_compilable, regex_compilable)
        eq_(self.exclude_list.is_marked(regex_compilable), True)

    def test_rename_regex_file_to_path(self):
        if False:
            while True:
                i = 10
        regex = '.*/one.*'
        if ISWINDOWS:
            regex = '.*\\\\one.*'
        regex2 = '.*one.*'
        self.exclude_list.add(regex)
        self.exclude_list.mark(regex)
        compiled_re = [x.pattern for x in self.exclude_list._excluded_compiled]
        files_re = [x.pattern for x in self.exclude_list.compiled_files]
        paths_re = [x.pattern for x in self.exclude_list.compiled_paths]
        assert regex in compiled_re
        assert regex not in files_re
        assert regex in paths_re
        self.exclude_list.rename(regex, regex2)
        compiled_re = [x.pattern for x in self.exclude_list._excluded_compiled]
        files_re = [x.pattern for x in self.exclude_list.compiled_files]
        paths_re = [x.pattern for x in self.exclude_list.compiled_paths]
        assert regex not in compiled_re
        assert regex2 in compiled_re
        assert regex2 in files_re
        assert regex2 not in paths_re

    def test_restore_default(self):
        if False:
            while True:
                i = 10
        'Only unmark previously added regexes and mark the pre-defined ones'
        regex = 'one'
        self.exclude_list.add(regex)
        self.exclude_list.mark(regex)
        self.exclude_list.restore_defaults()
        eq_(len(default_regexes), self.exclude_list.marked_count)
        eq_(self.exclude_list.is_marked(regex), False)
        compiled = [x for x in self.exclude_list.compiled]
        assert regex not in compiled
        for re in default_regexes:
            assert self.exclude_list.is_marked(re)
            found = False
            for compiled_re in compiled:
                if compiled_re.pattern == re:
                    found = True
            if not found:
                raise Exception(f'Default RE {re} not found in compiled list.')
        eq_(len(default_regexes), len(self.exclude_list.compiled))

class TestCaseListEmptyUnion(TestCaseListEmpty):
    """Same but with union regex"""

    def setup_method(self, method):
        if False:
            print('Hello World!')
        self.app = DupeGuru()
        self.app.exclude_list = ExcludeList(union_regex=True)
        self.exclude_list = self.app.exclude_list

    def test_add_mark_and_remove_regex(self):
        if False:
            while True:
                i = 10
        regex1 = 'one'
        regex2 = 'two'
        self.exclude_list.add(regex1)
        assert regex1 in self.exclude_list
        self.exclude_list.add(regex2)
        self.exclude_list.mark(regex1)
        self.exclude_list.mark(regex2)
        eq_(len(self.exclude_list), 2)
        eq_(len(self.exclude_list.compiled), 1)
        compiled_files = [x for x in self.exclude_list.compiled_files]
        eq_(len(compiled_files), 1)
        assert '|' in compiled_files[0].pattern
        self.exclude_list.remove(regex2)
        assert regex2 not in self.exclude_list
        eq_(len(self.exclude_list), 1)

    def test_rename_regex_file_to_path(self):
        if False:
            print('Hello World!')
        regex = '.*/one.*'
        if ISWINDOWS:
            regex = '.*\\\\one.*'
        regex2 = '.*one.*'
        self.exclude_list.add(regex)
        self.exclude_list.mark(regex)
        eq_(len([x for x in self.exclude_list]), 1)
        compiled_re = [x.pattern for x in self.exclude_list.compiled]
        files_re = [x.pattern for x in self.exclude_list.compiled_files]
        paths_re = [x.pattern for x in self.exclude_list.compiled_paths]
        assert regex in compiled_re
        assert regex not in files_re
        assert regex in paths_re
        self.exclude_list.rename(regex, regex2)
        eq_(len([x for x in self.exclude_list]), 1)
        compiled_re = [x.pattern for x in self.exclude_list.compiled]
        files_re = [x.pattern for x in self.exclude_list.compiled_files]
        paths_re = [x.pattern for x in self.exclude_list.compiled_paths]
        assert regex not in compiled_re
        assert regex2 in compiled_re
        assert regex2 in files_re
        assert regex2 not in paths_re

    def test_restore_default(self):
        if False:
            return 10
        'Only unmark previously added regexes and mark the pre-defined ones'
        regex = 'one'
        self.exclude_list.add(regex)
        self.exclude_list.mark(regex)
        self.exclude_list.restore_defaults()
        eq_(len(default_regexes), self.exclude_list.marked_count)
        eq_(self.exclude_list.is_marked(regex), False)
        compiled = [x for x in self.exclude_list.compiled]
        assert regex not in compiled
        compiled_escaped = {x.encode('unicode-escape').decode() for x in compiled[0].pattern.split('|')}
        default_escaped = {x.encode('unicode-escape').decode() for x in default_regexes}
        assert compiled_escaped == default_escaped
        eq_(len(default_regexes), len(compiled[0].pattern.split('|')))

class TestCaseDictEmpty(TestCaseListEmpty):
    """Same, but with dictionary implementation"""

    def setup_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        self.app = DupeGuru()
        self.app.exclude_list = ExcludeDict(union_regex=False)
        self.exclude_list = self.app.exclude_list

class TestCaseDictEmptyUnion(TestCaseDictEmpty):
    """Same, but with union regex"""

    def setup_method(self, method):
        if False:
            return 10
        self.app = DupeGuru()
        self.app.exclude_list = ExcludeDict(union_regex=True)
        self.exclude_list = self.app.exclude_list

    def test_add_mark_and_remove_regex(self):
        if False:
            for i in range(10):
                print('nop')
        regex1 = 'one'
        regex2 = 'two'
        self.exclude_list.add(regex1)
        assert regex1 in self.exclude_list
        self.exclude_list.add(regex2)
        self.exclude_list.mark(regex1)
        self.exclude_list.mark(regex2)
        eq_(len(self.exclude_list), 2)
        eq_(len(self.exclude_list.compiled), 1)
        compiled_files = [x for x in self.exclude_list.compiled_files]
        eq_(len(compiled_files), 1)
        self.exclude_list.remove(regex2)
        assert regex2 not in self.exclude_list
        eq_(len(self.exclude_list), 1)

    def test_rename_regex_file_to_path(self):
        if False:
            return 10
        regex = '.*/one.*'
        if ISWINDOWS:
            regex = '.*\\\\one.*'
        regex2 = '.*one.*'
        self.exclude_list.add(regex)
        self.exclude_list.mark(regex)
        marked_re = [x for (marked, x) in self.exclude_list if marked]
        eq_(len(marked_re), 1)
        compiled_re = [x.pattern for x in self.exclude_list.compiled]
        files_re = [x.pattern for x in self.exclude_list.compiled_files]
        paths_re = [x.pattern for x in self.exclude_list.compiled_paths]
        assert regex in compiled_re
        assert regex not in files_re
        assert regex in paths_re
        self.exclude_list.rename(regex, regex2)
        compiled_re = [x.pattern for x in self.exclude_list.compiled]
        files_re = [x.pattern for x in self.exclude_list.compiled_files]
        paths_re = [x.pattern for x in self.exclude_list.compiled_paths]
        assert regex not in compiled_re
        assert regex2 in compiled_re
        assert regex2 in files_re
        assert regex2 not in paths_re

    def test_restore_default(self):
        if False:
            print('Hello World!')
        'Only unmark previously added regexes and mark the pre-defined ones'
        regex = 'one'
        self.exclude_list.add(regex)
        self.exclude_list.mark(regex)
        self.exclude_list.restore_defaults()
        eq_(len(default_regexes), self.exclude_list.marked_count)
        eq_(self.exclude_list.is_marked(regex), False)
        compiled = [x for x in self.exclude_list.compiled]
        assert regex not in compiled
        compiled_escaped = {x.encode('unicode-escape').decode() for x in compiled[0].pattern.split('|')}
        default_escaped = {x.encode('unicode-escape').decode() for x in default_regexes}
        assert compiled_escaped == default_escaped
        eq_(len(default_regexes), len(compiled[0].pattern.split('|')))

def split_union(pattern_object):
    if False:
        i = 10
        return i + 15
    'Returns list of strings for each union pattern'
    return [x for x in pattern_object.pattern.split('|')]

class TestCaseCompiledList:
    """Test consistency between union or and separate versions."""

    def setup_method(self, method):
        if False:
            print('Hello World!')
        self.e_separate = ExcludeList(union_regex=False)
        self.e_separate.restore_defaults()
        self.e_union = ExcludeList(union_regex=True)
        self.e_union.restore_defaults()

    def test_same_number_of_expressions(self):
        if False:
            print('Hello World!')
        eq_(len(split_union(self.e_union.compiled[0])), len(default_regexes))
        eq_(len(self.e_separate.compiled), len(default_regexes))
        exprs = split_union(self.e_union.compiled[0])
        eq_(len(exprs), len(self.e_separate.compiled))
        for expr in self.e_separate.compiled:
            assert expr.pattern in exprs

    def test_compiled_files(self):
        if False:
            i = 10
            return i + 15
        if ISWINDOWS:
            regex1 = 'test\\\\one\\\\sub'
        else:
            regex1 = 'test/one/sub'
        self.e_separate.add(regex1)
        self.e_separate.mark(regex1)
        self.e_union.add(regex1)
        self.e_union.mark(regex1)
        separate_compiled_dirs = self.e_separate.compiled
        separate_compiled_files = [x for x in self.e_separate.compiled_files]
        union_compiled_dirs = self.e_union.compiled
        union_compiled_files = [x for x in self.e_union.compiled_files][0]
        print(f'compiled files: {union_compiled_files}')
        eq_(len(separate_compiled_dirs), len(default_regexes) + 1)
        eq_(len(separate_compiled_files), len(default_regexes))
        eq_(len(split_union(union_compiled_dirs[0])), len(default_regexes) + 1)
        eq_(len(split_union(union_compiled_files)), len(default_regexes))

class TestCaseCompiledDict(TestCaseCompiledList):
    """Test the dictionary version"""

    def setup_method(self, method):
        if False:
            print('Hello World!')
        self.e_separate = ExcludeDict(union_regex=False)
        self.e_separate.restore_defaults()
        self.e_union = ExcludeDict(union_regex=True)
        self.e_union.restore_defaults()