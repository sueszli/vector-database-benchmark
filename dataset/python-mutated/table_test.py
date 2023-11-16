from hscommon.testutil import CallLogger, eq_
from hscommon.gui.table import Table, GUITable, Row

class TestRow(Row):
    __test__ = False

    def __init__(self, table, index, is_new=False):
        if False:
            print('Hello World!')
        Row.__init__(self, table)
        self.is_new = is_new
        self._index = index

    def load(self):
        if False:
            print('Hello World!')
        pass

    def save(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_new = False

    @property
    def index(self):
        if False:
            while True:
                i = 10
        return self._index

class TestGUITable(GUITable):
    __test__ = False

    def __init__(self, rowcount, viewclass=CallLogger):
        if False:
            return 10
        GUITable.__init__(self)
        self.view = viewclass()
        self.view.model = self
        self.rowcount = rowcount
        self.updated_rows = None

    def _do_add(self):
        if False:
            return 10
        return (TestRow(self, len(self), is_new=True), len(self))

    def _is_edited_new(self):
        if False:
            return 10
        return self.edited is not None and self.edited.is_new

    def _fill(self):
        if False:
            print('Hello World!')
        for i in range(self.rowcount):
            self.append(TestRow(self, i))

    def _update_selection(self):
        if False:
            return 10
        self.updated_rows = self.selected_rows[:]

def table_with_footer():
    if False:
        for i in range(10):
            print('nop')
    table = Table()
    table.append(TestRow(table, 0))
    footer = TestRow(table, 1)
    table.footer = footer
    return (table, footer)

def table_with_header():
    if False:
        for i in range(10):
            print('nop')
    table = Table()
    table.append(TestRow(table, 1))
    header = TestRow(table, 0)
    table.header = header
    return (table, header)

def test_allow_edit_when_attr_is_property_with_fset():
    if False:
        for i in range(10):
            print('nop')

    class TestRow(Row):

        @property
        def foo(self):
            if False:
                print('Hello World!')
            pass

        @property
        def bar(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        @bar.setter
        def bar(self, value):
            if False:
                print('Hello World!')
            pass
    row = TestRow(Table())
    assert row.can_edit_cell('bar')
    assert not row.can_edit_cell('foo')
    assert not row.can_edit_cell('baz')

def test_can_edit_prop_has_priority_over_fset_checks():
    if False:
        i = 10
        return i + 15

    class TestRow(Row):

        @property
        def bar(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        @bar.setter
        def bar(self, value):
            if False:
                i = 10
                return i + 15
            pass
        can_edit_bar = False
    row = TestRow(Table())
    assert not row.can_edit_cell('bar')

def test_in():
    if False:
        i = 10
        return i + 15
    table = Table()
    some_list = [table]
    assert Table() not in some_list

def test_footer_del_all():
    if False:
        i = 10
        return i + 15
    (table, footer) = table_with_footer()
    del table[:]
    assert table.footer is None

def test_footer_del_row():
    if False:
        return 10
    (table, footer) = table_with_footer()
    del table[-1]
    assert table.footer is None
    eq_(len(table), 1)

def test_footer_is_appened_to_table():
    if False:
        while True:
            i = 10
    (table, footer) = table_with_footer()
    eq_(len(table), 2)
    assert table[1] is footer

def test_footer_remove():
    if False:
        return 10
    (table, footer) = table_with_footer()
    table.remove(footer)
    assert table.footer is None

def test_footer_replaces_old_footer():
    if False:
        return 10
    (table, footer) = table_with_footer()
    other = Row(table)
    table.footer = other
    assert table.footer is other
    eq_(len(table), 2)
    assert table[1] is other

def test_footer_rows_and_row_count():
    if False:
        for i in range(10):
            print('nop')
    (table, footer) = table_with_footer()
    eq_(table.row_count, 1)
    eq_(table.rows, table[:-1])

def test_footer_setting_to_none_removes_old_one():
    if False:
        return 10
    (table, footer) = table_with_footer()
    table.footer = None
    assert table.footer is None
    eq_(len(table), 1)

def test_footer_stays_there_on_append():
    if False:
        while True:
            i = 10
    (table, footer) = table_with_footer()
    table.append(Row(table))
    eq_(len(table), 3)
    assert table[2] is footer

def test_footer_stays_there_on_insert():
    if False:
        for i in range(10):
            print('nop')
    (table, footer) = table_with_footer()
    table.insert(3, Row(table))
    eq_(len(table), 3)
    assert table[2] is footer

def test_header_del_all():
    if False:
        i = 10
        return i + 15
    (table, header) = table_with_header()
    del table[:]
    assert table.header is None

def test_header_del_row():
    if False:
        for i in range(10):
            print('nop')
    (table, header) = table_with_header()
    del table[0]
    assert table.header is None
    eq_(len(table), 1)

def test_header_is_inserted_in_table():
    if False:
        return 10
    (table, header) = table_with_header()
    eq_(len(table), 2)
    assert table[0] is header

def test_header_remove():
    if False:
        for i in range(10):
            print('nop')
    (table, header) = table_with_header()
    table.remove(header)
    assert table.header is None

def test_header_replaces_old_header():
    if False:
        return 10
    (table, header) = table_with_header()
    other = Row(table)
    table.header = other
    assert table.header is other
    eq_(len(table), 2)
    assert table[0] is other

def test_header_rows_and_row_count():
    if False:
        return 10
    (table, header) = table_with_header()
    eq_(table.row_count, 1)
    eq_(table.rows, table[1:])

def test_header_setting_to_none_removes_old_one():
    if False:
        return 10
    (table, header) = table_with_header()
    table.header = None
    assert table.header is None
    eq_(len(table), 1)

def test_header_stays_there_on_insert():
    if False:
        print('Hello World!')
    (table, header) = table_with_header()
    table.insert(0, Row(table))
    eq_(len(table), 3)
    assert table[0] is header

def test_refresh_view_on_refresh():
    if False:
        print('Hello World!')
    table = TestGUITable(1)
    table.refresh()
    table.view.check_gui_calls(['refresh'])
    table.view.clear_calls()
    table.refresh(refresh_view=False)
    table.view.check_gui_calls([])

def test_restore_selection():
    if False:
        while True:
            i = 10
    table = TestGUITable(10)
    table.refresh()
    eq_(table.selected_indexes, [9])

def test_restore_selection_after_cancel_edits():
    if False:
        return 10

    class MyTable(TestGUITable):

        def _restore_selection(self, previous_selection):
            if False:
                while True:
                    i = 10
            self.selected_indexes = [6]
    table = MyTable(10)
    table.refresh()
    table.add()
    table.cancel_edits()
    eq_(table.selected_indexes, [6])

def test_restore_selection_with_previous_selection():
    if False:
        print('Hello World!')
    table = TestGUITable(10)
    table.refresh()
    table.selected_indexes = [2, 4]
    table.refresh()
    eq_(table.selected_indexes, [2, 4])

def test_restore_selection_custom():
    if False:
        for i in range(10):
            print('nop')

    class MyTable(TestGUITable):

        def _restore_selection(self, previous_selection):
            if False:
                print('Hello World!')
            self.selected_indexes = [6]
    table = MyTable(10)
    table.refresh()
    eq_(table.selected_indexes, [6])

def test_row_cell_value():
    if False:
        while True:
            i = 10
    row = Row(Table())
    row.from_ = 'foo'
    eq_(row.get_cell_value('from'), 'foo')
    row.set_cell_value('from', 'bar')
    eq_(row.get_cell_value('from'), 'bar')

def test_sort_table_also_tries_attributes_without_underscores():
    if False:
        print('Hello World!')
    table = Table()
    row1 = Row(table)
    row1._foo = 'a'
    row1.foo = 'b'
    row1.bar = 'c'
    row2 = Row(table)
    row2._foo = 'b'
    row2.foo = 'a'
    row2.bar = 'b'
    table.append(row1)
    table.append(row2)
    table.sort_by('foo')
    assert table[0] is row1
    assert table[1] is row2
    table.sort_by('bar')
    assert table[0] is row2
    assert table[1] is row1

def test_sort_table_updates_selection():
    if False:
        while True:
            i = 10
    table = TestGUITable(10)
    table.refresh()
    table.select([2, 4])
    table.sort_by('index', desc=True)
    eq_(len(table.updated_rows), 2)
    (r1, r2) = table.updated_rows
    eq_(r1.index, 7)
    eq_(r2.index, 5)

def test_sort_table_with_footer():
    if False:
        return 10
    (table, footer) = table_with_footer()
    table.sort_by('index', desc=True)
    assert table[-1] is footer

def test_sort_table_with_header():
    if False:
        i = 10
        return i + 15
    (table, header) = table_with_header()
    table.sort_by('index', desc=True)
    assert table[0] is header

def test_add_with_view_that_saves_during_refresh():
    if False:
        for i in range(10):
            print('nop')

    class TableView(CallLogger):

        def refresh(self):
            if False:
                print('Hello World!')
            self.model.save_edits()
    table = TestGUITable(10, viewclass=TableView)
    table.add()
    assert table.edited is not None