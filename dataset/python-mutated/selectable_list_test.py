from hscommon.testutil import eq_, callcounter, CallLogger
from hscommon.gui.selectable_list import SelectableList, GUISelectableList

def test_in():
    if False:
        while True:
            i = 10
    sl = SelectableList()
    some_list = [sl]
    assert SelectableList() not in some_list

def test_selection_range():
    if False:
        for i in range(10):
            print('nop')
    sl = SelectableList(['foo', 'bar', 'baz'])
    sl.selected_index = 3
    eq_(sl.selected_index, 2)
    del sl[2]
    eq_(sl.selected_index, 1)

def test_update_selection_called():
    if False:
        return 10
    sl = SelectableList(['foo', 'bar'])
    sl._update_selection = callcounter()
    sl.select(1)
    eq_(sl._update_selection.callcount, 1)
    sl.selected_index = 0
    eq_(sl._update_selection.callcount, 1)

def test_guicalls():
    if False:
        return 10
    sl = GUISelectableList(['foo', 'bar'])
    sl.view = CallLogger()
    sl.view.check_gui_calls(['refresh'])
    sl[1] = 'baz'
    sl.view.check_gui_calls(['refresh'])
    sl.append('foo')
    sl.view.check_gui_calls(['refresh'])
    del sl[2]
    sl.view.check_gui_calls(['refresh'])
    sl.remove('baz')
    sl.view.check_gui_calls(['refresh'])
    sl.insert(0, 'foo')
    sl.view.check_gui_calls(['refresh'])
    sl.select(1)
    sl.view.check_gui_calls(['update_selection'])

def test_search_by_prefix():
    if False:
        while True:
            i = 10
    sl = SelectableList(['foo', 'bAr', 'baZ'])
    eq_(sl.search_by_prefix('b'), 1)
    eq_(sl.search_by_prefix('BA'), 1)
    eq_(sl.search_by_prefix('BAZ'), 2)
    eq_(sl.search_by_prefix('BAZZ'), -1)