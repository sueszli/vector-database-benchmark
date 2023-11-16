from core.tests.base import TestApp, GetTestGroups

def app_with_results():
    if False:
        print('Hello World!')
    app = TestApp()
    (objects, matches, groups) = GetTestGroups()
    app.app.results.groups = groups
    app.rtable.refresh()
    return app

def test_delta_flags_delta_mode_off():
    if False:
        print('Hello World!')
    app = app_with_results()
    app.rtable.delta_values = False
    assert not app.rtable[0].is_cell_delta('size')
    assert not app.rtable[1].is_cell_delta('size')

def test_delta_flags_delta_mode_on_delta_columns():
    if False:
        while True:
            i = 10
    app = app_with_results()
    app.rtable.delta_values = True
    assert not app.rtable[0].is_cell_delta('size')
    assert app.rtable[1].is_cell_delta('size')

def test_delta_flags_delta_mode_on_non_delta_columns():
    if False:
        while True:
            i = 10
    app = app_with_results()
    app.rtable.delta_values = True
    assert app.rtable[1].is_cell_delta('name')
    assert not app.rtable[3].is_cell_delta('name')
    assert not app.rtable[4].is_cell_delta('name')

def test_delta_flags_delta_mode_on_non_delta_columns_case_insensitive():
    if False:
        print('Hello World!')
    app = app_with_results()
    app.app.results.groups[1].ref.name = 'ibAbtu'
    app.app.results.groups[1].dupes[0].name = 'IBaBTU'
    app.rtable.delta_values = True
    assert not app.rtable[4].is_cell_delta('name')