import os.path as op
from itertools import combinations
from core.tests.base import TestApp, NamedObject, with_app, eq_
from core.engine import Group, Match
no = NamedObject

def app_with_dupes(dupes):
    if False:
        for i in range(10):
            print('nop')
    app = TestApp()
    groups = []
    for dupelist in dupes:
        g = Group()
        for (dupe1, dupe2) in combinations(dupelist, 2):
            g.add_match(Match(dupe1, dupe2, 100))
        groups.append(g)
    app.app.results.groups = groups
    app.app._results_changed()
    return app

def app_normal_results():
    if False:
        print('Hello World!')
    dupes = [[no('foo1.ext1', size=1, folder='folder1'), no('foo2.ext2', size=2, folder='folder2')]]
    return app_with_dupes(dupes)

@with_app(app_normal_results)
def test_kind_subcrit(app):
    if False:
        return 10
    app.select_pri_criterion('Kind')
    eq_(app.pdialog.criteria_list[:], ['ext1', 'ext2'])

@with_app(app_normal_results)
def test_kind_reprioritization(app):
    if False:
        return 10
    app.select_pri_criterion('Kind')
    app.pdialog.criteria_list.select([1])
    app.pdialog.add_selected()
    app.pdialog.perform_reprioritization()
    eq_(app.rtable[0].data['name'], 'foo2.ext2')

@with_app(app_normal_results)
def test_folder_subcrit(app):
    if False:
        return 10
    app.select_pri_criterion('Folder')
    eq_(app.pdialog.criteria_list[:], ['folder1', 'folder2'])

@with_app(app_normal_results)
def test_folder_reprioritization(app):
    if False:
        i = 10
        return i + 15
    app.select_pri_criterion('Folder')
    app.pdialog.criteria_list.select([1])
    app.pdialog.add_selected()
    app.pdialog.perform_reprioritization()
    eq_(app.rtable[0].data['name'], 'foo2.ext2')

@with_app(app_normal_results)
def test_prilist_display(app):
    if False:
        i = 10
        return i + 15
    app.select_pri_criterion('Kind')
    app.pdialog.criteria_list.select([1])
    app.pdialog.add_selected()
    app.select_pri_criterion('Folder')
    app.pdialog.criteria_list.select([1])
    app.pdialog.add_selected()
    app.select_pri_criterion('Size')
    app.pdialog.criteria_list.select([1])
    app.pdialog.add_selected()
    expected = ['Kind (ext2)', 'Folder (folder2)', 'Size (Lowest)']
    eq_(app.pdialog.prioritization_list[:], expected)

@with_app(app_normal_results)
def test_size_subcrit(app):
    if False:
        i = 10
        return i + 15
    app.select_pri_criterion('Size')
    eq_(app.pdialog.criteria_list[:], ['Highest', 'Lowest'])

@with_app(app_normal_results)
def test_size_reprioritization(app):
    if False:
        while True:
            i = 10
    app.select_pri_criterion('Size')
    app.pdialog.criteria_list.select([0])
    app.pdialog.add_selected()
    app.pdialog.perform_reprioritization()
    eq_(app.rtable[0].data['name'], 'foo2.ext2')

@with_app(app_normal_results)
def test_reorder_prioritizations(app):
    if False:
        return 10
    app.add_pri_criterion('Kind', 0)
    app.add_pri_criterion('Kind', 1)
    app.pdialog.prioritization_list.move_indexes([1], 0)
    expected = ['Kind (ext2)', 'Kind (ext1)']
    eq_(app.pdialog.prioritization_list[:], expected)

@with_app(app_normal_results)
def test_remove_crit_from_list(app):
    if False:
        i = 10
        return i + 15
    app.add_pri_criterion('Kind', 0)
    app.add_pri_criterion('Kind', 1)
    app.pdialog.prioritization_list.select(0)
    app.pdialog.remove_selected()
    expected = ['Kind (ext2)']
    eq_(app.pdialog.prioritization_list[:], expected)

@with_app(app_normal_results)
def test_add_crit_without_selection(app):
    if False:
        print('Hello World!')
    app.pdialog.add_selected()

def app_one_name_ends_with_number():
    if False:
        for i in range(10):
            print('nop')
    dupes = [[no('foo.ext'), no('foo1.ext')]]
    return app_with_dupes(dupes)

@with_app(app_one_name_ends_with_number)
def test_filename_reprioritization(app):
    if False:
        return 10
    app.add_pri_criterion('Filename', 0)
    app.pdialog.perform_reprioritization()
    eq_(app.rtable[0].data['name'], 'foo1.ext')

def app_with_subfolders():
    if False:
        return 10
    dupes = [[no('foo1', folder='baz'), no('foo2', folder='foo/bar')], [no('foo3', folder='baz'), no('foo4', folder='foo')]]
    return app_with_dupes(dupes)

@with_app(app_with_subfolders)
def test_folder_crit_is_sorted(app):
    if False:
        for i in range(10):
            print('nop')
    app.select_pri_criterion('Folder')
    eq_(app.pdialog.criteria_list[:], ['baz', 'foo', op.join('foo', 'bar')])

@with_app(app_with_subfolders)
def test_folder_crit_includes_subfolders(app):
    if False:
        i = 10
        return i + 15
    app.add_pri_criterion('Folder', 1)
    app.pdialog.perform_reprioritization()
    eq_(app.rtable[0].data['name'], 'foo2')
    eq_(app.rtable[2].data['name'], 'foo4')

@with_app(app_with_subfolders)
def test_display_something_on_empty_extensions(app):
    if False:
        while True:
            i = 10
    app.select_pri_criterion('Kind')
    eq_(app.pdialog.criteria_list[:], ['None'])

def app_one_name_longer_than_the_other():
    if False:
        i = 10
        return i + 15
    dupes = [[no('shortest.ext'), no('loooongest.ext')]]
    return app_with_dupes(dupes)

@with_app(app_one_name_longer_than_the_other)
def test_longest_filename_prioritization(app):
    if False:
        i = 10
        return i + 15
    app.add_pri_criterion('Filename', 2)
    app.pdialog.perform_reprioritization()
    eq_(app.rtable[0].data['name'], 'loooongest.ext')