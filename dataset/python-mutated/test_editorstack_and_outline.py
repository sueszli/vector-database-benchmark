"""
Tests syncing between the EditorStack and OutlineExplorerWidget.
"""
import os
import os.path as osp
import sys
from unittest.mock import Mock
import pytest
from spyder.config.base import running_in_ci
from spyder.plugins.editor.widgets.editorstack import EditorStack

def get_tree_elements(treewidget):
    if False:
        for i in range(10):
            print('nop')
    'Get elements present in the Outline tree widget.'
    root_item = treewidget.get_top_level_items()[0]
    root_ref = root_item.ref
    filename = osp.basename(root_ref.name)
    root_tree = {filename: []}
    stack = [(root_tree[filename], node) for node in root_ref.children]
    while len(stack) > 0:
        (parent_tree, node) = stack.pop(0)
        this_tree = {node.name: []}
        parent_tree.append(this_tree)
        this_stack = [(this_tree[node.name], child) for child in node.children]
        stack = this_stack + stack
    return root_tree

@pytest.fixture(scope='module')
def test_files(tmpdir_factory):
    if False:
        while True:
            i = 10
    'Create and save some python codes and text in temporary files.'
    tmpdir = tmpdir_factory.mktemp('files')
    filename1 = osp.join(tmpdir.strpath, 'foo1.py')
    with open(filename1, 'w') as f:
        f.write('# -*- coding: utf-8 -*-\ndef foo:\n    print(Hello World!)\n')
    filename2 = osp.join(tmpdir.strpath, 'text1.txt')
    with open(filename2, 'w') as f:
        f.write('This is a simple text file for\ntesting the Outline Explorer.\n')
    filename3 = osp.join(tmpdir.strpath, 'foo2.py')
    with open(filename3, 'w') as f:
        f.write('# -*- coding: utf-8 -*-\n# ---- a comment\n')
    return [filename1, filename2, filename3]

@pytest.fixture
def editorstack(qtbot, outlineexplorer):
    if False:
        print('Hello World!')

    def _create_editorstack(files):
        if False:
            return 10
        editorstack = EditorStack(None, [], False)
        editorstack.set_find_widget(Mock())
        editorstack.set_io_actions(Mock(), Mock(), Mock(), Mock())
        editorstack.analysis_timer = Mock()
        editorstack.save_dialog_on_tests = True
        editorstack.set_outlineexplorer(outlineexplorer)
        qtbot.addWidget(editorstack)
        editorstack.show()
        for (index, file) in enumerate(files):
            focus = index == 0
            editorstack.load(file, set_current=focus)
        return editorstack
    return _create_editorstack

def test_load_files(editorstack, outlineexplorer, test_files):
    if False:
        i = 10
        return i + 15
    '\n    Test that the content of the outline explorer is updated correctly\n    after a file is loaded in the editor.\n    '
    editorstack = editorstack([])
    treewidget = outlineexplorer.treewidget
    expected_result = [['foo1.py'], ['foo1.py', 'text1.txt'], ['foo1.py', 'text1.txt', 'foo2.py']]
    for (index, file) in enumerate(test_files):
        editorstack.load(file)
        assert editorstack.get_current_filename() == file
        assert editorstack.get_stack_index() == index
        results = [item.text(0) for item in treewidget.get_visible_items()]
        assert results == expected_result[index]
        assert editorstack.get_stack_index() == index

def test_close_editor(editorstack, outlineexplorer, test_files):
    if False:
        print('Hello World!')
    '\n    Test that the content of the outline explorer is empty after the\n    editorstack has been closed.\n\n    Regression test for spyder-ide/spyder#7798.\n    '
    editorstack = editorstack(test_files)
    treewidget = outlineexplorer.treewidget
    assert treewidget.get_visible_items()
    editorstack.close()
    assert not treewidget.get_visible_items()

def test_close_a_file(editorstack, outlineexplorer, test_files):
    if False:
        print('Hello World!')
    '\n    Test that the content of the outline explorer is updated corrrectly\n    after a file has been closed in the editorstack.\n\n    Regression test for spyder-ide/spyder#7798.\n    '
    editorstack = editorstack(test_files)
    treewidget = outlineexplorer.treewidget
    editorstack.close_file(index=1)
    results = [item.text(0) for item in treewidget.get_visible_items()]
    assert results == ['foo1.py', 'foo2.py']

def test_sort_file_alphabetically(editorstack, outlineexplorer, test_files):
    if False:
        i = 10
        return i + 15
    '\n    Test that the option to sort the files in alphabetical order in the\n    outline explorer is working as expected.\n\n    This feature was introduced in spyder-ide/spyder#8015.\n    '
    editorstack = editorstack(test_files)
    treewidget = outlineexplorer.treewidget
    results = [item.text(0) for item in treewidget.get_visible_items()]
    assert results == ['foo1.py', 'text1.txt', 'foo2.py']
    treewidget.toggle_sort_files_alphabetically(True)
    results = [item.text(0) for item in treewidget.get_visible_items()]
    assert results == ['foo1.py', 'foo2.py', 'text1.txt']

def test_sync_file_order(editorstack, outlineexplorer, test_files):
    if False:
        return 10
    '\n    Test that the order of the files in the Outline Explorer is updated when\n    tabs are moved in the EditorStack.\n\n    This feature was introduced in spyder-ide/spyder#8015.\n    '
    editorstack = editorstack(test_files)
    treewidget = outlineexplorer.treewidget
    results = [item.text(0) for item in treewidget.get_visible_items()]
    assert results == ['foo1.py', 'text1.txt', 'foo2.py']
    editorstack.tabs.tabBar().moveTab(0, 1)
    results = [item.text(0) for item in treewidget.get_visible_items()]
    assert results == ['text1.txt', 'foo1.py', 'foo2.py']

@pytest.mark.skipif(running_in_ci() or os.name == 'nt', reason='Fails on CIs and on Windows')
def test_toggle_off_show_all_files(editorstack, outlineexplorer, test_files, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that toggling off the option to show all files in the Outline Explorer\n    hide all root file items but the one corresponding to the currently\n    selected Editor and assert that the remaning root file item is\n    expanded correctly.\n    '
    editorstack = editorstack(test_files)
    treewidget = outlineexplorer.treewidget
    assert editorstack.get_stack_index() == 0
    treewidget.toggle_show_all_files(False)
    qtbot.wait(500)
    results = [item.text(0) for item in treewidget.get_visible_items()]
    assert results == ['foo1.py']

@pytest.mark.skipif(sys.platform.startswith('linux'), reason='Fails on Linux')
def test_single_file_sync(editorstack, outlineexplorer, test_files, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that the content of the Outline Explorer is updated correctly\n    when the current Editor in the Editorstack changes.\n    '
    editorstack = editorstack(test_files)
    treewidget = outlineexplorer.treewidget
    treewidget.toggle_show_all_files(False)
    assert editorstack.get_stack_index() == 0
    with qtbot.waitSignal(editorstack.editor_focus_changed):
        editorstack.tabs.setCurrentIndex(2)
    results = [item.text(0) for item in treewidget.get_visible_items()]
    assert results == ['foo2.py']

def test_toggle_on_show_all_files(editorstack, outlineexplorer, test_files):
    if False:
        print('Hello World!')
    '\n    Test that toggling back the option to show all files, after the\n    order of the files in the Editorstack was changed while it was in single\n    file mode, show all the root file items in the correct order.\n    '
    editorstack = editorstack(test_files)
    treewidget = outlineexplorer.treewidget
    treewidget.toggle_show_all_files(False)
    editorstack.tabs.tabBar().moveTab(0, 1)
    treewidget.toggle_show_all_files(True)
    results = [item.text(0) for item in treewidget.get_visible_items()]
    assert results == ['text1.txt', 'foo1.py', 'foo2.py']
if __name__ == '__main__':
    pytest.main([os.path.basename(__file__), '-vv', '-rw'])