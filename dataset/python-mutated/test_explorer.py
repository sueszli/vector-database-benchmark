"""
Tests for explorer.py
"""
import os
import os.path as osp
import sys
import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QDialog, QDialogButtonBox, QInputDialog, QMessageBox, QTextEdit
from spyder.plugins.explorer.widgets.main_widget import FileExplorerTest
from spyder.plugins.projects.widgets.main_widget import ProjectExplorerTest
HERE = osp.abspath(osp.dirname(__file__))

@pytest.fixture
def file_explorer(qtbot):
    if False:
        while True:
            i = 10
    'Set up FileExplorerTest.'
    widget = FileExplorerTest()
    widget.show()
    qtbot.addWidget(widget)
    return widget

@pytest.fixture
def file_explorer_associations(qtbot):
    if False:
        i = 10
        return i + 15
    'Set up FileExplorerTest.'
    if os.name == 'nt':
        ext = '.exe'
    elif sys.platform == 'darwin':
        ext = '.app'
    else:
        ext = '.desktop'
    associations = {'*.txt': [('App 1', '/some/fake/some_app_1' + ext)], '*.json,*.csv': [('App 2', '/some/fake/some_app_2' + ext), ('App 1', '/some/fake/some_app_1' + ext)]}
    widget = FileExplorerTest(file_associations=associations)
    qtbot.addWidget(widget)
    return widget

def create_timer(func, interval=500):
    if False:
        return 10
    'Helper function to help interact with modal dialogs.'
    timer = QTimer()
    timer.setInterval(interval)
    timer.setSingleShot(True)
    timer.timeout.connect(func)
    timer.start()
    return timer

@pytest.fixture(params=[FileExplorerTest, ProjectExplorerTest])
def explorer_with_files(qtbot, create_folders_files, request):
    if False:
        return 10
    'Setup Project/File Explorer widget.'
    cb = QApplication.clipboard()
    (paths, project_dir, destination_dir, top_folder) = create_folders_files
    explorer_orig = request.param(directory=project_dir)
    explorer_dest = request.param(directory=destination_dir)
    qtbot.addWidget(explorer_orig)
    qtbot.addWidget(explorer_dest)
    return (explorer_orig, explorer_dest, paths, top_folder, cb)

def test_file_explorer(file_explorer):
    if False:
        i = 10
        return i + 15
    'Run FileExplorerTest.'
    file_explorer.resize(640, 480)
    file_explorer.show()
    assert file_explorer

@pytest.mark.parametrize('path_method', ['absolute', 'relative'])
def test_copy_path(explorer_with_files, path_method):
    if False:
        print('Hello World!')
    'Test copy absolute and relative paths.'
    (project, __, file_paths, __, cb) = explorer_with_files
    explorer_directory = project.explorer.treewidget.fsmodel.rootPath()
    copied_from = project.explorer.treewidget._parent.__class__.__name__
    project.explorer.treewidget.copy_path(fnames=file_paths, method=path_method)
    cb_output = cb.text(mode=cb.Clipboard)
    path_list = [path.strip(',"') for path in cb_output.splitlines()]
    assert len(path_list) == len(file_paths)
    for (path, expected_path) in zip(path_list, file_paths):
        if path_method == 'relative':
            expected_path = osp.relpath(expected_path, explorer_directory)
            if copied_from == 'ProjectExplorerWidget':
                expected_path = os.sep.join(expected_path.strip(os.sep).split(os.sep)[1:])
        assert osp.normpath(path) == osp.normpath(expected_path)

def test_copy_file(explorer_with_files):
    if False:
        return 10
    'Test copy file(s)/folders(s) to clipboard.'
    (project, __, file_paths, __, cb) = explorer_with_files
    project.explorer.treewidget.copy_file_clipboard(fnames=file_paths)
    cb_data = cb.mimeData().urls()
    assert len(cb_data) == len(file_paths)
    for (url, expected_path) in zip(cb_data, file_paths):
        file_name = url.toLocalFile()
        assert osp.normpath(file_name) == osp.normpath(expected_path)
        try:
            assert osp.isdir(file_name)
        except AssertionError:
            assert osp.isfile(file_name)
            with open(file_name, 'r') as fh:
                text = fh.read()
            assert text == 'File Path:\n' + str(file_name)

def test_save_file(explorer_with_files):
    if False:
        for i in range(10):
            print('nop')
    'Test save file(s)/folders(s) from clipboard.'
    (project, dest, file_paths, __, __) = explorer_with_files
    project.explorer.treewidget.copy_file_clipboard(fnames=file_paths)
    dest.explorer.treewidget.save_file_clipboard(fnames=[dest.directory])
    for item in file_paths:
        destination_item = osp.join(dest.directory, osp.basename(item))
        assert osp.exists(destination_item)
        if osp.isfile(destination_item):
            with open(destination_item, 'r') as fh:
                text = fh.read()
            assert text == 'File Path:\n' + str(item).replace(os.sep, '/')

def test_delete_file(explorer_with_files, mocker):
    if False:
        return 10
    'Test delete file(s)/folders(s).'
    (project, __, __, top_folder, __) = explorer_with_files
    mocker.patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes)
    project.explorer.treewidget.delete(fnames=[top_folder])
    assert not osp.exists(top_folder)

def test_rename_file_with_files(explorer_with_files, mocker, qtbot):
    if False:
        i = 10
        return i + 15
    'Test that rename_file renames the file and sends out right signal.'
    (project, __, file_paths, __, __) = explorer_with_files
    for old_path in file_paths:
        if osp.isfile(old_path):
            old_basename = osp.basename(old_path)
            new_basename = 'new' + old_basename
            new_path = osp.join(osp.dirname(old_path), new_basename)
            mocker.patch.object(QInputDialog, 'getText', return_value=(new_basename, True))
            treewidget = project.explorer.treewidget
            with qtbot.waitSignal(treewidget.sig_renamed) as blocker:
                treewidget.rename_file(old_path)
            assert blocker.args == [old_path, new_path]
            assert not osp.exists(old_path)
            with open(new_path, 'r') as fh:
                text = fh.read()
            assert text == 'File Path:\n' + str(old_path).replace(os.sep, '/')

def test_single_click_to_open(qtbot, file_explorer):
    if False:
        return 10
    'Test single and double click open option for the file explorer.'
    file_explorer.show()
    treewidget = file_explorer.explorer.treewidget
    model = treewidget.model()
    cwd = os.getcwd()
    qtbot.keyClick(treewidget, Qt.Key_Up)
    initial_index = treewidget.currentIndex()

    def run_test_helper(single_click, initial_index):
        if False:
            i = 10
            return i + 15
        treewidget.setCurrentIndex(initial_index)
        file_explorer.label3.setText('')
        file_explorer.label1.setText('')
        for __ in range(4):
            qtbot.keyClick(treewidget, Qt.Key_Down)
            index = treewidget.currentIndex()
            path = model.data(index)
            if path:
                full_path = os.path.join(cwd, path)
                if os.path.isfile(full_path):
                    rect = treewidget.visualRect(index)
                    pos = rect.center()
                    qtbot.mouseClick(treewidget.viewport(), Qt.LeftButton, pos=pos)
                    if single_click:
                        assert full_path == file_explorer.label1.text()
                    else:
                        assert full_path != file_explorer.label1.text()
    treewidget.set_conf('single_click_to_open', True)
    run_test_helper(single_click=True, initial_index=initial_index)
    treewidget.set_conf('single_click_to_open', False)
    run_test_helper(single_click=False, initial_index=initial_index)

@pytest.mark.order(1)
def test_get_common_file_associations(qtbot, file_explorer_associations):
    if False:
        i = 10
        return i + 15
    widget = file_explorer_associations.explorer.treewidget
    associations = widget.get_common_file_associations(['/some/path/file.txt', '/some/path/file1.json', '/some/path/file2.csv'])
    if os.name == 'nt':
        ext = '.exe'
    elif sys.platform == 'darwin':
        ext = '.app'
    else:
        ext = '.desktop'
    assert associations[0][-1] == '/some/fake/some_app_1' + ext

@pytest.mark.order(1)
def test_get_file_associations(qtbot, file_explorer_associations):
    if False:
        print('Hello World!')
    widget = file_explorer_associations.explorer.treewidget
    associations = widget.get_file_associations('/some/path/file.txt')
    if os.name == 'nt':
        ext = '.exe'
    elif sys.platform == 'darwin':
        ext = '.app'
    else:
        ext = '.desktop'
    assert associations[0][-1] == '/some/fake/some_app_1' + ext

@pytest.mark.order(1)
def test_create_file_manager_actions(qtbot, file_explorer_associations, tmp_path):
    if False:
        while True:
            i = 10
    widget = file_explorer_associations.explorer.treewidget
    fpath = tmp_path / 'text.txt'
    fpath.write_text(u'hello!')
    fpath_2 = tmp_path / 'text.json'
    fpath_2.write_text(u'hello!')
    fpath_3 = tmp_path / 'text.md'
    fpath_3.write_text(u'hello!')
    actions = widget._create_file_associations_actions([str(fpath)])
    action_texts = [action.text().lower() for action in actions]
    assert any(('app 1' in text for text in action_texts))
    assert any(('default external application' in text for text in action_texts))
    actions = widget._create_file_associations_actions([str(fpath), str(fpath_2)])
    action_texts = [action.text().lower() for action in actions]
    assert any(('app 1' in text for text in action_texts))
    assert any(('default external application' in text for text in action_texts))
    actions = widget._create_file_associations_actions([str(fpath_3)])
    action_texts = [action.text().lower() for action in actions]
    assert not action_texts

@pytest.mark.order(1)
def test_clicked(qtbot, file_explorer_associations, tmp_path):
    if False:
        print('Hello World!')
    widget = file_explorer_associations.explorer.treewidget
    some_dir = tmp_path / 'some_dir'
    some_dir.mkdir()
    fpath = some_dir / 'text.txt'
    fpath.write_text(u'hello!')
    widget.chdir(str(some_dir))
    qtbot.wait(500)
    qtbot.keyClick(widget, Qt.Key_Up)

    def interact():
        if False:
            print('Hello World!')
        msgbox = widget.findChild(QMessageBox)
        assert msgbox
        qtbot.keyClick(msgbox, Qt.Key_Return)
    _ = create_timer(interact)
    qtbot.keyClick(widget, Qt.Key_Return)

    def interact_2():
        if False:
            i = 10
            return i + 15
        msgbox = widget.findChild(QMessageBox)
        assert not msgbox
    widget.set_file_associations({})
    _ = create_timer(interact_2)
    qtbot.keyClick(widget, Qt.Key_Return)

@pytest.mark.order(1)
def test_check_launch_error_codes(qtbot, file_explorer_associations):
    if False:
        i = 10
        return i + 15
    widget = file_explorer_associations.explorer.treewidget
    return_codes = {'some-command': 0, 'some-other-command': 0}
    assert widget.check_launch_error_codes(return_codes)

    def interact():
        if False:
            print('Hello World!')
        msgbox = widget.findChild(QMessageBox)
        assert msgbox
        qtbot.keyClick(msgbox, Qt.Key_Return)
    return_codes = {'some-command': 1}
    _ = create_timer(interact)
    res = widget.check_launch_error_codes(return_codes)
    assert not res

    def interact_2():
        if False:
            print('Hello World!')
        msgbox = widget.findChild(QMessageBox)
        assert msgbox
        qtbot.keyClick(msgbox, Qt.Key_Return)
    return_codes = {'some-command': 1, 'some-other-command': 1}
    _ = create_timer(interact_2)
    res = widget.check_launch_error_codes(return_codes)
    assert not res

@pytest.mark.order(1)
def test_open_association(qtbot, file_explorer_associations, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    widget = file_explorer_associations.explorer.treewidget
    some_dir = tmp_path / 'some_dir'
    some_dir.mkdir()
    fpath = some_dir / 'text.txt'
    fpath.write_text(u'hello!')
    qtbot.keyClick(widget, Qt.Key_Down)

    def interact():
        if False:
            return 10
        msgbox = widget.findChild(QMessageBox)
        assert msgbox
        qtbot.keyClick(msgbox, Qt.Key_Return)
    _ = create_timer(interact)
    widget.open_association('some-app')

@pytest.mark.order(1)
def test_update_filters(file_explorer, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that view is updated if the filter button is activated and\n    filters are changed.\n\n    This is a regression test for spyder-ide/spyder#14328\n    '
    widget = file_explorer.explorer.treewidget
    explorer_file = osp.join(osp.dirname(HERE), 'explorer.py')
    assert osp.isfile(explorer_file)
    idx0 = widget.get_index(explorer_file)
    assert idx0.isValid()
    widget.filter_button.toggle()

    def interact():
        if False:
            while True:
                i = 10
        dlg = widget.findChild(QDialog)
        assert dlg
        filters = dlg.findChild(QTextEdit)
        filters.setPlainText('*.png')
        button_box = dlg.findChild(QDialogButtonBox)
        button_box.button(QDialogButtonBox.Ok).clicked.emit()
    _ = create_timer(interact)
    widget.edit_filter()
    qtbot.wait(1000)
    idx1 = widget.get_index(explorer_file)
    assert not idx1.isValid()
if __name__ == '__main__':
    pytest.main()