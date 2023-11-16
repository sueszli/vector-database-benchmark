"""Tests for recover.py"""
import os.path as osp
import pytest
import shutil
from qtpy.QtWidgets import QDialogButtonBox, QPushButton, QTableWidget
from spyder.plugins.editor.widgets.recover import make_temporary_files, RecoveryDialog

@pytest.fixture
def recovery_env(tmpdir):
    if False:
        while True:
            i = 10
    'Create a dir with various autosave files and cleans up afterwards.'
    yield make_temporary_files(str(tmpdir))
    shutil.rmtree(str(tmpdir))

def test_recoverydialog_has_cancel_button(qtbot, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that RecoveryDialog has a Cancel button.\n\n    Test that a RecoveryDialog has a button in a dialog button box and that\n    this button cancels the dialog window.\n    '
    dialog = RecoveryDialog([])
    qtbot.addWidget(dialog)
    button = dialog.findChild(QDialogButtonBox).findChild(QPushButton)
    with qtbot.waitSignal(dialog.rejected):
        button.click()

def test_recoverydialog_table_labels(qtbot, recovery_env):
    if False:
        return 10
    'Test that table in RecoveryDialog has the correct labels.'
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)

    def text(i, j):
        if False:
            for i in range(10):
                print('nop')
        return table.cellWidget(i, j).text()
    assert osp.join(orig_dir, 'ham.py') in text(0, 0)
    assert osp.join(autosave_dir, 'ham.py') in text(0, 1)
    assert osp.join(orig_dir, 'spam.py') in text(1, 0)
    assert 'no longer exists' in text(1, 0)
    assert osp.join(autosave_dir, 'spam.py') in text(1, 1)
    assert 'not recorded' in text(2, 0)
    assert osp.join(autosave_dir, 'cheese.py') in text(2, 1)
    assert table.rowCount() == 3

def test_recoverydialog_exec_if_nonempty_when_empty(qtbot, tmpdir, mocker):
    if False:
        while True:
            i = 10
    '\n    Test that exec_if_nonempty does nothing if autosave files do not exist.\n\n    Specifically, test that it does not `exec_()` the dialog.\n    '
    dialog = RecoveryDialog([('ham', 'spam')])
    mocker.patch.object(dialog, 'exec_')
    assert dialog.exec_if_nonempty() == dialog.Accepted
    dialog.exec_.assert_not_called()

def test_recoverydialog_exec_if_nonempty_when_nonempty(qtbot, recovery_env, mocker):
    if False:
        i = 10
        return i + 15
    'Test that exec_if_nonempty executes dialog if autosave dir not empty.'
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    dialog = RecoveryDialog(autosave_mapping)
    mocker.patch.object(dialog, 'exec_', return_value='eggs')
    assert dialog.exec_if_nonempty() == 'eggs'
    assert dialog.exec_.called

def test_recoverydialog_exec_if_nonempty_when_no_autosave_dir(qtbot, recovery_env, mocker):
    if False:
        return 10
    '\n    Test that exec_if_nonempty does nothing if autosave dir does not exist.\n\n    Specifically, test that it does not `exec_()` the dialog.\n    '
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    shutil.rmtree(autosave_dir)
    dialog = RecoveryDialog(autosave_mapping)
    mocker.patch.object(dialog, 'exec_')
    assert dialog.exec_if_nonempty() == dialog.Accepted
    dialog.exec_.assert_not_called()

def test_recoverydialog_restore_button(qtbot, recovery_env):
    if False:
        for i in range(10):
            print('nop')
    "\n    Test the `Restore` button in `RecoveryDialog`.\n\n    Test that after pressing the 'Restore' button, the original file is\n    replaced by the autosave file, the latter is removed, and the row in the\n    grid is deactivated.\n    "
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    button = table.cellWidget(0, 2).findChildren(QPushButton)[0]
    button.click()
    with open(osp.join(orig_dir, 'ham.py')) as f:
        assert f.read() == 'ham = "autosave"\n'
    assert not osp.isfile(osp.join(autosave_dir, 'ham.py'))
    for col in range(table.columnCount()):
        assert not table.cellWidget(0, col).isEnabled()

def test_recoverydialog_restore_when_original_does_not_exist(qtbot, recovery_env):
    if False:
        print('Hello World!')
    "\n    Test the `Restore` button when the original file does not exist.\n\n    Test that after pressing the 'Restore' button, the autosave file is moved\n    to the location of the original file and the row in the grid is\n    deactivated.\n    "
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    button = table.cellWidget(1, 2).findChildren(QPushButton)[0]
    button.click()
    with open(osp.join(orig_dir, 'spam.py')) as f:
        assert f.read() == 'spam = "autosave"\n'
    assert not osp.isfile(osp.join(autosave_dir, 'spam.py'))
    for col in range(table.columnCount()):
        assert not table.cellWidget(1, col).isEnabled()

def test_recoverydialog_restore_when_original_not_recorded(qtbot, recovery_env, mocker):
    if False:
        for i in range(10):
            print('nop')
    "\n    Test the `Restore` button when the original file name is not known.\n\n    Test that after pressing the 'Restore' button, the autosave file is moved\n    to a location specified by the user and the row in the grid is deactivated.\n    "
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    new_name = osp.join(orig_dir, 'monty.py')
    mocker.patch('spyder.plugins.editor.widgets.recover.getsavefilename', return_value=(new_name, 'ignored'))
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    button = table.cellWidget(2, 2).findChildren(QPushButton)[0]
    button.click()
    with open(new_name) as f:
        assert f.read() == 'cheese = "autosave"\n'
    assert not osp.isfile(osp.join(autosave_dir, 'cheese.py'))
    for col in range(table.columnCount()):
        assert not table.cellWidget(2, col).isEnabled()

def test_recoverydialog_restore_fallback(qtbot, recovery_env, mocker):
    if False:
        print('Hello World!')
    "\n    Test fallback for when os.replace() fails when recovering a file.\n\n    Test that after pressing the 'Restore' button, if os.replace() fails,\n    the fallback to copy and delete kicks in and the restore succeeds.\n    Regression test for spyder-ide/spyder#8631.\n    "
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    mocker.patch('spyder.plugins.editor.widgets.recover.os.replace', side_effect=OSError)
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    button = table.cellWidget(0, 2).findChildren(QPushButton)[0]
    button.click()
    with open(osp.join(orig_dir, 'ham.py')) as f:
        assert f.read() == 'ham = "autosave"\n'
    assert not osp.isfile(osp.join(autosave_dir, 'ham.py'))
    for col in range(table.columnCount()):
        assert not table.cellWidget(0, col).isEnabled()

def test_recoverydialog_restore_when_error(qtbot, recovery_env, mocker):
    if False:
        while True:
            i = 10
    '\n    Test that errors during a restore action are handled gracefully.\n\n    Test that if an error arises when restoring a file, both the original and\n    the autosave files are kept unchanged, a dialog is displayed, and the row\n    in the grid is not deactivated.\n    '
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    mocker.patch('spyder.plugins.editor.widgets.recover.os.replace', side_effect=OSError)
    mocker.patch('spyder.plugins.editor.widgets.recover.shutil.copy2', side_effect=IOError)
    mock_QMessageBox = mocker.patch('spyder.plugins.editor.widgets.recover.QMessageBox')
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    button = table.cellWidget(0, 2).findChildren(QPushButton)[0]
    button.click()
    with open(osp.join(orig_dir, 'ham.py')) as f:
        assert f.read() == 'ham = "original"\n'
    with open(osp.join(autosave_dir, 'ham.py')) as f:
        assert f.read() == 'ham = "autosave"\n'
    assert mock_QMessageBox.called
    for col in range(table.columnCount()):
        assert table.cellWidget(0, col).isEnabled()

def test_recoverydialog_accepted_after_all_restored(qtbot, recovery_env, mocker):
    if False:
        print('Hello World!')
    '\n    Test that the recovery dialog is accepted after all files are restored.\n\n    Click all `Restore` buttons and test that the dialog is accepted\n    afterwards, but not before.\n    '
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    new_name = osp.join(orig_dir, 'monty.py')
    mocker.patch('spyder.plugins.editor.widgets.recover.getsavefilename', return_value=(new_name, 'ignored'))
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    with qtbot.assertNotEmitted(dialog.accepted):
        for row in range(table.rowCount() - 1):
            table.cellWidget(row, 2).findChildren(QPushButton)[0].click()
    with qtbot.waitSignal(dialog.accepted):
        row = table.rowCount() - 1
        table.cellWidget(row, 2).findChildren(QPushButton)[0].click()

def test_recoverydialog_discard_button(qtbot, recovery_env):
    if False:
        while True:
            i = 10
    "\n    Test the `Discard` button in the recovery dialog.\n\n    Test that after pressing the 'Discard' button, the autosave file is\n    deleted, the original file unchanged, and the row in the grid is\n    deactivated.\n    "
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    button = table.cellWidget(0, 2).findChildren(QPushButton)[1]
    button.click()
    assert not osp.isfile(osp.join(autosave_dir, 'ham.py'))
    with open(osp.join(orig_dir, 'ham.py')) as f:
        assert f.read() == 'ham = "original"\n'
    for col in range(table.columnCount()):
        assert not table.cellWidget(0, col).isEnabled()

def test_recoverydialog_discard_when_error(qtbot, recovery_env, mocker):
    if False:
        while True:
            i = 10
    '\n    Test that errors during a discard action are handled gracefully.\n\n    Test that if an error arises when discarding a file, both the original and\n    the autosave files are kept unchanged, a dialog is displayed, and the row\n    in the grid is not deactivated.\n    '
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    mocker.patch('spyder.plugins.editor.widgets.recover.os.remove', side_effect=OSError)
    mock_QMessageBox = mocker.patch('spyder.plugins.editor.widgets.recover.QMessageBox')
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    button = table.cellWidget(0, 2).findChildren(QPushButton)[1]
    button.click()
    with open(osp.join(orig_dir, 'ham.py')) as f:
        assert f.read() == 'ham = "original"\n'
    with open(osp.join(autosave_dir, 'ham.py')) as f:
        assert f.read() == 'ham = "autosave"\n'
    assert mock_QMessageBox.called
    for col in range(table.columnCount()):
        assert table.cellWidget(0, col).isEnabled()

def test_recoverydialog_open_button(qtbot, recovery_env):
    if False:
        print('Hello World!')
    "\n    Test the `Open` button in the recovery dialog.\n\n    Test that after pressing the 'Open' button, `files_to_open` contains\n    the autosave and the original file, and the row in the grid is\n    deactivated.\n    "
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    button = table.cellWidget(0, 2).findChildren(QPushButton)[2]
    button.click()
    assert dialog.files_to_open == [osp.join(orig_dir, 'ham.py'), osp.join(autosave_dir, 'ham.py')]
    for col in range(table.columnCount()):
        assert not table.cellWidget(0, col).isEnabled()

def test_recoverydialog_open_when_no_original(qtbot, recovery_env):
    if False:
        while True:
            i = 10
    '\n    Test the `Open` button when the original file is not known.\n\n    Test that when the user requests to open an autosave file for which the\n    original file is not known, `files_to_open` contains only the autosave\n    file.\n    '
    (orig_dir, autosave_dir, autosave_mapping) = recovery_env
    dialog = RecoveryDialog(autosave_mapping)
    table = dialog.findChild(QTableWidget)
    button = table.cellWidget(2, 2).findChildren(QPushButton)[2]
    button.click()
    assert dialog.files_to_open == [osp.join(autosave_dir, 'cheese.py')]
    for col in range(table.columnCount()):
        assert not table.cellWidget(2, col).isEnabled()