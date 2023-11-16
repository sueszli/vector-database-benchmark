"""Tests for editor URI and mailto go to handling."""
import os
import tempfile
from qtpy.QtCore import Qt, QPoint, QTimer
from qtpy.QtGui import QDesktopServices, QTextCursor
from qtpy.QtWidgets import QMessageBox
import pytest
from spyder.utils.vcs import get_git_remotes
HERE = os.path.abspath(__file__)
TEST_FOLDER = os.path.abspath(os.path.dirname(__file__))
(_, TEMPFILE_PATH) = tempfile.mkstemp()
TEST_FILES = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) if f.endswith('.py')]
TEST_FILE_ABS = TEST_FILES[0].replace(' ', '%20')
TEST_FILE_REL = 'conftest.py'

@pytest.mark.parametrize('params', [('file://{}\n'.format(TEMPFILE_PATH), 'file://' + TEMPFILE_PATH, TEMPFILE_PATH, 'file://' + TEMPFILE_PATH), ('"file://{}"\n'.format(TEST_FILE_ABS), 'file://' + TEST_FILE_ABS, TEST_FILE_ABS, 'file://' + TEST_FILE_ABS), ('"file://./{}"\n'.format(TEST_FILE_REL), 'file://./' + TEST_FILE_REL, os.path.join(TEST_FOLDER, TEST_FILE_REL), 'file://./' + TEST_FILE_REL), ('"file:///not%20there"', 'file:///not%20there', '/not%20there', 'file:///not%20there'), ('"file:///not_there"', 'file:///not_there', '/not_there', 'file:///not_there'), ('" https://google.com"\n', 'https://google.com', None, 'https://google.com'), ('# https://google.com"\n', 'https://google.com', None, 'https://google.com'), ('" mailto:some@email.com"\n', 'mailto:some@email.com', None, 'mailto:some@email.com'), ('# mailto:some@email.com\n', 'mailto:some@email.com', None, 'mailto:some@email.com'), ('some@email.com\n', 'some@email.com', None, 'mailto:some@email.com'), ('# some@email.com\n', 'some@email.com', None, 'mailto:some@email.com'), ('# gl:gitlab-org/gitlab-ce#62529\n', 'gl:gitlab-org/gitlab-ce#62529', None, 'https://gitlab.com/gitlab-org/gitlab-ce/issues/62529'), ('# bb:birkenfeld/pygments-main#1516\n', 'bb:birkenfeld/pygments-main#1516', None, 'https://bitbucket.org/birkenfeld/pygments-main/issues/1516'), ('# gh:spyder-ide/spyder#123\n', 'gh:spyder-ide/spyder#123', None, 'https://github.com/spyder-ide/spyder/issues/123'), ('# gh:spyder-ide/spyder#123\n', 'gh:spyder-ide/spyder#123', None, 'https://github.com/spyder-ide/spyder/issues/123'), pytest.param(('# gh-123\n', 'gh-123', HERE, 'https://github.com/spyder-ide/spyder/issues/123'), marks=pytest.mark.skipif(not get_git_remotes(HERE), reason='not in a git repository'))])
def test_goto_uri(qtbot, codeeditor, mocker, params):
    if False:
        i = 10
        return i + 15
    'Test that the uri search is working correctly.'
    code_editor = codeeditor
    code_editor.show()
    mocker.patch.object(QDesktopServices, 'openUrl')
    (param, expected_output_1, full_file_path, expected_output_2) = params
    if full_file_path:
        code_editor.filename = full_file_path
    code_editor.set_text(param)
    code_editor.moveCursor(QTextCursor.Start)
    (x, y) = code_editor.get_coordinates('cursor')
    point = code_editor.calculate_real_position(QPoint(x + 30, y))
    code_editor.moveCursor(QTextCursor.End)
    qtbot.mouseMove(code_editor, point, delay=500)
    with qtbot.waitSignal(code_editor.sig_uri_found, timeout=3000) as blocker:
        qtbot.keyPress(code_editor, Qt.Key_Control, delay=500)
        args = blocker.args
        print([param, expected_output_1])
        print([args])
        output_1 = args[0]
        output_2 = code_editor.go_to_uri_from_cursor(expected_output_1)
        assert expected_output_1 in output_1
        assert expected_output_2 == output_2

def test_goto_uri_project_root_path(qtbot, codeeditor, mocker, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test that the uri search is working correctly.'
    code_editor = codeeditor
    code_editor.show()
    mock_project_dir = str(tmpdir)
    expected_output_path = os.path.join(mock_project_dir, 'some-file.txt')
    with open(expected_output_path, 'w') as fh:
        fh.write('BOOM!\n')
    code_editor.set_current_project_path(mock_project_dir)
    code_editor.filename = 'foo.txt'
    mocker.patch.object(QDesktopServices, 'openUrl')
    code_editor.set_text('file://^/some-file.txt')
    code_editor.moveCursor(QTextCursor.Start)
    (x, y) = code_editor.get_coordinates('cursor')
    point = code_editor.calculate_real_position(QPoint(x + 23, y))
    code_editor.moveCursor(QTextCursor.End)
    qtbot.mouseMove(code_editor, point, delay=500)
    with qtbot.waitSignal(code_editor.sig_file_uri_preprocessed, timeout=3000) as blocker:
        qtbot.keyPress(code_editor, Qt.Key_Control, delay=500)
        args = blocker.args
        assert args[0] == expected_output_path
    qtbot.wait(500)
    expected_output_path = os.path.expanduser('~/some-file.txt')
    code_editor.set_current_project_path()
    with qtbot.waitSignal(code_editor.sig_file_uri_preprocessed, timeout=3000) as blocker:
        qtbot.keyPress(code_editor, Qt.Key_Control, delay=500)
        args = blocker.args
        assert args[0] == expected_output_path

def test_goto_uri_message_box(qtbot, codeeditor, mocker):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that a message box is displayed when the shorthand issue notation is\n    used (gh-123) indicating the user that the file is not under a repository\n    '
    code_editor = codeeditor
    code_editor.filename = TEMPFILE_PATH
    code_editor._last_hover_pattern_key = 'issue'

    def interact():
        if False:
            i = 10
            return i + 15
        msgbox = code_editor.findChild(QMessageBox)
        assert msgbox
        qtbot.keyClick(msgbox, Qt.Key_Return)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.setInterval(500)
    timer.timeout.connect(interact)
    timer.start()
    code_editor.go_to_uri_from_cursor('gh-123')
    code_editor.filename = None
    code_editor._last_hover_pattern_key = None
    code_editor._last_hover_pattern_text = None

def test_pattern_highlight_regression(qtbot, codeeditor):
    if False:
        while True:
            i = 10
    'Fix regression on spyder-ide/spyder#11376.'
    code_editor = codeeditor
    code_editor.show()
    code_editor.set_text("'''\ngl-")
    qtbot.wait(500)
    code_editor.moveCursor(QTextCursor.End)
    qtbot.wait(500)
    qtbot.keyClick(code_editor, Qt.Key_1)
    qtbot.wait(1000)