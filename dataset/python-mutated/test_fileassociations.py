"""
Tests for explorer plugin utilities.
"""
import os
import sys
from qtpy.QtCore import Qt, QTimer
import pytest
from spyder.plugins.explorer.widgets.fileassociations import ApplicationsDialog, InputTextDialog

@pytest.mark.order(1)
def test_input_text_dialog(qtbot):
    if False:
        i = 10
        return i + 15
    widget = InputTextDialog()
    qtbot.addWidget(widget)
    widget.show()
    widget.validate()
    assert not widget.button_ok.isEnabled()
    widget.set_text('hello')
    widget.validate()
    assert widget.button_ok.isEnabled()
    widget.set_text('')
    widget.set_regex_validation('hello')
    qtbot.keyClicks(widget.lineedit, 'hello world!')
    assert widget.text() == 'hello'
    assert widget.button_ok.isEnabled()
    widget.validate()
    widget.set_text('')
    widget.set_regex_validation('hello')
    qtbot.keyClicks(widget.lineedit, 'hell')
    assert not widget.button_ok.isEnabled()
    widget.validate()

@pytest.mark.order(1)
def test_apps_dialog(qtbot, tmp_path):
    if False:
        while True:
            i = 10
    widget = ApplicationsDialog()
    qtbot.addWidget(widget)
    widget.show()
    if os.name == 'nt':
        ext = '.exe'
    elif sys.platform == 'darwin':
        ext = '.app'
    else:
        ext = '.desktop'
    mock_apps = {'some app 1': '/some/fake/some app 1' + ext, 'some app 2': '/some/fake/path/some app 2' + ext, 'some app 3': '/some/fake/path/some app 3' + ext}
    widget.setup(mock_apps)
    qtbot.keyClicks(widget.edit_filter, '1')
    count_hidden = 0
    for row in range(widget.list.count()):
        item = widget.list.item(row)
        count_hidden += int(item.isHidden())
    assert count_hidden == 2
    widget.list.setCurrentItem(widget.list.item(0))
    assert widget.application_name == 'some app 1'
    assert widget.application_path == '/some/fake/some app 1' + ext
    widget.set_extension('.hello')
    assert '.hello' in widget.label.text()
    widget.edit_filter.setText('')
    count_hidden = 0
    for row in range(widget.list.count()):
        item = widget.list.item(row)
        count_hidden += int(item.isHidden())
    assert count_hidden == 0
    fpath = '/some/other/path'
    widget.browse(fpath)
    assert widget.list.count() == 3
    if os.name == 'nt':
        path_obj = tmp_path / ('some-new-app' + ext)
        path_obj.write_bytes(b'\x00\x00')
        fpath = str(path_obj)
    elif sys.platform == 'darwin':
        path_obj = tmp_path / ('some-new-app' + ext)
        path_obj.mkdir()
        fpath = str(path_obj)
    else:
        path_obj = tmp_path / ('some-new-app' + ext)
        path_obj.write_text(u'\n[Desktop Entry]\nName=Suer app\nType=Application\nExec=/something/bleerp\nIcon=/blah/blah.xpm\n')
        fpath = str(path_obj)
    widget.browse(fpath)
    assert widget.list.count() == 4
    widget.browse(fpath)
    assert widget.list.count() == 4

def create_timer(func, interval=500):
    if False:
        print('Hello World!')
    'Helper function to help interact with modal dialogs.'
    timer = QTimer()
    timer.setInterval(interval)
    timer.setSingleShot(True)
    timer.timeout.connect(func)
    timer.start()
    return timer

@pytest.mark.order(1)
def test_file_assoc_widget(file_assoc_widget):
    if False:
        return 10
    (qtbot, widget) = file_assoc_widget
    assert widget.data == widget.test_data
    extension = 'blooper.foo,'

    def interact_with_dialog_1():
        if False:
            for i in range(10):
                print('nop')
        qtbot.keyClicks(widget._dlg_input.lineedit, extension)
        assert widget._dlg_input.lineedit.text() == extension
        assert not widget._dlg_input.button_ok.isEnabled()
        qtbot.keyClick(widget._dlg_input.button_cancel, Qt.Key_Return)
    _ = create_timer(interact_with_dialog_1)
    qtbot.mouseClick(widget.button_add, Qt.LeftButton)
    extension = '*.zpam,MANIFEST.in'

    def interact_with_dialog_2():
        if False:
            for i in range(10):
                print('nop')
        qtbot.keyClicks(widget._dlg_input.lineedit, extension)
        qtbot.keyClick(widget._dlg_input.button_ok, Qt.Key_Return)
    _ = create_timer(interact_with_dialog_2)
    qtbot.mouseClick(widget.button_add, Qt.LeftButton)
    assert widget.list_extensions.count() == 3
    assert widget.list_extensions.item(2).text() == extension
    widget.add_association(value='mehh')
    assert widget.list_extensions.count() == 3
    widget.add_association(value='*.boom')
    assert widget.list_extensions.count() == 4
    widget.add_association(value='*.csv')
    assert widget.list_extensions.count() == 4
    widget._add_association(value='*.csv')
    assert widget.list_extensions.count() == 4
    extension = '*.zpam'

    def interact_with_dialog_3():
        if False:
            i = 10
            return i + 15
        widget._dlg_input.lineedit.clear()
        qtbot.keyClicks(widget._dlg_input.lineedit, extension)
        qtbot.keyClick(widget._dlg_input.button_ok, Qt.Key_Return)
    _ = create_timer(interact_with_dialog_3)
    qtbot.mouseClick(widget.button_edit, Qt.LeftButton)
    assert widget.list_extensions.count() == 4
    assert widget.list_extensions.item(2).text() == extension
    qtbot.mouseClick(widget.button_remove, Qt.LeftButton)
    assert widget.list_extensions.count() == 3
    widget.list_applications.setCurrentRow(1)
    qtbot.mouseClick(widget.button_default, Qt.LeftButton)
    assert 'App name 2' in widget.list_applications.item(0).text()

    def interact_with_dialog_4():
        if False:
            while True:
                i = 10
        assert not widget._dlg_applications.button_ok.isEnabled()
        count = widget._dlg_applications.list.count()
        if count > 0:
            widget._dlg_applications.list.setCurrentRow(count - 1)
            qtbot.keyClick(widget._dlg_applications.button_ok, Qt.Key_Return)
        else:
            qtbot.keyClick(widget._dlg_applications.button_cancel, Qt.Key_Return)
    _ = create_timer(interact_with_dialog_4)
    qtbot.mouseClick(widget.button_add_application, Qt.LeftButton)
    count = widget.list_applications.count()
    assert count in [2, 3]
    (app_name, app_path) = widget.test_data['*.csv'][0]
    widget._add_application(app_name, app_path)
    count = widget.list_applications.count()
    assert count in [2, 3]
    widget.list_applications.setCurrentRow(0)
    qtbot.mouseClick(widget.button_remove_application, Qt.LeftButton)
    count = widget.list_applications.count()
    assert count in [1, 2]
    assert 'App name 1' in widget.list_applications.item(0).text()