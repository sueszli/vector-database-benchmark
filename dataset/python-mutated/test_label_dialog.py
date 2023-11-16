import pytest
from qtpy import QtCore
from qtpy import QtWidgets
from labelme.widgets import LabelDialog
from labelme.widgets import LabelQLineEdit

@pytest.mark.gui
def test_LabelQLineEdit(qtbot):
    if False:
        print('Hello World!')
    list_widget = QtWidgets.QListWidget()
    list_widget.addItems(['cat', 'dog', 'person'])
    widget = LabelQLineEdit()
    widget.setListWidget(list_widget)
    qtbot.addWidget(widget)
    item = widget.list_widget.findItems('cat', QtCore.Qt.MatchExactly)[0]
    widget.list_widget.setCurrentItem(item)
    assert widget.list_widget.currentItem().text() == 'cat'
    qtbot.keyPress(widget, QtCore.Qt.Key_Down)
    assert widget.list_widget.currentItem().text() == 'dog'
    qtbot.keyPress(widget, QtCore.Qt.Key_P)
    qtbot.keyPress(widget, QtCore.Qt.Key_E)
    qtbot.keyPress(widget, QtCore.Qt.Key_R)
    qtbot.keyPress(widget, QtCore.Qt.Key_S)
    qtbot.keyPress(widget, QtCore.Qt.Key_O)
    qtbot.keyPress(widget, QtCore.Qt.Key_N)
    assert widget.text() == 'person'

@pytest.mark.gui
def test_LabelDialog_addLabelHistory(qtbot):
    if False:
        print('Hello World!')
    labels = ['cat', 'dog', 'person']
    widget = LabelDialog(labels=labels, sort_labels=True)
    qtbot.addWidget(widget)
    widget.addLabelHistory('bicycle')
    assert widget.labelList.count() == 4
    widget.addLabelHistory('bicycle')
    assert widget.labelList.count() == 4
    item = widget.labelList.item(0)
    assert item.text() == 'bicycle'

@pytest.mark.gui
def test_LabelDialog_popUp(qtbot):
    if False:
        i = 10
        return i + 15
    labels = ['cat', 'dog', 'person']
    widget = LabelDialog(labels=labels, sort_labels=True)
    qtbot.addWidget(widget)

    def interact():
        if False:
            print('Hello World!')
        qtbot.keyClick(widget.edit, QtCore.Qt.Key_P)
        qtbot.keyClick(widget.edit, QtCore.Qt.Key_Enter)
        qtbot.keyClick(widget.edit, QtCore.Qt.Key_Enter)
    QtCore.QTimer.singleShot(500, interact)
    (label, flags, group_id, description) = widget.popUp('cat')
    assert label == 'person'
    assert flags == {}
    assert group_id is None
    assert description == ''

    def interact():
        if False:
            return 10
        qtbot.keyClick(widget.edit, QtCore.Qt.Key_Enter)
        qtbot.keyClick(widget.edit, QtCore.Qt.Key_Enter)
    QtCore.QTimer.singleShot(500, interact)
    (label, flags, group_id, description) = widget.popUp()
    assert label == 'person'
    assert flags == {}
    assert group_id is None
    assert description == ''

    def interact():
        if False:
            return 10
        qtbot.keyClick(widget.edit, QtCore.Qt.Key_Up)
        qtbot.keyClick(widget.edit, QtCore.Qt.Key_Enter)
        qtbot.keyClick(widget.edit, QtCore.Qt.Key_Enter)
    QtCore.QTimer.singleShot(500, interact)
    (label, flags, group_id, description) = widget.popUp()
    assert label == 'dog'
    assert flags == {}
    assert group_id is None
    assert description == ''