"""Tests for the class and method panel selector."""
from flaky import flaky
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QListView
import pytest
text = '\nfrom collections import OrderedDict\n\n\nclass SomeObject:  # Block number 4\n\n    def __init__(self):\n        pass\n\n    def hello_1(self):  # Block number 9\n        pass\n\n    def hello_2(self):\n        pass\n\n\nclass SomeOtherObject:\n\n    def __init__(self):\n        pass\n\n    def hello_3(self):\n        pass\n\n        def nested_func():\n            pass\n'

@pytest.mark.order(2)
@flaky(max_runs=5)
def test_class_func_selector(completions_codeeditor, qtbot):
    if False:
        return 10
    (code_editor, _) = completions_codeeditor
    panel = code_editor.classfuncdropdown
    panel.setVisible(True)
    code_editor.toggle_automatic_completions(False)
    code_editor.set_text(text)
    qtbot.wait(3000)
    class_names = [item['name'] for item in panel.classes]
    func_names = [item['name'] for item in panel.funcs]
    assert len(panel.classes) == 2
    assert len(panel.funcs) == 6
    assert 'SomeObject' in class_names
    assert 'SomeOtherObject' in class_names
    assert '__init__' in func_names
    assert 'nested_func' in func_names
    for _ in range(7):
        qtbot.keyPress(code_editor, Qt.Key_Down)
    assert panel.class_cb.currentText() == 'SomeObject'
    assert panel.method_cb.currentText() == 'SomeObject.__init__'
    for _ in range(18):
        qtbot.keyPress(code_editor, Qt.Key_Down)
    assert panel.class_cb.currentText() == 'SomeOtherObject'
    assert panel.method_cb.currentText() == 'SomeOtherObject.hello_3.nested_func'
    qtbot.mouseClick(panel.class_cb, Qt.LeftButton, pos=QPoint(5, 5))
    listview = panel.class_cb.findChild(QListView)
    qtbot.keyPress(listview, Qt.Key_Up)
    qtbot.keyPress(listview, Qt.Key_Return)
    qtbot.wait(1000)
    cursor = code_editor.textCursor()
    assert cursor.blockNumber() == 4
    assert panel.method_cb.currentIndex() == 0
    panel.method_cb.setFocus()
    qtbot.mouseClick(panel.method_cb, Qt.LeftButton, pos=QPoint(5, 5))
    listview = panel.method_cb.findChild(QListView)
    qtbot.keyPress(listview, Qt.Key_Down)
    qtbot.keyPress(listview, Qt.Key_Down)
    qtbot.keyPress(listview, Qt.Key_Return)
    qtbot.wait(1000)
    cursor = code_editor.textCursor()
    assert cursor.blockNumber() == 9
    assert panel.class_cb.currentIndex() == 1
    panel.setVisible(False)
    code_editor.toggle_automatic_completions(True)