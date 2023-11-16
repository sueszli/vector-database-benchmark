"""Tests for the Array Builder Widget."""
import sys
from qtpy.QtCore import Qt
import pytest
from spyder.widgets.arraybuilder import ArrayBuilderDialog

@pytest.fixture
def botinline(qtbot):
    if False:
        return 10
    dialog = ArrayBuilderDialog(inline=True)
    qtbot.addWidget(dialog)
    dialog.show()
    return (qtbot, dialog, dialog.array_widget)

@pytest.fixture
def botinlinefloat(qtbot):
    if False:
        print('Hello World!')
    dialog = ArrayBuilderDialog(inline=True, force_float=True)
    qtbot.addWidget(dialog)
    dialog.show()
    return (qtbot, dialog, dialog.array_widget)

@pytest.fixture
def botarray(qtbot):
    if False:
        i = 10
        return i + 15
    dialog = ArrayBuilderDialog(inline=False)
    qtbot.addWidget(dialog)
    dialog.show()
    return (qtbot, dialog, dialog.array_widget)

def test_array_inline_array(botinline):
    if False:
        i = 10
        return i + 15
    (qtbot, dialog, widget) = botinline
    qtbot.keyClicks(widget, '1 2 3  4 5 6')
    qtbot.keyPress(widget, Qt.Key_Return)
    value = dialog.text()
    assert value == 'np.array([[1, 2, 3],\n          [4, 5, 6]])'

def test_array_inline_matrix(botinline):
    if False:
        i = 10
        return i + 15
    (qtbot, dialog, widget) = botinline
    qtbot.keyClicks(widget, '4 5 6  7 8 9')
    qtbot.keyPress(widget, Qt.Key_Return, modifier=Qt.ControlModifier)
    value = dialog.text()
    assert value == 'np.matrix([[4, 5, 6],\n           [7, 8, 9]])'

def test_array_inline_array_invalid(botinline):
    if False:
        while True:
            i = 10
    (qtbot, dialog, widget) = botinline
    qtbot.keyClicks(widget, '1 2  3 4  5 6 7')
    qtbot.keyPress(widget, Qt.Key_Return)
    dialog.update_warning()
    assert not dialog.is_valid()

def test_array_inline_1d_array(botinline):
    if False:
        print('Hello World!')
    (qtbot, dialog, widget) = botinline
    qtbot.keyClicks(widget, '4 5 6')
    qtbot.keyPress(widget, Qt.Key_Return, modifier=Qt.ControlModifier)
    value = dialog.text()
    assert value == 'np.matrix([4, 5, 6])'

def test_array_inline_nan_array(botinline):
    if False:
        for i in range(10):
            print('nop')
    (qtbot, dialog, widget) = botinline
    qtbot.keyClicks(widget, '4 nan 6 8 9')
    qtbot.keyPress(widget, Qt.Key_Return, modifier=Qt.ControlModifier)
    value = dialog.text()
    assert value == 'np.matrix([4, np.nan, 6, 8, 9])'

def test_array_inline_inf_array(botinline):
    if False:
        for i in range(10):
            print('nop')
    (qtbot, dialog, widget) = botinline
    qtbot.keyClicks(widget, '4 inf 6 8 9')
    qtbot.keyPress(widget, Qt.Key_Return, modifier=Qt.ControlModifier)
    value = dialog.text()
    assert value == 'np.matrix([4, np.inf, 6, 8, 9])'

def test_array_inline_force_float_array(botinlinefloat):
    if False:
        print('Hello World!')
    (qtbot, dialog, widget) = botinlinefloat
    qtbot.keyClicks(widget, '4 5 6 8 9')
    qtbot.keyPress(widget, Qt.Key_Return, modifier=Qt.ControlModifier)
    value = dialog.text()
    assert value == 'np.matrix([4.0, 5.0, 6.0, 8.0, 9.0])'

def test_array_inline_force_float_error_array(botinlinefloat):
    if False:
        i = 10
        return i + 15
    (qtbot, dialog, widget) = botinlinefloat
    qtbot.keyClicks(widget, '4 5 6 a 9')
    qtbot.keyPress(widget, Qt.Key_Return, modifier=Qt.ControlModifier)
    value = dialog.text()
    assert value == 'np.matrix([4.0, 5.0, 6.0, a, 9.0])'

def test_array_table_array(botarray):
    if False:
        print('Hello World!')
    (qtbot, dialog, widget) = botarray
    qtbot.keyClick(widget, Qt.Key_1)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_2)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Backtab)
    qtbot.keyClick(widget, Qt.Key_3)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_4)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_5)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_6)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Return, modifier=Qt.NoModifier)
    value = dialog.text()
    assert value == 'np.array([[1, 2, 3],\n          [4, 5, 6]])'

def test_array_table_matrix(botarray):
    if False:
        for i in range(10):
            print('nop')
    (qtbot, dialog, widget) = botarray
    qtbot.keyClick(widget, Qt.Key_1)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_2)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Backtab)
    qtbot.keyClick(widget, Qt.Key_3)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_4)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_5)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_6)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Return, modifier=Qt.ControlModifier)
    value = dialog.text()
    assert value == 'np.matrix([[1, 2, 3],\n           [4, 5, 6]])'

def test_array_table_array_empty_items(botarray):
    if False:
        print('Hello World!')
    (qtbot, dialog, widget) = botarray
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_2)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Backtab)
    qtbot.keyClick(widget, Qt.Key_3)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_5)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_6)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Return, modifier=Qt.NoModifier)
    value = dialog.text()
    assert value == 'np.array([[0, 2, 3],\n          [0, 5, 6]])'

def test_array_table_array_spaces_in_item(botarray):
    if False:
        print('Hello World!')
    (qtbot, dialog, widget) = botarray
    qtbot.keyClicks(widget, '   ')
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_2)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Backtab)
    qtbot.keyClick(widget, Qt.Key_3)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_5)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_6)
    qtbot.keyClick(widget, Qt.Key_Tab)
    qtbot.keyClick(widget, Qt.Key_Return, modifier=Qt.NoModifier)
    value = dialog.text()
    assert value == 'np.array([[0, 2, 3],\n          [0, 5, 6]])'

@pytest.mark.skipif(sys.platform == 'darwin', reason='It fails on macOS')
def test_array_table_matrix_empty(botarray):
    if False:
        print('Hello World!')
    (qtbot, dialog, widget) = botarray
    qtbot.keyClick(widget, Qt.Key_Return, modifier=Qt.NoModifier)
    value = dialog.text()
    assert value == ''