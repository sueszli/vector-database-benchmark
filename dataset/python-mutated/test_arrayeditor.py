"""
Tests for the array editor.
"""
import os
import sys
from unittest.mock import Mock, ANY
from flaky import flaky
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from qtpy.QtCore import Qt
from scipy.io import loadmat
from spyder.plugins.variableexplorer.widgets.arrayeditor import ArrayEditor, ArrayModel
HERE = os.path.dirname(os.path.realpath(__file__))

def launch_arrayeditor(data, title='', xlabels=None, ylabels=None):
    if False:
        while True:
            i = 10
    'Helper routine to launch an arrayeditor and return its result.'
    dlg = ArrayEditor()
    assert dlg.setup_and_check(data, title, xlabels=xlabels, ylabels=ylabels)
    dlg.show()
    dlg.accept()
    return dlg.get_value()

@pytest.fixture
def setup_arrayeditor(qtbot, data):
    if False:
        return 10
    'Setups an arrayeditor.'
    dlg = ArrayEditor()
    dlg.setup_and_check(data)
    dlg.show()
    qtbot.addWidget(dlg)
    return dlg

def test_object_arrays(qtbot):
    if False:
        i = 10
        return i + 15
    'Test that object arrays are working properly.'
    arr = np.array([u'a', 1, [2]], dtype=object)
    assert_array_equal(arr, launch_arrayeditor(arr, 'object array'))

@pytest.mark.parametrize('data', [np.array([[np.array([1, 2])], 2], dtype=object)])
def test_object_arrays_display(setup_arrayeditor):
    if False:
        i = 10
        return i + 15
    '\n    Test that value_to_display is being used to display the values of\n    object arrays.\n    '
    dlg = setup_arrayeditor
    idx = dlg.arraywidget.model.index(0, 0)
    assert u'[Numpy array]' == dlg.arraywidget.model.data(idx)

@pytest.mark.parametrize('data', [loadmat(os.path.join(HERE, 'issue_11216.mat'))['S']])
def test_attribute_errors(setup_arrayeditor):
    if False:
        print('Hello World!')
    "\n    Verify that we don't get a AttributeError for certain structured arrays.\n\n    Fixes spyder-ide/spyder#11216 .\n    "
    dlg = setup_arrayeditor
    data = loadmat(os.path.join(HERE, 'issue_11216.mat'))
    contents = dlg.arraywidget.model.get_value(dlg.arraywidget.model.index(0, 0))
    assert_array_equal(contents, data['S'][0][0][0])

@pytest.mark.parametrize('data', [np.ones(2, dtype=[('X', 'f8', (2, 10)), ('S', 'S10')])])
def test_type_errors(setup_arrayeditor, qtbot):
    if False:
        print('Hello World!')
    "\n    Verify that we don't get a TypeError for certain structured arrays.\n\n    Fixes spyder-ide/spyder#5254.\n    "
    dlg = setup_arrayeditor
    qtbot.keyClick(dlg.arraywidget.view, Qt.Key_Down, modifier=Qt.ShiftModifier)
    contents = dlg.arraywidget.model.get_value(dlg.arraywidget.model.index(0, 0))
    assert_array_equal(contents, np.ones(10))

@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Only works on Linux')
@pytest.mark.parametrize('data', [np.array([1, 2, 3], dtype=np.float32)])
def test_arrayeditor_format(setup_arrayeditor, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Changes the format of the array and validates its selected content.'
    dlg = setup_arrayeditor
    qtbot.keyClick(dlg.arraywidget.view, Qt.Key_Down, modifier=Qt.ShiftModifier)
    qtbot.keyClick(dlg.arraywidget.view, Qt.Key_Down, modifier=Qt.ShiftModifier)
    contents = dlg.arraywidget.view._sel_to_text(dlg.arraywidget.view.selectedIndexes())
    assert contents == '1\n2\n'
    dlg.arraywidget.view.model().set_format_spec('.18e')
    assert dlg.arraywidget.view.model().get_format_spec() == '.18e'
    qtbot.keyClick(dlg.arraywidget.view, Qt.Key_Down, modifier=Qt.ShiftModifier)
    qtbot.keyClick(dlg.arraywidget.view, Qt.Key_Down, modifier=Qt.ShiftModifier)
    contents = dlg.arraywidget.view._sel_to_text(dlg.arraywidget.view.selectedIndexes())
    assert contents == '1.000000000000000000e+00\n2.000000000000000000e+00\n'

@pytest.mark.parametrize('data', [np.array([10000])])
def test_arrayeditor_format_thousands(setup_arrayeditor):
    if False:
        return 10
    'Check that format can include thousands separator.'
    model = setup_arrayeditor.arraywidget.model
    model.set_format_spec(',.2f')
    assert model.data(model.index(0, 0)) == '10,000.00'

def test_arrayeditor_with_inf_array(qtbot, recwarn):
    if False:
        print('Hello World!')
    'See: spyder-ide/spyder#8093'
    arr = np.array([np.inf])
    res = launch_arrayeditor(arr, 'inf array')
    assert len(recwarn) == 0
    assert arr == res

def test_arrayeditor_with_string_array(qtbot):
    if False:
        i = 10
        return i + 15
    arr = np.array(['kjrekrjkejr'])
    assert arr == launch_arrayeditor(arr, 'string array')

def test_arrayeditor_with_unicode_array(qtbot):
    if False:
        print('Hello World!')
    arr = np.array([u'ñññéáíó'])
    assert arr == launch_arrayeditor(arr, 'unicode array')

def test_arrayeditor_with_masked_array(qtbot):
    if False:
        while True:
            i = 10
    arr = np.ma.array([[1, 0], [1, 0]], mask=[[True, False], [False, False]])
    assert_array_equal(arr, launch_arrayeditor(arr, 'masked array'))

def test_arrayeditor_with_record_array(qtbot):
    if False:
        print('Hello World!')
    arr = np.zeros((2, 2), {'names': ('red', 'green', 'blue'), 'formats': (np.float32, np.float32, np.float32)})
    assert_array_equal(arr, launch_arrayeditor(arr, 'record array'))

@pytest.mark.skipif(not os.name == 'nt', reason='It segfaults sometimes on Linux')
def test_arrayeditor_with_record_array_with_titles(qtbot):
    if False:
        return 10
    arr = np.array([(0, 0.0), (0, 0.0), (0, 0.0)], dtype=[(('title 1', 'x'), '|i1'), (('title 2', 'y'), '>f4')])
    assert_array_equal(arr, launch_arrayeditor(arr, 'record array with titles'))

def test_arrayeditor_with_float_array(qtbot):
    if False:
        print('Hello World!')
    arr = np.random.rand(5, 5)
    assert_array_equal(arr, launch_arrayeditor(arr, 'float array', xlabels=['a', 'b', 'c', 'd', 'e']))

def test_arrayeditor_with_complex_array(qtbot):
    if False:
        print('Hello World!')
    arr = np.round(np.random.rand(5, 5) * 10) + np.round(np.random.rand(5, 5) * 10) * 1j
    assert_array_equal(arr, launch_arrayeditor(arr, 'complex array', xlabels=np.linspace(-12, 12, 5), ylabels=np.linspace(-12, 12, 5)))

def test_arrayeditor_with_bool_array(qtbot):
    if False:
        i = 10
        return i + 15
    arr_in = np.array([True, False, True])
    arr_out = launch_arrayeditor(arr_in, 'bool array')
    assert arr_in is arr_out

def test_arrayeditor_with_int8_array(qtbot):
    if False:
        i = 10
        return i + 15
    arr = np.array([1, 2, 3], dtype='int8')
    assert_array_equal(arr, launch_arrayeditor(arr, 'int array'))

def test_arrayeditor_with_float16_array(qtbot):
    if False:
        while True:
            i = 10
    arr = np.zeros((5, 5), dtype=np.float16)
    assert_array_equal(arr, launch_arrayeditor(arr, 'float16 array'))

def test_arrayeditor_with_3d_array(qtbot):
    if False:
        print('Hello World!')
    arr = np.zeros((3, 3, 4))
    arr[0, 0, 0] = 1
    arr[0, 0, 1] = 2
    arr[0, 0, 2] = 3
    assert_array_equal(arr, launch_arrayeditor(arr, '3D array'))

def test_arrayeditor_with_empty_3d_array(qtbot):
    if False:
        for i in range(10):
            print('nop')
    arr = np.zeros((0, 10, 2))
    assert_array_equal(arr, launch_arrayeditor(arr, '3D array'))
    arr = np.zeros((1, 10, 2))
    assert_array_equal(arr, launch_arrayeditor(arr, '3D array'))

def test_arrayeditor_edit_1d_array(qtbot):
    if False:
        i = 10
        return i + 15
    exp_arr = np.array([1, 0, 2, 3, 4])
    arr = np.arange(0, 5)
    dlg = ArrayEditor()
    assert dlg.setup_and_check(arr, '1D array', xlabels=None, ylabels=None)
    with qtbot.waitExposed(dlg):
        dlg.show()
    view = dlg.arraywidget.view
    qtbot.keyPress(view, Qt.Key_Down)
    qtbot.keyPress(view, Qt.Key_Up)
    qtbot.keyClicks(view, '1')
    qtbot.keyPress(view, Qt.Key_Down)
    qtbot.keyClicks(view, '0')
    qtbot.keyPress(view, Qt.Key_Down)
    qtbot.keyPress(view, Qt.Key_Return)
    assert np.sum(exp_arr == dlg.get_value()) == 5

@pytest.mark.skipif(sys.platform == 'darwin', reason='It fails on macOS')
def test_arrayeditor_edit_2d_array(qtbot):
    if False:
        print('Hello World!')
    arr = np.ones((3, 3))
    diff_arr = arr.copy()
    dlg = ArrayEditor()
    assert dlg.setup_and_check(arr, '2D array', xlabels=None, ylabels=None)
    with qtbot.waitExposed(dlg):
        dlg.show()
    view = dlg.arraywidget.view
    qtbot.keyPress(view, Qt.Key_Down)
    qtbot.keyPress(view, Qt.Key_Right)
    qtbot.keyClicks(view, '3')
    qtbot.keyPress(view, Qt.Key_Down)
    qtbot.keyPress(view, Qt.Key_Right)
    qtbot.keyClicks(view, '0')
    qtbot.keyPress(view, Qt.Key_Left)
    qtbot.keyPress(view, Qt.Key_Return)
    assert np.sum(diff_arr != dlg.get_value()) == 2

@pytest.mark.skipif(sys.platform.startswith('linux'), reason='Sometimes fails on Linux ')
def test_arrayeditor_edit_complex_array(qtbot):
    if False:
        for i in range(10):
            print('nop')
    'See: spyder-ide/spyder#7848'
    cnum = -1 + 0.5j
    arr = (np.random.random((10, 10)) - 0.5) * cnum
    dlg = ArrayEditor()
    assert dlg.setup_and_check(arr, '2D complex array', xlabels=None, ylabels=None)
    with qtbot.waitExposed(dlg):
        dlg.show()
    view = dlg.arraywidget.view
    qtbot.keyPress(view, Qt.Key_Down)
    qtbot.wait(300)
    cell_editor = view.viewport().focusWidget()
    qtbot.keyClicks(cell_editor, str(cnum))
    qtbot.keyPress(cell_editor, Qt.Key_Return)
    dlg.accept()

def test_arraymodel_set_data_overflow(monkeypatch):
    if False:
        return 10
    '\n    Test that entry of an overflowing integer is caught and handled properly.\n\n    Unit regression test for spyder-ide/spyder#6114.\n    '
    MockQMessageBox = Mock()
    attr_to_patch = 'spyder.plugins.variableexplorer.widgets.arrayeditor.QMessageBox'
    monkeypatch.setattr(attr_to_patch, MockQMessageBox)
    if not os.name == 'nt':
        int32_bit_exponent = 66
    else:
        int32_bit_exponent = 34
    test_parameters = [(1, np.int32, int32_bit_exponent), (2, np.int64, 66)]
    for (idx, int_type, bit_exponent) in test_parameters:
        test_array = np.array([[5], [6], [7], [3], [4]], dtype=int_type)
        model = ArrayModel(test_array.copy())
        index = model.createIndex(0, 2)
        assert not model.setData(index, str(int(2 ** bit_exponent)))
        MockQMessageBox.critical.assert_called_with(ANY, 'Error', ANY)
        assert MockQMessageBox.critical.call_count == idx
        assert np.sum(test_array == model._data) == len(test_array)

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin', reason='It fails on macOS')
def test_arrayeditor_edit_overflow(qtbot, monkeypatch):
    if False:
        return 10
    '\n    Test that entry of an overflowing integer is caught and handled properly.\n\n    Integration regression test for spyder-ide/spyder#6114.\n    '
    MockQMessageBox = Mock()
    attr_to_patch = 'spyder.plugins.variableexplorer.widgets.arrayeditor.QMessageBox'
    monkeypatch.setattr(attr_to_patch, MockQMessageBox)
    if not os.name == 'nt':
        int32_bit_exponent = 66
    else:
        int32_bit_exponent = 34
    test_parameters = [(1, np.int32, int32_bit_exponent), (2, np.int64, 66)]
    expected_array = np.array([5, 6, 7, 3, 4])
    for (idx, int_type, bit_exponent) in test_parameters:
        test_array = np.arange(0, 5).astype(int_type)
        dialog = ArrayEditor()
        assert dialog.setup_and_check(test_array, '1D array', xlabels=None, ylabels=None)
        with qtbot.waitExposed(dialog):
            dialog.show()
        view = dialog.arraywidget.view
        qtbot.keyClick(view, Qt.Key_Down)
        qtbot.keyClick(view, Qt.Key_Up)
        qtbot.keyClicks(view, '5')
        qtbot.keyClick(view, Qt.Key_Down)
        qtbot.keyClick(view, Qt.Key_Space)
        qtbot.keyClicks(view.focusWidget(), str(int(2 ** bit_exponent)))
        qtbot.keyClick(view.focusWidget(), Qt.Key_Down)
        MockQMessageBox.critical.assert_called_with(ANY, 'Error', ANY)
        assert MockQMessageBox.critical.call_count == idx
        qtbot.keyClicks(view, '7')
        qtbot.keyClick(view, Qt.Key_Up)
        qtbot.keyClicks(view, '6')
        qtbot.keyClick(view, Qt.Key_Down)
        qtbot.wait(200)
        dialog.accept()
        qtbot.wait(500)
        assert np.sum(expected_array == dialog.get_value()) == len(expected_array)
if __name__ == '__main__':
    pytest.main()