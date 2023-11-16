"""Test qutebrowser.qt.machinery."""
import re
import sys
import html
import argparse
import typing
from typing import Any, Optional, List, Dict, Union
import dataclasses
import pytest
from qutebrowser.qt import machinery
MACHINERY_VARS = {'USE_PYQT5', 'USE_PYQT6', 'USE_PYSIDE6', 'IS_QT5', 'IS_QT6', 'IS_PYQT', 'IS_PYSIDE', 'INFO'}
assert set(typing.get_type_hints(machinery).keys()) == MACHINERY_VARS

@pytest.fixture(autouse=True)
def undo_init(monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        while True:
            i = 10
    "Pretend Qt support isn't initialized yet and Qt was never imported."
    monkeypatch.setattr(machinery, '_initialized', False)
    monkeypatch.delenv('QUTE_QT_WRAPPER', raising=False)
    for wrapper in machinery.WRAPPERS:
        monkeypatch.delitem(sys.modules, wrapper, raising=False)
    for var in MACHINERY_VARS:
        monkeypatch.delattr(machinery, var)

@pytest.mark.parametrize('exception', [machinery.Unavailable(), machinery.NoWrapperAvailableError(machinery.SelectionInfo())])
def test_importerror_exceptions(exception: Exception):
    if False:
        return 10
    with pytest.raises(ImportError):
        raise exception

def test_selectioninfo_set_module_error():
    if False:
        i = 10
        return i + 15
    info = machinery.SelectionInfo()
    info.set_module_error('PyQt5', ImportError('Python imploded'))
    assert info == machinery.SelectionInfo(wrapper=None, reason=machinery.SelectionReason.unknown, outcomes={'PyQt5': 'ImportError: Python imploded'})

def test_selectioninfo_use_wrapper():
    if False:
        i = 10
        return i + 15
    info = machinery.SelectionInfo()
    info.use_wrapper('PyQt6')
    assert info == machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.unknown, outcomes={'PyQt6': 'success'})

@pytest.mark.parametrize('info, expected', [(machinery.SelectionInfo(wrapper='PyQt5', reason=machinery.SelectionReason.cli), 'Qt wrapper: PyQt5 (via --qt-wrapper)'), (machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.env), 'Qt wrapper: PyQt6 (via QUTE_QT_WRAPPER)'), (machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.auto, outcomes={'PyQt6': 'success', 'PyQt5': 'ImportError: Python imploded'}), 'Qt wrapper info:\n  PyQt6: success\n  PyQt5: ImportError: Python imploded\n  -> selected: PyQt6 (via autoselect)')])
def test_selectioninfo_str(info: machinery.SelectionInfo, expected: str):
    if False:
        print('Hello World!')
    assert str(info) == expected
    assert info.to_html() == html.escape(expected).replace('\n', '<br>')

@pytest.mark.parametrize('order', [['PyQt5', 'PyQt6'], ['PyQt6', 'PyQt5']])
def test_selectioninfo_str_wrapper_precedence(order: List[str]):
    if False:
        return 10
    'The order of the wrappers should be the same as in machinery.WRAPPERS.'
    info = machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.auto)
    for module in order:
        info.set_module_error(module, ImportError('Python imploded'))
    lines = str(info).splitlines()[1:-1]
    wrappers = [line.split(':')[0].strip() for line in lines]
    assert wrappers == machinery.WRAPPERS

@pytest.fixture
def modules():
    if False:
        i = 10
        return i + 15
    'Return a dict of modules to import-patch, all unavailable by default.'
    return dict.fromkeys(machinery.WRAPPERS, False)

@pytest.mark.parametrize('available, expected', [pytest.param({'PyQt5': ModuleNotFoundError('hiding somewhere'), 'PyQt6': ModuleNotFoundError('hiding somewhere')}, machinery.SelectionInfo(wrapper=None, reason=machinery.SelectionReason.auto, outcomes={'PyQt5': 'ModuleNotFoundError: hiding somewhere', 'PyQt6': 'ModuleNotFoundError: hiding somewhere'}), id='none-available'), pytest.param({'PyQt5': ModuleNotFoundError('hiding somewhere'), 'PyQt6': True}, machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.auto, outcomes={'PyQt6': 'success'}), id='only-pyqt6'), pytest.param({'PyQt5': True, 'PyQt6': ModuleNotFoundError('hiding somewhere')}, machinery.SelectionInfo(wrapper='PyQt5', reason=machinery.SelectionReason.auto, outcomes={'PyQt6': 'ModuleNotFoundError: hiding somewhere', 'PyQt5': 'success'}), id='only-pyqt5'), pytest.param({'PyQt5': True, 'PyQt6': True}, machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.auto, outcomes={'PyQt6': 'success'}), id='both'), pytest.param({'PyQt6': ImportError('Fake ImportError for PyQt6.'), 'PyQt5': True}, machinery.SelectionInfo(wrapper=None, reason=machinery.SelectionReason.auto, outcomes={'PyQt6': 'ImportError: Fake ImportError for PyQt6.'}), id='import-error')])
def test_autoselect(stubs: Any, available: Dict[str, Union[bool, Exception]], expected: machinery.SelectionInfo, monkeypatch: pytest.MonkeyPatch):
    if False:
        while True:
            i = 10
    stubs.ImportFake(available, monkeypatch).patch()
    assert machinery._autoselect_wrapper() == expected

@dataclasses.dataclass
class SelectWrapperCase:
    name: str
    expected: machinery.SelectionInfo
    args: Optional[argparse.Namespace] = None
    env: Optional[str] = None
    override: Optional[str] = None

    def __str__(self):
        if False:
            print('Hello World!')
        return self.name

class TestSelectWrapper:

    @pytest.mark.parametrize('tc', [SelectWrapperCase('pyqt6-arg', args=argparse.Namespace(qt_wrapper='PyQt6'), expected=machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.cli)), SelectWrapperCase('pyqt5-arg', args=argparse.Namespace(qt_wrapper='PyQt5'), expected=machinery.SelectionInfo(wrapper='PyQt5', reason=machinery.SelectionReason.cli)), SelectWrapperCase('pyqt6-arg-empty-env', args=argparse.Namespace(qt_wrapper='PyQt5'), env='', expected=machinery.SelectionInfo(wrapper='PyQt5', reason=machinery.SelectionReason.cli)), SelectWrapperCase('pyqt6-env', env='PyQt6', expected=machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.env)), SelectWrapperCase('pyqt5-env', env='PyQt5', expected=machinery.SelectionInfo(wrapper='PyQt5', reason=machinery.SelectionReason.env)), SelectWrapperCase('pyqt5-arg-pyqt6-env', args=argparse.Namespace(qt_wrapper='PyQt5'), env='PyQt6', expected=machinery.SelectionInfo(wrapper='PyQt5', reason=machinery.SelectionReason.cli)), SelectWrapperCase('pyqt6-arg-pyqt5-env', args=argparse.Namespace(qt_wrapper='PyQt6'), env='PyQt5', expected=machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.cli)), SelectWrapperCase('pyqt6-arg-pyqt6-env', args=argparse.Namespace(qt_wrapper='PyQt6'), env='PyQt6', expected=machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.cli)), SelectWrapperCase('override-only', override='PyQt6', expected=machinery.SelectionInfo(wrapper='PyQt6', reason=machinery.SelectionReason.override)), SelectWrapperCase('override-arg', args=argparse.Namespace(qt_wrapper='PyQt5'), override='PyQt6', expected=machinery.SelectionInfo(wrapper='PyQt5', reason=machinery.SelectionReason.cli)), SelectWrapperCase('override-env', env='PyQt5', override='PyQt6', expected=machinery.SelectionInfo(wrapper='PyQt5', reason=machinery.SelectionReason.env))], ids=str)
    def test_select(self, tc: SelectWrapperCase, monkeypatch: pytest.MonkeyPatch):
        if False:
            return 10
        if tc.env is None:
            monkeypatch.delenv('QUTE_QT_WRAPPER', raising=False)
        else:
            monkeypatch.setenv('QUTE_QT_WRAPPER', tc.env)
        if tc.override is not None:
            monkeypatch.setattr(machinery, '_WRAPPER_OVERRIDE', tc.override)
        assert machinery._select_wrapper(tc.args) == tc.expected

    @pytest.mark.parametrize('args, env', [(None, None), (argparse.Namespace(qt_wrapper=None), None), (argparse.Namespace(qt_wrapper=None), '')])
    def test_autoselect_by_default(self, args: Optional[argparse.Namespace], env: Optional[str], monkeypatch: pytest.MonkeyPatch):
        if False:
            while True:
                i = 10
        'Test that the default behavior is to autoselect a wrapper.\n\n        Autoselection itself is tested further down.\n        '
        if env is None:
            monkeypatch.delenv('QUTE_QT_WRAPPER', raising=False)
        else:
            monkeypatch.setenv('QUTE_QT_WRAPPER', env)
        assert machinery._select_wrapper(args).reason == machinery.SelectionReason.auto

    def test_after_qt_import(self, monkeypatch: pytest.MonkeyPatch):
        if False:
            i = 10
            return i + 15
        monkeypatch.setitem(sys.modules, 'PyQt6', None)
        with pytest.warns(UserWarning, match='PyQt6 already imported'):
            machinery._select_wrapper(args=None)

    def test_invalid_override(self, monkeypatch: pytest.MonkeyPatch):
        if False:
            return 10
        monkeypatch.setattr(machinery, '_WRAPPER_OVERRIDE', 'invalid')
        with pytest.raises(AssertionError):
            machinery._select_wrapper(args=None)

class TestInit:

    @pytest.fixture
    def empty_args(self) -> argparse.Namespace:
        if False:
            i = 10
            return i + 15
        return argparse.Namespace(qt_wrapper=None)

    def test_multiple_implicit(self, monkeypatch: pytest.MonkeyPatch):
        if False:
            while True:
                i = 10
        monkeypatch.setattr(machinery, '_initialized', True)
        machinery.init_implicit()
        machinery.init_implicit()

    def test_multiple_explicit(self, monkeypatch: pytest.MonkeyPatch, empty_args: argparse.Namespace):
        if False:
            return 10
        monkeypatch.setattr(machinery, '_initialized', True)
        with pytest.raises(machinery.Error, match='init\\(\\) already called before application init'):
            machinery.init(args=empty_args)

    @pytest.fixture(params=['auto', '', None])
    def qt_auto_env(self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
        if False:
            return 10
        'Trigger wrapper autoselection via environment variable.\n\n        Autoselection should be used in three scenarios:\n\n        - The environment variable is set to "auto".\n        - The environment variable is set to an empty string.\n        - The environment variable is not set at all.\n\n        We run test_none_available_*() for all three scenarios.\n        '
        if request.param is None:
            monkeypatch.delenv('QUTE_QT_WRAPPER', raising=False)
        else:
            monkeypatch.setenv('QUTE_QT_WRAPPER', request.param)

    def test_none_available_implicit(self, stubs: Any, modules: Dict[str, bool], monkeypatch: pytest.MonkeyPatch, qt_auto_env: None):
        if False:
            return 10
        stubs.ImportFake(modules, monkeypatch).patch()
        message_lines = ['No Qt wrapper was importable.', '', 'Qt wrapper info:', '  PyQt6: ImportError: Fake ImportError for PyQt6.', '  PyQt5: not imported', '  -> selected: None (via autoselect)']
        with pytest.raises(machinery.NoWrapperAvailableError, match=re.escape('\n'.join(message_lines))):
            machinery.init_implicit()

    def test_none_available_explicit(self, stubs: Any, modules: Dict[str, bool], monkeypatch: pytest.MonkeyPatch, empty_args: argparse.Namespace, qt_auto_env: None):
        if False:
            while True:
                i = 10
        stubs.ImportFake(modules, monkeypatch).patch()
        info = machinery.init(args=empty_args)
        assert info == machinery.SelectionInfo(wrapper=None, reason=machinery.SelectionReason.auto, outcomes={'PyQt6': 'ImportError: Fake ImportError for PyQt6.'})

    @pytest.mark.parametrize('selected_wrapper, true_vars', [('PyQt6', ['USE_PYQT6', 'IS_QT6', 'IS_PYQT']), ('PyQt5', ['USE_PYQT5', 'IS_QT5', 'IS_PYQT']), ('PySide6', ['USE_PYSIDE6', 'IS_QT6', 'IS_PYSIDE'])])
    @pytest.mark.parametrize('explicit', [True, False])
    def test_properly(self, monkeypatch: pytest.MonkeyPatch, selected_wrapper: str, true_vars: str, explicit: bool, empty_args: argparse.Namespace):
        if False:
            while True:
                i = 10
        info = machinery.SelectionInfo(wrapper=selected_wrapper, reason=machinery.SelectionReason.fake)
        monkeypatch.setattr(machinery, '_select_wrapper', lambda args: info)
        if explicit:
            ret = machinery.init(empty_args)
            assert ret == info
        else:
            machinery.init_implicit()
        assert machinery.INFO == info
        bool_vars = MACHINERY_VARS - {'INFO'}
        expected_vars = dict.fromkeys(bool_vars, False)
        expected_vars.update(dict.fromkeys(true_vars, True))
        actual_vars = {var: getattr(machinery, var) for var in bool_vars}
        assert expected_vars == actual_vars