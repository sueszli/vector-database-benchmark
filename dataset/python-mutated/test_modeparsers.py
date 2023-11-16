"""Tests for mode parsers."""
from qutebrowser.qt.core import Qt
from qutebrowser.qt.gui import QKeySequence
import pytest
from qutebrowser.keyinput import modeparsers, keyutils
from qutebrowser.config import configexc

@pytest.fixture
def commandrunner(stubs):
    if False:
        return 10
    return stubs.FakeCommandRunner()

class TestsNormalKeyParser:

    @pytest.fixture(autouse=True)
    def patch_stuff(self, monkeypatch, stubs, keyinput_bindings):
        if False:
            print('Hello World!')
        'Set up mocks and read the test config.'
        monkeypatch.setattr('qutebrowser.keyinput.basekeyparser.usertypes.Timer', stubs.FakeTimer)

    @pytest.fixture
    def keyparser(self, commandrunner):
        if False:
            return 10
        kp = modeparsers.NormalKeyParser(win_id=0, commandrunner=commandrunner)
        return kp

    def test_keychain(self, keyparser, commandrunner):
        if False:
            while True:
                i = 10
        'Test valid keychain.'
        chain = keyutils.KeySequence.parse('zba')
        for info in chain:
            keyparser.handle(info.to_event())
        assert commandrunner.commands == [('message-info ba', None)]
        assert not keyparser._sequence

    def test_partial_keychain_timeout(self, keyparser, config_stub, qtbot, commandrunner):
        if False:
            i = 10
            return i + 15
        'Test partial keychain timeout.'
        config_stub.val.input.partial_timeout = 100
        timer = keyparser._partial_timer
        assert not timer.isActive()
        keyparser.handle(keyutils.KeyInfo(Qt.Key.Key_B, Qt.KeyboardModifier.NoModifier).to_event())
        assert timer.isSingleShot()
        assert timer.interval() == 100
        assert timer.isActive()
        assert not commandrunner.commands
        assert keyparser._sequence == keyutils.KeySequence.parse('b')
        with qtbot.wait_signal(keyparser.keystring_updated) as blocker:
            timer.timeout.emit()
        assert not commandrunner.commands
        assert not keyparser._sequence
        assert blocker.args == ['']

class TestHintKeyParser:

    @pytest.fixture
    def hintmanager(self, stubs):
        if False:
            for i in range(10):
                print('nop')
        return stubs.FakeHintManager()

    @pytest.fixture
    def keyparser(self, config_stub, key_config_stub, commandrunner, hintmanager):
        if False:
            for i in range(10):
                print('nop')
        return modeparsers.HintKeyParser(win_id=0, hintmanager=hintmanager, commandrunner=commandrunner)

    @pytest.mark.parametrize('bindings, keychain, prefix, hint', [(['aa', 'as'], 'as', 'a', 'as'), (['21', '22'], '<Num+2><Num+2>', '2', '22'), (['äa', 'äs'], 'äs', 'ä', 'äs'), (['не', 'на'], 'не', '<Н>', 'не')])
    def test_match(self, keyparser, hintmanager, bindings, keychain, prefix, hint, pyqt_enum_workaround):
        if False:
            i = 10
            return i + 15
        with pyqt_enum_workaround(keyutils.KeyParseError):
            keyparser.update_bindings(bindings)
        seq = keyutils.KeySequence.parse(keychain)
        assert len(seq) == 2
        match = keyparser.handle(seq[0].to_event())
        assert match == QKeySequence.SequenceMatch.PartialMatch
        assert hintmanager.keystr == prefix
        match = keyparser.handle(seq[1].to_event())
        assert match == QKeySequence.SequenceMatch.ExactMatch
        assert hintmanager.keystr == hint

    def test_match_key_mappings(self, config_stub, keyparser, hintmanager, pyqt_enum_workaround):
        if False:
            return 10
        with pyqt_enum_workaround(configexc.ValidationError):
            config_stub.val.bindings.key_mappings = {'α': 'a', 'σ': 's'}
        keyparser.update_bindings(['aa', 'as'])
        seq = keyutils.KeySequence.parse('ασ')
        assert len(seq) == 2
        match = keyparser.handle(seq[0].to_event())
        assert match == QKeySequence.SequenceMatch.PartialMatch
        assert hintmanager.keystr == 'a'
        match = keyparser.handle(seq[1].to_event())
        assert match == QKeySequence.SequenceMatch.ExactMatch
        assert hintmanager.keystr == 'as'

    def test_command(self, keyparser, config_stub, hintmanager, commandrunner):
        if False:
            while True:
                i = 10
        config_stub.val.bindings.commands = {'hint': {'abc': 'message-info abc'}}
        keyparser.update_bindings(['xabcy'])
        steps = [(Qt.Key.Key_X, QKeySequence.SequenceMatch.PartialMatch, 'x'), (Qt.Key.Key_A, QKeySequence.SequenceMatch.PartialMatch, ''), (Qt.Key.Key_B, QKeySequence.SequenceMatch.PartialMatch, ''), (Qt.Key.Key_C, QKeySequence.SequenceMatch.ExactMatch, '')]
        for (key, expected_match, keystr) in steps:
            info = keyutils.KeyInfo(key, Qt.KeyboardModifier.NoModifier)
            match = keyparser.handle(info.to_event())
            assert match == expected_match
            assert hintmanager.keystr == keystr
            if key != Qt.Key.Key_C:
                assert not commandrunner.commands
        assert commandrunner.commands == [('message-info abc', None)]