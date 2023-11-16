import contextlib
import pytest
from qutebrowser.qt.core import Qt
from qutebrowser.mainwindow import messageview
from qutebrowser.utils import usertypes, message

@pytest.fixture
def view(qtbot, config_stub):
    if False:
        print('Hello World!')
    config_stub.val.messages.timeout = 100
    mv = messageview.MessageView()
    qtbot.add_widget(mv)
    return mv

@pytest.mark.parametrize('level', [usertypes.MessageLevel.info, usertypes.MessageLevel.warning, usertypes.MessageLevel.error])
@pytest.mark.flaky
def test_single_message(qtbot, view, level):
    if False:
        for i in range(10):
            print('nop')
    with qtbot.wait_exposed(view, timeout=5000):
        view.show_message(message.MessageInfo(level, 'test'))
    assert view._messages[0].isVisible()

def test_message_hiding(qtbot, view):
    if False:
        i = 10
        return i + 15
    'Messages should be hidden after the timer times out.'
    with qtbot.wait_signal(view._clear_timer.timeout):
        view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test'))
    assert not view._messages

def test_size_hint(view):
    if False:
        i = 10
        return i + 15
    'The message height should increase with more messages.'
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test1'))
    height1 = view.sizeHint().height()
    assert height1 > 0
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test2'))
    height2 = view.sizeHint().height()
    assert height2 == height1 * 2

def test_word_wrap(view, qtbot):
    if False:
        for i in range(10):
            print('nop')
    'A long message should be wrapped.'
    with qtbot.wait_signal(view._clear_timer.timeout):
        view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'short'))
        assert len(view._messages) == 1
        height1 = view.sizeHint().height()
        assert height1 > 0
    text = 'Athene, the bright-eyed goddess, answered him at once: Father of us all, Son of Cronos, Highest King, clearly that man deserved to be destroyed: so let all be destroyed who act as he did. But my heart aches for Odysseus, wise but ill fated, who suffers far from his friends on an island deep in the sea.'
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, text))
    assert len(view._messages) == 1
    height2 = view.sizeHint().height()
    assert height2 > height1
    assert view._messages[0].wordWrap()

@pytest.mark.parametrize('rich, higher, expected_format', [(True, True, Qt.TextFormat.RichText), (False, False, Qt.TextFormat.PlainText), (None, False, Qt.TextFormat.PlainText)])
@pytest.mark.parametrize('replace', ['test', None])
def test_rich_text(view, qtbot, rich, higher, expected_format, replace):
    if False:
        while True:
            i = 10
    'Rich text should be rendered appropriately.\n\n    This makes sure the title has been rendered as plain text by comparing the\n    heights of the two widgets. To ensure consistent results, we disable word-wrapping.\n    '
    level = usertypes.MessageLevel.info
    text = 'with <h1>markup</h1>'
    text2 = 'with <h1>markup</h1> 2'
    info1 = message.MessageInfo(level, text, replace=replace)
    info2 = message.MessageInfo(level, text2, replace=replace, rich=rich)
    ctx = qtbot.wait_signal(view._clear_timer.timeout) if replace is None else contextlib.nullcontext()
    with ctx:
        view.show_message(info1)
        assert len(view._messages) == 1
        view._messages[0].setWordWrap(False)
        height1 = view.sizeHint().height()
        assert height1 > 0
        assert view._messages[0].textFormat() == Qt.TextFormat.PlainText
    view.show_message(info2)
    assert len(view._messages) == 1
    view._messages[0].setWordWrap(False)
    height2 = view.sizeHint().height()
    assert height2 > 0
    assert view._messages[0].textFormat() == expected_format
    if higher:
        assert height2 > height1
    else:
        assert height2 == height1

@pytest.mark.parametrize('info1, info2, count', [(message.MessageInfo(usertypes.MessageLevel.info, 'test'), message.MessageInfo(usertypes.MessageLevel.info, 'test'), 1), (message.MessageInfo(usertypes.MessageLevel.info, 'test'), message.MessageInfo(usertypes.MessageLevel.info, 'test2'), 2), (message.MessageInfo(usertypes.MessageLevel.info, 'test'), message.MessageInfo(usertypes.MessageLevel.error, 'test'), 2), (message.MessageInfo(usertypes.MessageLevel.info, 'test', rich=True), message.MessageInfo(usertypes.MessageLevel.info, 'test', rich=False), 2), (message.MessageInfo(usertypes.MessageLevel.info, 'test'), message.MessageInfo(usertypes.MessageLevel.info, 'test', replace='test'), 2)])
def test_show_message_twice(view, info1, info2, count):
    if False:
        return 10
    'Show the exact same message twice -> only one should be shown.'
    view.show_message(info1)
    view.show_message(info2)
    assert len(view._messages) == count

def test_show_message_twice_after_first_disappears(qtbot, view):
    if False:
        i = 10
        return i + 15
    'Show the same message twice after the first is gone.'
    with qtbot.wait_signal(view._clear_timer.timeout):
        view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test'))
    assert not view._messages
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test'))
    assert len(view._messages) == 1

def test_changing_timer_with_messages_shown(qtbot, view, config_stub):
    if False:
        for i in range(10):
            print('nop')
    'When we change messages.timeout, the timer should be restarted.'
    config_stub.val.messages.timeout = 900000
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test'))
    with qtbot.wait_signal(view._clear_timer.timeout):
        config_stub.val.messages.timeout = 100

@pytest.mark.parametrize('count, expected', [(1, 100), (3, 300), (5, 500), (7, 500)])
def test_show_multiple_messages_longer(view, count, expected):
    if False:
        for i in range(10):
            print('nop')
    'When there are multiple messages, messages should be shown longer.\n\n    There is an upper maximum to avoid messages never disappearing.\n    '
    for message_number in range(1, count + 1):
        view.show_message(message.MessageInfo(usertypes.MessageLevel.info, f'test {message_number}'))
    assert view._clear_timer.interval() == expected

@pytest.mark.parametrize('replace1, replace2, length', [(None, None, 2), ('testid', 'testid', 1), (None, 'testid', 2), ('testid', None, 2), ('testid1', 'testid2', 2)])
def test_replaced_messages(view, replace1, replace2, length):
    if False:
        for i in range(10):
            print('nop')
    'Show two stack=False messages which should replace each other.'
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test', replace=replace1))
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test 2', replace=replace2))
    assert len(view._messages) == length

def test_replacing_different_severity(view):
    if False:
        return 10
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test', replace='testid'))
    with pytest.raises(AssertionError):
        view.show_message(message.MessageInfo(usertypes.MessageLevel.error, 'test 2', replace='testid'))

def test_replacing_changed_text(view):
    if False:
        for i in range(10):
            print('nop')
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test', replace='testid'))
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test 2'))
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test 3', replace='testid'))
    assert len(view._messages) == 2
    assert view._messages[0].text() == 'test 3'
    assert view._messages[1].text() == 'test 2'

def test_replacing_geometry(qtbot, view):
    if False:
        for i in range(10):
            print('nop')
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test', replace='testid'))
    with qtbot.wait_signal(view.update_geometry):
        view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test 2', replace='testid'))

@pytest.mark.parametrize('button, count', [(Qt.MouseButton.LeftButton, 0), (Qt.MouseButton.MiddleButton, 0), (Qt.MouseButton.RightButton, 0), (Qt.MouseButton.BackButton, 2)])
def test_click_messages(qtbot, view, button, count):
    if False:
        print('Hello World!')
    'Messages should disappear when we click on them.'
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test mouse click'))
    view.show_message(message.MessageInfo(usertypes.MessageLevel.info, 'test mouse click 2'))
    qtbot.mousePress(view, button)
    assert len(view._messages) == count