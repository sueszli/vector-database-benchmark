"""Tests for caret browsing mode."""
import textwrap
import pytest
from qutebrowser.qt.core import QUrl
from qutebrowser.utils import usertypes
from qutebrowser.browser import browsertab

@pytest.fixture
def caret(web_tab, qtbot, mode_manager):
    if False:
        i = 10
        return i + 15
    web_tab.container.expose()
    with qtbot.wait_signal(web_tab.load_finished, timeout=10000):
        web_tab.load_url(QUrl('qute://testdata/data/caret.html'))
    with qtbot.wait_signal(web_tab.caret.selection_toggled):
        mode_manager.enter(usertypes.KeyMode.caret)
    return web_tab.caret

class Selection:
    """Helper to interact with the caret selection."""

    def __init__(self, qtbot, caret):
        if False:
            i = 10
            return i + 15
        self._qtbot = qtbot
        self._caret = caret

    def check(self, expected, *, strip=False):
        if False:
            for i in range(10):
                print('nop')
        "Check whether we got the expected selection.\n\n        Since (especially on Windows) the selection is empty if we're checking\n        too quickly, we try to read it multiple times.\n        "
        for _ in range(10):
            with self._qtbot.wait_callback() as callback:
                self._caret.selection(callback)
            selection = callback.args[0]
            if selection:
                if strip:
                    selection = selection.strip()
                assert selection == expected
                return
            elif not selection and (not expected):
                return
            self._qtbot.wait(50)
        pytest.fail('Failed to get selection!')

    def check_multiline(self, expected, *, strip=False):
        if False:
            for i in range(10):
                print('nop')
        self.check(textwrap.dedent(expected).strip(), strip=strip)

    def toggle(self, *, line=False):
        if False:
            for i in range(10):
                print('nop')
        'Toggle the selection and return the new selection state.'
        with self._qtbot.wait_signal(self._caret.selection_toggled) as blocker:
            self._caret.toggle_selection(line=line)
        return blocker.args[0]

@pytest.fixture
def selection(qtbot, caret):
    if False:
        return 10
    return Selection(qtbot, caret)

def test_toggle(caret, selection, qtbot):
    if False:
        while True:
            i = 10
    'Make sure calling toggleSelection produces the correct callback values.\n\n    This also makes sure that the SelectionState enum in JS lines up with the\n    Python browsertab.SelectionState enum.\n    '
    assert selection.toggle() == browsertab.SelectionState.normal
    assert selection.toggle(line=True) == browsertab.SelectionState.line
    assert selection.toggle() == browsertab.SelectionState.normal
    assert selection.toggle() == browsertab.SelectionState.none

def test_selection_callback_wrong_mode(qtbot, caplog, webengine_tab, mode_manager):
    if False:
        return 10
    "Test what calling the selection callback outside of caret mode.\n\n    It should be ignored, as something could have left caret mode while the\n    async callback was happening, so we don't want to mess with the status bar.\n    "
    assert mode_manager.mode == usertypes.KeyMode.normal
    with qtbot.assert_not_emitted(webengine_tab.caret.selection_toggled):
        webengine_tab.caret._toggle_sel_translate('normal')
    msg = 'Ignoring caret selection callback in KeyMode.normal'
    assert caplog.messages == [msg]

class TestDocument:

    def test_selecting_entire_document(self, caret, selection):
        if False:
            while True:
                i = 10
        selection.toggle()
        caret.move_to_end_of_document()
        selection.check_multiline('\n            one two three\n            eins zwei drei\n\n            four five six\n            vier fünf sechs\n        ', strip=True)

    def test_moving_to_end_and_start(self, caret, selection):
        if False:
            for i in range(10):
                print('nop')
        caret.move_to_end_of_document()
        caret.move_to_start_of_document()
        selection.toggle()
        caret.move_to_end_of_word()
        selection.check('one')

    def test_moving_to_end_and_start_with_selection(self, caret, selection):
        if False:
            print('Hello World!')
        caret.move_to_end_of_document()
        selection.toggle()
        caret.move_to_start_of_document()
        selection.check_multiline('\n            one two three\n            eins zwei drei\n\n            four five six\n            vier fünf sechs\n        ', strip=True)

class TestBlock:

    def test_selecting_block(self, caret, selection):
        if False:
            for i in range(10):
                print('nop')
        selection.toggle()
        caret.move_to_end_of_next_block()
        selection.check_multiline('\n            one two three\n            eins zwei drei\n        ')

    def test_moving_back_to_the_end_of_prev_block_with_sel(self, caret, selection):
        if False:
            while True:
                i = 10
        caret.move_to_end_of_next_block(2)
        selection.toggle()
        caret.move_to_end_of_prev_block()
        caret.move_to_prev_word()
        selection.check_multiline('\n            drei\n\n            four five six\n        ')

    def test_moving_back_to_the_end_of_prev_block(self, caret, selection):
        if False:
            while True:
                i = 10
        caret.move_to_end_of_next_block(2)
        caret.move_to_end_of_prev_block()
        selection.toggle()
        caret.move_to_prev_word()
        selection.check('drei')

    def test_moving_back_to_the_start_of_prev_block_with_sel(self, caret, selection):
        if False:
            i = 10
            return i + 15
        caret.move_to_end_of_next_block(2)
        selection.toggle()
        caret.move_to_start_of_prev_block()
        selection.check_multiline('\n            eins zwei drei\n\n            four five six\n        ')

    def test_moving_back_to_the_start_of_prev_block(self, caret, selection):
        if False:
            return 10
        caret.move_to_end_of_next_block(2)
        caret.move_to_start_of_prev_block()
        selection.toggle()
        caret.move_to_next_word()
        selection.check('eins ')

    def test_moving_to_the_start_of_next_block_with_sel(self, caret, selection):
        if False:
            for i in range(10):
                print('nop')
        selection.toggle()
        caret.move_to_start_of_next_block()
        selection.check('one two three\n')

    def test_moving_to_the_start_of_next_block(self, caret, selection):
        if False:
            for i in range(10):
                print('nop')
        caret.move_to_start_of_next_block()
        selection.toggle()
        caret.move_to_end_of_word()
        selection.check('eins')

class TestLine:

    def test_selecting_a_line(self, caret, selection):
        if False:
            i = 10
            return i + 15
        selection.toggle()
        caret.move_to_end_of_line()
        selection.check('one two three')

    def test_moving_and_selecting_a_line(self, caret, selection):
        if False:
            while True:
                i = 10
        caret.move_to_next_line()
        selection.toggle()
        caret.move_to_end_of_line()
        selection.check('eins zwei drei')

    def test_selecting_next_line(self, caret, selection):
        if False:
            for i in range(10):
                print('nop')
        selection.toggle()
        caret.move_to_next_line()
        selection.check('one two three\n')

    def test_moving_to_end_and_to_start_of_line(self, caret, selection):
        if False:
            while True:
                i = 10
        caret.move_to_end_of_line()
        caret.move_to_start_of_line()
        selection.toggle()
        caret.move_to_end_of_word()
        selection.check('one')

    def test_selecting_a_line_backwards(self, caret, selection):
        if False:
            return 10
        caret.move_to_end_of_line()
        selection.toggle()
        caret.move_to_start_of_line()
        selection.check('one two three')

    def test_selecting_previous_line(self, caret, selection):
        if False:
            while True:
                i = 10
        caret.move_to_next_line()
        selection.toggle()
        caret.move_to_prev_line()
        selection.check('one two three\n')

    def test_moving_to_previous_line(self, caret, selection):
        if False:
            return 10
        caret.move_to_next_line()
        caret.move_to_prev_line()
        selection.toggle()
        caret.move_to_next_line()
        selection.check('one two three\n')

class TestWord:

    def test_selecting_a_word(self, caret, selection):
        if False:
            print('Hello World!')
        selection.toggle()
        caret.move_to_end_of_word()
        selection.check('one')

    def test_moving_to_end_and_selecting_a_word(self, caret, selection):
        if False:
            while True:
                i = 10
        caret.move_to_end_of_word()
        selection.toggle()
        caret.move_to_end_of_word()
        selection.check(' two')

    def test_moving_to_next_word_and_selecting_a_word(self, caret, selection):
        if False:
            print('Hello World!')
        caret.move_to_next_word()
        selection.toggle()
        caret.move_to_end_of_word()
        selection.check('two')

    def test_moving_to_next_word_and_selecting_until_next_word(self, caret, selection):
        if False:
            for i in range(10):
                print('nop')
        caret.move_to_next_word()
        selection.toggle()
        caret.move_to_next_word()
        selection.check('two ')

    def test_moving_to_previous_word_and_selecting_a_word(self, caret, selection):
        if False:
            return 10
        caret.move_to_end_of_word()
        selection.toggle()
        caret.move_to_prev_word()
        selection.check('one')

    def test_moving_to_previous_word(self, caret, selection):
        if False:
            for i in range(10):
                print('nop')
        caret.move_to_end_of_word()
        caret.move_to_prev_word()
        selection.toggle()
        caret.move_to_end_of_word()
        selection.check('one')

class TestChar:

    def test_selecting_a_char(self, caret, selection):
        if False:
            print('Hello World!')
        selection.toggle()
        caret.move_to_next_char()
        selection.check('o')

    def test_moving_and_selecting_a_char(self, caret, selection):
        if False:
            while True:
                i = 10
        caret.move_to_next_char()
        selection.toggle()
        caret.move_to_next_char()
        selection.check('n')

    def test_selecting_previous_char(self, caret, selection):
        if False:
            print('Hello World!')
        caret.move_to_end_of_word()
        selection.toggle()
        caret.move_to_prev_char()
        selection.check('e')

    def test_moving_to_previous_char(self, caret, selection):
        if False:
            return 10
        caret.move_to_end_of_word()
        caret.move_to_prev_char()
        selection.toggle()
        caret.move_to_end_of_word()
        selection.check('e')

def test_drop_selection(caret, selection):
    if False:
        print('Hello World!')
    selection.toggle()
    caret.move_to_end_of_word()
    caret.drop_selection()
    selection.check('')

class TestSearch:

    @pytest.mark.no_xvfb
    def test_yanking_a_searched_line(self, caret, selection, mode_manager, web_tab, qtbot):
        if False:
            print('Hello World!')
        mode_manager.leave(usertypes.KeyMode.caret)
        with qtbot.wait_callback() as callback:
            web_tab.search.search('fiv', result_cb=callback)
        callback.assert_called_with(True)
        mode_manager.enter(usertypes.KeyMode.caret)
        caret.move_to_end_of_line()
        selection.check('five six')

    @pytest.mark.no_xvfb
    def test_yanking_a_searched_line_with_multiple_matches(self, caret, selection, mode_manager, web_tab, qtbot):
        if False:
            i = 10
            return i + 15
        mode_manager.leave(usertypes.KeyMode.caret)
        with qtbot.wait_callback() as callback:
            web_tab.search.search('w', result_cb=callback)
        callback.assert_called_with(True)
        with qtbot.wait_callback() as callback:
            web_tab.search.next_result(callback=callback)
        callback.assert_called_with(browsertab.SearchNavigationResult.found)
        mode_manager.enter(usertypes.KeyMode.caret)
        caret.move_to_end_of_line()
        selection.check('wei drei')

class TestFollowSelected:
    LOAD_STARTED_DELAY = 50

    @pytest.fixture(params=[True, False], autouse=True)
    def toggle_js(self, request, config_stub):
        if False:
            return 10
        config_stub.val.content.javascript.enabled = request.param

    def test_follow_selected_without_a_selection(self, qtbot, caret, selection, web_tab, mode_manager):
        if False:
            for i in range(10):
                print('nop')
        caret.move_to_next_word()
        mode_manager.leave(usertypes.KeyMode.caret)
        with qtbot.wait_signal(caret.follow_selected_done):
            with qtbot.assert_not_emitted(web_tab.load_started, wait=self.LOAD_STARTED_DELAY):
                caret.follow_selected()

    def test_follow_selected_with_text(self, qtbot, caret, selection, web_tab):
        if False:
            for i in range(10):
                print('nop')
        caret.move_to_next_word()
        selection.toggle()
        caret.move_to_end_of_word()
        with qtbot.wait_signal(caret.follow_selected_done):
            with qtbot.assert_not_emitted(web_tab.load_started, wait=self.LOAD_STARTED_DELAY):
                caret.follow_selected()

    def test_follow_selected_with_link(self, caret, selection, config_stub, qtbot, web_tab):
        if False:
            for i in range(10):
                print('nop')
        selection.toggle()
        caret.move_to_end_of_word()
        with qtbot.wait_signal(web_tab.load_finished):
            with qtbot.wait_signal(caret.follow_selected_done):
                caret.follow_selected()
        assert web_tab.url().path() == '/data/hello.txt'

class TestReverse:

    def test_does_not_change_selection(self, caret, selection):
        if False:
            return 10
        selection.toggle()
        caret.reverse_selection()
        selection.check('')

    def test_repetition_of_movement_results_in_empty_selection(self, caret, selection):
        if False:
            print('Hello World!')
        selection.toggle()
        caret.move_to_end_of_word()
        caret.reverse_selection()
        caret.move_to_end_of_word()
        selection.check('')

    def test_reverse(self, caret, selection):
        if False:
            print('Hello World!')
        selection.toggle()
        caret.move_to_end_of_word()
        caret.reverse_selection()
        caret.move_to_next_char()
        selection.check('ne')
        caret.reverse_selection()
        caret.move_to_next_char()
        selection.check('ne ')
        caret.move_to_end_of_line()
        selection.check('ne two three')
        caret.reverse_selection()
        caret.move_to_start_of_line()
        selection.check('one two three')

class TestLineSelection:

    def test_toggle(self, caret, selection):
        if False:
            return 10
        selection.toggle(line=True)
        selection.check('one two three')

    def test_toggle_untoggle(self, caret, selection):
        if False:
            print('Hello World!')
        selection.toggle()
        selection.check('')
        selection.toggle(line=True)
        selection.check('one two three')
        selection.toggle()
        selection.check('one two three')

    def test_from_center(self, caret, selection):
        if False:
            i = 10
            return i + 15
        caret.move_to_next_char(4)
        selection.toggle(line=True)
        selection.check('one two three')

    def test_more_lines(self, caret, selection):
        if False:
            i = 10
            return i + 15
        selection.toggle(line=True)
        caret.move_to_next_line(2)
        selection.check_multiline('\n            one two three\n            eins zwei drei\n\n            four five six\n        ', strip=True)

    def test_not_selecting_char(self, caret, selection):
        if False:
            i = 10
            return i + 15
        selection.toggle(line=True)
        caret.move_to_next_char()
        selection.check('one two three')
        caret.move_to_prev_char()
        selection.check('one two three')

    def test_selecting_prev_next_word(self, caret, selection):
        if False:
            for i in range(10):
                print('nop')
        selection.toggle(line=True)
        caret.move_to_next_word()
        selection.check('one two three')
        caret.move_to_prev_word()
        selection.check('one two three')

    def test_selecting_end_word(self, caret, selection):
        if False:
            while True:
                i = 10
        selection.toggle(line=True)
        caret.move_to_end_of_word()
        selection.check('one two three')

    def test_selecting_prev_next_line(self, caret, selection):
        if False:
            for i in range(10):
                print('nop')
        selection.toggle(line=True)
        caret.move_to_next_line()
        selection.check_multiline('\n            one two three\n            eins zwei drei\n        ', strip=True)
        caret.move_to_prev_line()
        selection.check('one two three')

    def test_not_selecting_start_end_line(self, caret, selection):
        if False:
            print('Hello World!')
        selection.toggle(line=True)
        caret.move_to_end_of_line()
        selection.check('one two three')
        caret.move_to_start_of_line()
        selection.check('one two three')

    def test_selecting_block(self, caret, selection):
        if False:
            print('Hello World!')
        selection.toggle(line=True)
        caret.move_to_end_of_next_block()
        selection.check_multiline('\n            one two three\n            eins zwei drei\n        ', strip=True)

    @pytest.mark.not_mac(reason='https://github.com/qutebrowser/qutebrowser/issues/5459')
    def test_selecting_start_end_document(self, caret, selection):
        if False:
            i = 10
            return i + 15
        selection.toggle(line=True)
        caret.move_to_end_of_document()
        selection.check_multiline('\n            one two three\n            eins zwei drei\n\n            four five six\n            vier fünf sechs\n        ', strip=True)
        caret.move_to_start_of_document()
        selection.check('one two three')