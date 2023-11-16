import string
import functools
import itertools
import operator
import pytest
from qutebrowser.qt.core import QUrl
from qutebrowser.utils import usertypes
import qutebrowser.browser.hints

@pytest.fixture(autouse=True)
def setup(win_registry, mode_manager):
    if False:
        while True:
            i = 10
    pass

@pytest.fixture
def tabbed_browser(tabbed_browser_stubs, web_tab):
    if False:
        return 10
    tb = tabbed_browser_stubs[0]
    tb.widget.tabs = [web_tab]
    tb.widget.current_index = 1
    tb.widget.cur_url = QUrl('https://www.example.com/')
    web_tab.container.expose()
    return tb

def test_show_benchmark(benchmark, tabbed_browser, qtbot, mode_manager):
    if False:
        print('Hello World!')
    'Benchmark showing/drawing of hint labels.'
    tab = tabbed_browser.widget.tabs[0]
    with qtbot.wait_signal(tab.load_finished):
        tab.load_url(QUrl('qute://testdata/data/hints/benchmark.html'))
    manager = qutebrowser.browser.hints.HintManager(win_id=0)

    def bench():
        if False:
            i = 10
            return i + 15
        with qtbot.wait_signal(mode_manager.entered):
            manager.start()
        with qtbot.wait_signal(mode_manager.left):
            mode_manager.leave(usertypes.KeyMode.hint)
    benchmark(bench)

def test_match_benchmark(benchmark, tabbed_browser, qtbot, mode_manager, qapp, config_stub):
    if False:
        i = 10
        return i + 15
    'Benchmark matching of hint labels.'
    tab = tabbed_browser.widget.tabs[0]
    with qtbot.wait_signal(tab.load_finished):
        tab.load_url(QUrl('qute://testdata/data/hints/benchmark.html'))
    config_stub.val.hints.scatter = False
    manager = qutebrowser.browser.hints.HintManager(win_id=0)
    with qtbot.wait_signal(mode_manager.entered):
        manager.start()

    def bench():
        if False:
            i = 10
            return i + 15
        manager.handle_partial_key('a')
        qapp.processEvents()
        manager.handle_partial_key('')
        qapp.processEvents()
    benchmark(bench)
    with qtbot.wait_signal(mode_manager.left):
        mode_manager.leave(usertypes.KeyMode.hint)

@pytest.mark.parametrize('min_len', [0, 3])
@pytest.mark.parametrize('num_chars', [5, 9])
@pytest.mark.parametrize('num_elements', itertools.chain(range(1, 26), [125]))
def test_scattered_hints_count(min_len, num_chars, num_elements):
    if False:
        while True:
            i = 10
    "Test scattered hints function.\n\n    Tests many properties from an invocation of _hint_scattered, including\n\n    1. Hints must be unique\n    2. There can only be two hint lengths, only 1 apart\n    3. There are no unique prefixes for long hints, such as 'la' with no 'l<x>'\n    "
    manager = qutebrowser.browser.hints.HintManager(win_id=0)
    chars = string.ascii_lowercase[:num_chars]
    hints = manager._hint_scattered(min_len, chars, list(range(num_elements)))
    assert len(hints) == len(set(hints))
    assert not any((x for x in hints if len(x) < min_len))
    hint_lens = {len(h) for h in hints}
    assert len(hint_lens) <= 2
    if len(hint_lens) == 2:
        assert abs(functools.reduce(operator.sub, hint_lens)) <= 1
    longest_hint_len = max(hint_lens)
    shortest_hint_len = min(hint_lens)
    longest_hints = [x for x in hints if len(x) == longest_hint_len]
    if min_len < max(hint_lens) - 1:
        count_map = {}
        for x in longest_hints:
            prefix = x[:-1]
            count_map[prefix] = count_map.get(prefix, 0) + 1
        assert all((e != 1 for e in count_map.values()))
    if longest_hint_len > min_len and longest_hint_len > 1:
        assert num_chars ** (longest_hint_len - 1) < num_elements
    assert num_chars ** longest_hint_len >= num_elements
    if longest_hint_len > min_len and longest_hint_len > 1:
        assert num_chars ** (longest_hint_len - 1) < num_elements
        if shortest_hint_len == longest_hint_len:
            assert num_chars ** longest_hint_len - num_elements < len(chars) - 1