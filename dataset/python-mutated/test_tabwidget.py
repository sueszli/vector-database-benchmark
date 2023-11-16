"""Tests for the custom TabWidget/TabBar."""
import functools
import pytest
from unittest.mock import Mock
from qutebrowser.qt.gui import QIcon, QPixmap
from qutebrowser.mainwindow import tabwidget
from qutebrowser.utils import usertypes

class TestTabWidget:
    """Tests for TabWidget."""

    @pytest.fixture
    def widget(self, qtbot, monkeypatch, config_stub):
        if False:
            for i in range(10):
                print('nop')
        w = tabwidget.TabWidget(0)
        qtbot.add_widget(w)
        monkeypatch.setattr(tabwidget.objects, 'backend', usertypes.Backend.QtWebKit)
        w.show()
        return w

    def test_small_icon_doesnt_crash(self, widget, qtbot, fake_web_tab):
        if False:
            while True:
                i = 10
        "Test that setting a small icon doesn't produce a crash.\n\n        Regression test for #1015.\n        "
        pixmap = QPixmap(72, 1)
        icon = QIcon(pixmap)
        tab = fake_web_tab()
        widget.addTab(tab, icon, 'foobar')
        with qtbot.wait_exposed(widget):
            widget.show()

    def test_tab_size_same(self, widget, fake_web_tab):
        if False:
            return 10
        'Ensure by default, all tab sizes are the same.'
        num_tabs = 10
        for i in range(num_tabs):
            widget.addTab(fake_web_tab(), 'foobar' + str(i))
        first_size = widget.tabBar().tabSizeHint(0)
        first_size_min = widget.tabBar().minimumTabSizeHint(0)
        for i in range(num_tabs):
            assert first_size == widget.tabBar().tabSizeHint(i)
            assert first_size_min == widget.tabBar().minimumTabSizeHint(i)

    @pytest.fixture
    def paint_spy(self, monkeypatch):
        if False:
            print('Hello World!')
        spy = Mock()
        monkeypatch.setattr(tabwidget, 'QStylePainter', spy)
        return spy

    def test_tab_text_edlided_for_narrow_tabs(self, paint_spy, widget, fake_web_tab):
        if False:
            for i in range(10):
                print('nop')
        'Make sure text gets elided for narrow tabs.'
        widget.setMaximumWidth(100)
        widget.addTab(fake_web_tab(), 'one two three four')
        fake_paint_event = Mock()
        fake_paint_event.region.return_value.intersects.return_value = True
        widget.tabBar().paintEvent(fake_paint_event)
        style_opt = paint_spy.return_value.drawControl.call_args_list[0][0][1]
        assert len(style_opt.text) < len(widget.tabBar().tabText(0))
        assert style_opt.text.endswith('…')
        assert len(style_opt.text) > len('…')

    def test_tab_text_not_edlided_for_wide_tabs(self, paint_spy, widget, fake_web_tab):
        if False:
            i = 10
            return i + 15
        "Make sure text doesn't get elided for wide tabs."
        widget.setMaximumWidth(200)
        widget.addTab(fake_web_tab(), 'one two three four')
        fake_paint_event = Mock()
        fake_paint_event.region.return_value.intersects.return_value = True
        widget.tabBar().paintEvent(fake_paint_event)
        style_opt = paint_spy.return_value.drawControl.call_args_list[0][0][1]
        assert style_opt.text.endswith(widget.tabBar().tabText(0))

    @pytest.mark.parametrize('shrink_pinned', [True, False])
    @pytest.mark.parametrize('vertical', [True, False])
    def test_pinned_size(self, widget, fake_web_tab, config_stub, shrink_pinned, vertical):
        if False:
            return 10
        'Ensure by default, pinned min sizes are forced to title.\n\n        If pinned.shrink is not true, then all tabs should be the same\n\n        If tabs are vertical, all tabs should be the same'
        num_tabs = 10
        for i in range(num_tabs):
            widget.addTab(fake_web_tab(), 'foobar' + str(i))
        config_stub.val.tabs.title.format_pinned = '_' * 10
        config_stub.val.tabs.title.format = '_' * 2
        config_stub.val.tabs.pinned.shrink = shrink_pinned
        if vertical:
            config_stub.val.tabs.width = 50
            config_stub.val.tabs.position = 'left'
        pinned_num = [1, num_tabs - 1]
        for num in pinned_num:
            tab = widget.widget(num)
            tab.set_pinned(True)
        first_size = widget.tabBar().tabSizeHint(0)
        first_size_min = widget.tabBar().minimumTabSizeHint(0)
        for i in range(num_tabs):
            if i in pinned_num and shrink_pinned and (not vertical):
                assert first_size.width() > widget.tabBar().tabSizeHint(i).width()
                assert first_size_min.width() < widget.tabBar().minimumTabSizeHint(i).width()
            else:
                assert first_size == widget.tabBar().tabSizeHint(i)
                assert first_size_min == widget.tabBar().minimumTabSizeHint(i)

    @pytest.mark.parametrize('num_tabs', [4, 10, 50, 100])
    def test_update_tab_titles_benchmark(self, benchmark, widget, qtbot, fake_web_tab, num_tabs):
        if False:
            return 10
        'Benchmark for update_tab_titles.'
        for i in range(num_tabs):
            widget.addTab(fake_web_tab(), 'foobar' + str(i))
        with qtbot.wait_exposed(widget):
            widget.show()
        benchmark(widget.update_tab_titles)

    def test_tab_min_width(self, widget, fake_web_tab, config_stub, qtbot):
        if False:
            for i in range(10):
                print('nop')
        widget.addTab(fake_web_tab(), 'foobar')
        widget.addTab(fake_web_tab(), 'foobar1')
        min_size = widget.tabBar().tabRect(0).width() + 10
        config_stub.val.tabs.min_width = min_size
        assert widget.tabBar().tabRect(0).width() == min_size

    def test_tab_max_width(self, widget, fake_web_tab, config_stub, qtbot):
        if False:
            print('Hello World!')
        widget.addTab(fake_web_tab(), 'foobar')
        max_size = widget.tabBar().tabRect(0).width() - 10
        config_stub.val.tabs.max_width = max_size
        assert widget.tabBar().tabRect(0).width() == max_size

    def test_tab_stays_hidden(self, widget, fake_web_tab, config_stub):
        if False:
            print('Hello World!')
        assert widget.tabBar().isVisible()
        config_stub.val.tabs.show = 'never'
        assert not widget.tabBar().isVisible()
        for i in range(12):
            widget.addTab(fake_web_tab(), 'foobar' + str(i))
        assert not widget.tabBar().isVisible()

    @pytest.mark.parametrize('num_tabs', [4, 70])
    @pytest.mark.parametrize('rev', [True, False])
    def test_add_remove_tab_benchmark(self, benchmark, widget, qtbot, fake_web_tab, num_tabs, rev):
        if False:
            for i in range(10):
                print('nop')
        'Benchmark for addTab and removeTab.'

        def _run_bench():
            if False:
                i = 10
                return i + 15
            with qtbot.wait_exposed(widget):
                widget.show()
            for i in range(num_tabs):
                idx = i if rev else 0
                widget.insertTab(idx, fake_web_tab(), 'foobar' + str(i))
            to_del = range(num_tabs)
            if rev:
                to_del = reversed(to_del)
            for i in to_del:
                widget.removeTab(i)
        benchmark(_run_bench)

    def test_tab_pinned_benchmark(self, benchmark, widget, fake_web_tab):
        if False:
            return 10
        'Benchmark for _tab_pinned.'
        widget.addTab(fake_web_tab(), 'foobar')
        tab_bar = widget.tabBar()
        benchmark(functools.partial(tab_bar._tab_pinned, 0))