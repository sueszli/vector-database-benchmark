from unittest import mock
import hypothesis
import hypothesis.strategies
import pytest
from qutebrowser.qt.core import Qt
from qutebrowser.qt.gui import QTextDocument, QColor
from qutebrowser.qt.widgets import QTextEdit
from qutebrowser.completion import completiondelegate

@pytest.mark.parametrize('pat,txt,segments', [('foo', 'foo', [(0, 3)]), ('foo', 'foobar', [(0, 3)]), ('foo', 'FOObar', [(0, 3)]), ('foo', 'barfoo', [(3, 3)]), ('foo', 'barfoobaz', [(3, 3)]), ('foo', 'barfoobazfoo', [(3, 3), (9, 3)]), ('foo', 'foofoo', [(0, 3), (3, 3)]), ('a b', 'cadb', [(1, 1), (3, 1)]), ('foo', '<foo>', [(1, 3)]), ('<a>', '<a>bc', [(0, 3)]), ('foo', "'foo'", [(1, 3)]), ('x', "'x'", [(1, 1)]), ('lt', '<lt', [(1, 2)]), ('bar', '𝙛𝙤𝙤bar', [(6, 3)]), ('an anomaly', 'an anomaly', [(0, 2), (3, 7)])])
def test_highlight(pat, txt, segments):
    if False:
        for i in range(10):
            print('nop')
    doc = QTextDocument(txt)
    highlighter = completiondelegate._Highlighter(doc, pat, Qt.GlobalColor.red)
    highlighter.setFormat = mock.Mock()
    highlighter.highlightBlock(txt)
    highlighter.setFormat.assert_has_calls([mock.call(s[0], s[1], mock.ANY) for s in segments])

def test_benchmark_highlight(benchmark):
    if False:
        while True:
            i = 10
    txt = 'boofoobar'
    pat = 'foo bar'
    doc = QTextDocument(txt)

    def bench():
        if False:
            for i in range(10):
                print('nop')
        highlighter = completiondelegate._Highlighter(doc, pat, Qt.GlobalColor.red)
        highlighter.highlightBlock(txt)
    benchmark(bench)

@hypothesis.given(text=hypothesis.strategies.text())
def test_pattern_hypothesis(text):
    if False:
        i = 10
        return i + 15
    "Make sure we can't produce invalid patterns."
    doc = QTextDocument()
    completiondelegate._Highlighter(doc, text, Qt.GlobalColor.red)

def test_highlighted(qtbot):
    if False:
        i = 10
        return i + 15
    "Make sure highlighting works.\n\n    Note that with Qt > 5.12.1 we need to call setPlainText *after*\n    creating the highlighter for highlighting to work. Ideally, we'd test\n    whether CompletionItemDelegate._get_textdoc() works properly, but testing\n    that is kind of hard, so we just test it in isolation here.\n    "
    doc = QTextDocument()
    completiondelegate._Highlighter(doc, 'Hello', Qt.GlobalColor.red)
    doc.setPlainText('Hello World')
    edit = QTextEdit()
    qtbot.add_widget(edit)
    edit.setDocument(doc)
    colors = [f.foreground().color() for f in doc.allFormats()]
    assert QColor('red') in colors