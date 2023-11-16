"""Tests for syntaxhighlighters.py"""
import pytest
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QTextDocument
from spyder.utils.syntaxhighlighters import HtmlSH, PythonSH, MarkdownSH

def compare_formats(actualFormats, expectedFormats, sh):
    if False:
        while True:
            i = 10
    assert len(actualFormats) == len(expectedFormats)
    for (actual, expected) in zip(actualFormats, expectedFormats):
        assert actual.start == expected[0]
        assert actual.length == expected[1]
        assert actual.format.foreground().color().name() == sh.formats[expected[2]].foreground().color().name()

def test_HtmlSH_basic():
    if False:
        while True:
            i = 10
    txt = '<p style="color:red;">Foo <!--comment--> bar.</p>'
    doc = QTextDocument(txt)
    sh = HtmlSH(doc, color_scheme='Spyder')
    sh.rehighlightBlock(doc.firstBlock())
    res = [(0, 2, 'builtin'), (2, 6, 'keyword'), (8, 1, 'normal'), (9, 12, 'string'), (21, 1, 'builtin'), (22, 4, 'normal'), (26, 14, 'comment'), (40, 5, 'normal'), (45, 4, 'builtin')]
    compare_formats(doc.firstBlock().layout().additionalFormats(), res, sh)

def test_HtmlSH_unclosed_commend():
    if False:
        for i in range(10):
            print('nop')
    txt = '-->'
    doc = QTextDocument(txt)
    sh = HtmlSH(doc, color_scheme='Spyder')
    sh.rehighlightBlock(doc.firstBlock())
    res = [(0, 3, 'normal')]
    compare_formats(doc.firstBlock().layout().additionalFormats(), res, sh)

def test_PythonSH_UTF16_number():
    if False:
        i = 10
        return i + 15
    'UTF16 string'
    txt = '𨭎𨭎𨭎𨭎 = 100000000'
    doc = QTextDocument(txt)
    sh = PythonSH(doc, color_scheme='Spyder')
    sh.rehighlightBlock(doc.firstBlock())
    res = [(0, 11, 'normal'), (11, 9, 'number')]
    compare_formats(doc.firstBlock().layout().additionalFormats(), res, sh)

def test_PythonSH_UTF16_string():
    if False:
        return 10
    'UTF16 string'
    txt = '𨭎𨭎𨭎𨭎 = "𨭎𨭎𨭎𨭎"'
    doc = QTextDocument(txt)
    sh = PythonSH(doc, color_scheme='Spyder')
    sh.rehighlightBlock(doc.firstBlock())
    res = [(0, 11, 'normal'), (11, 10, 'string')]
    compare_formats(doc.firstBlock().layout().additionalFormats(), res, sh)

def test_python_string_prefix():
    if False:
        for i in range(10):
            print('nop')
    prefixes = ('r', 'u', 'R', 'U', 'f', 'F', 'fr', 'Fr', 'fR', 'FR', 'rf', 'rF', 'Rf', 'RF', 'b', 'B', 'br', 'Br', 'bR', 'BR', 'rb', 'rB', 'Rb', 'RB')
    for prefix in prefixes:
        txt = "[%s'test', %s'''test''']" % (prefix, prefix)
        doc = QTextDocument(txt)
        sh = PythonSH(doc, color_scheme='Spyder')
        sh.rehighlightBlock(doc.firstBlock())
        offset = len(prefix)
        res = [(0, 1, 'normal'), (1, 6 + offset, 'string'), (7 + offset, 2, 'normal'), (9 + offset, 10 + offset, 'string'), (19 + 2 * offset, 1, 'normal')]
        compare_formats(doc.firstBlock().layout().additionalFormats(), res, sh)

def test_Markdown_basic():
    if False:
        return 10
    txt = 'Some __random__ **text** with ~~different~~ [styles](link_url)'
    doc = QTextDocument(txt)
    sh = MarkdownSH(doc, color_scheme='Spyder')
    sh.rehighlightBlock(doc.firstBlock())
    res = [(0, 5, 'normal'), (5, 10, 'italic'), (15, 1, 'normal'), (16, 8, 'strong'), (24, 6, 'normal'), (30, 13, 'italic'), (43, 1, 'normal'), (44, 8, 'string'), (52, 1, 'normal'), (53, 8, 'string'), (61, 1, 'normal')]
    compare_formats(doc.firstBlock().layout().additionalFormats(), res, sh)

@pytest.mark.parametrize('line', ['# --- First variant', '#------ 2nd variant', '### 3rd variant'])
def test_python_outline_explorer_comment(line):
    if False:
        return 10
    assert PythonSH.OECOMMENT.match(line)

@pytest.mark.parametrize('line', ['#---', '#--------', '#---   ', '# -------'])
def test_python_not_an_outline_explorer_comment(line):
    if False:
        return 10
    assert not PythonSH.OECOMMENT.match(line)
if __name__ == '__main__':
    pytest.main()