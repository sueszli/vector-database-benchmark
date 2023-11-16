"""
Tests for texteditor.py
"""
import pytest
from spyder.plugins.variableexplorer.widgets.texteditor import TextEditor
TEXT = '01234567890123456789012345678901234567890123456789012345678901234567890123456789\ndedekdh elkd ezd ekjd lekdj elkdfjelfjk e'

@pytest.fixture
def texteditor(qtbot):
    if False:
        for i in range(10):
            print('nop')
    'Set up TextEditor.'

    def create_texteditor(text, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        editor = TextEditor(text, **kwargs)
        qtbot.addWidget(editor)
        return editor
    return create_texteditor

def test_texteditor(texteditor):
    if False:
        return 10
    'Run TextEditor dialog.'
    editor = texteditor(TEXT)
    editor.show()
    assert editor
    dlg_text = editor.get_value()
    assert TEXT == dlg_text

@pytest.mark.parametrize('title', [u'ñ', u'r'])
def test_title(texteditor, title):
    if False:
        while True:
            i = 10
    editor = texteditor(TEXT, title=title)
    editor.show()
    dlg_title = editor.windowTitle()
    assert title in dlg_title
if __name__ == '__main__':
    pytest.main()