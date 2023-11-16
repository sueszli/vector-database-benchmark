from rich.text import Text
from textual.widgets import RichLog

def test_make_renderable_expand_tabs():
    if False:
        for i in range(10):
            print('nop')
    text_log = RichLog()
    renderable = text_log._make_renderable('\tfoo')
    assert isinstance(renderable, Text)
    assert renderable.plain == '        foo'