from textual.app import App, ComposeResult
from textual.widgets import TextArea
from textual.widgets.text_area import Selection
TEXT = 'def hello(name):\n    print("hello" + name)\n\ndef goodbye(name):\n    print("goodbye" + name)\n'

class TextAreaSelection(App):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        text_area = TextArea(TEXT, language='python')
        text_area.selection = Selection(start=(0, 0), end=(2, 0))
        yield text_area
app = TextAreaSelection()
if __name__ == '__main__':
    app.run()