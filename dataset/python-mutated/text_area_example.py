from textual.app import App, ComposeResult
from textual.widgets import TextArea
TEXT = 'def hello(name):\n    print("hello" + name)\n\ndef goodbye(name):\n    print("goodbye" + name)\n'

class TextAreaExample(App):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield TextArea(TEXT, language='python')
app = TextAreaExample()
if __name__ == '__main__':
    app.run()