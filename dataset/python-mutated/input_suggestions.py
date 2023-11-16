from textual.app import App, ComposeResult
from textual.suggester import SuggestFromList
from textual.widgets import Input
fruits = ['apple', 'pear', 'mango', 'peach', 'strawberry', 'blueberry', 'banana']

class FruitsApp(App[None]):
    CSS = '\n    Input > .input--suggestion {\n        color: red;\n        text-style: italic;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Input('straw', suggester=SuggestFromList(fruits))
if __name__ == '__main__':
    app = FruitsApp()
    app.run()