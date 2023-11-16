from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Label

class BorderAlphaApp(App[None]):
    CSS = '   \n    .boxes {\n        height: 3;\n        width: 100%;\n    }\n    \n    #box0 {\n        border: heavy green 0%;\n    }\n    #box1 {\n        border: heavy green 20%;\n    }\n    #box2 {\n        border: heavy green 40%;\n    }\n    #box3 {\n        border: heavy green 60%;\n    }\n    #box4 {\n        border: heavy green 80%;\n    }\n    #box5 {\n        border: heavy green 100%;\n    }        \n    '

    def compose(self) -> ComposeResult:
        if False:
            return 10
        with Vertical():
            for box in range(6):
                yield Label(id=f'box{box}', classes='boxes')
if __name__ == '__main__':
    BorderAlphaApp().run()