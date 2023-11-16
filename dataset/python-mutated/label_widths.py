from textual.app import App
from textual.widgets import Label, Static
from rich.panel import Panel

class LabelWrap(App):
    CSS = 'Screen {\n              align: center middle;\n          }\n\n          #l_data {\n              border: blank;\n              background: lightgray;\n          }\n\n          #s_data {\n              border: blank;\n              background: lightgreen;\n          }\n\n          #p_data {\n              border: blank;\n              background: lightgray;\n          }'

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.data = 'Apple Banana Cherry Mango Fig Guava Pineapple:Dragon Unicorn Centaur Phoenix Chimera Castle'

    def compose(self):
        if False:
            return 10
        yield Label(self.data, id='l_data')
        yield Static(self.data, id='s_data')
        yield Label(Panel(self.data), id='p_data')

    def on_mount(self):
        if False:
            while True:
                i = 10
        self.dark = False
if __name__ == '__main__':
    app = LabelWrap()
    app.run()