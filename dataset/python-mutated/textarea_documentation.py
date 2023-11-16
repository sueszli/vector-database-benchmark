from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    ui.textarea(label='Text', placeholder='start typing', on_change=lambda e: result.set_text('you typed: ' + e.value))
    result = ui.label()

def more() -> None:
    if False:
        return 10

    @text_demo('Clearable', '\n        The `clearable` prop from [Quasar](https://quasar.dev/) adds a button to the input that clears the text.    \n    ')
    def clearable():
        if False:
            while True:
                i = 10
        i = ui.textarea(value='some text').props('clearable')
        ui.label().bind_text_from(i, 'value')