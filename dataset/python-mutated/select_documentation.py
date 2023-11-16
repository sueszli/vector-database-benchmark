from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    select1 = ui.select([1, 2, 3], value=1)
    select2 = ui.select({1: 'One', 2: 'Two', 3: 'Three'}).bind_value(select1, 'value')

def more() -> None:
    if False:
        return 10

    @text_demo('Search-as-you-type', '\n        You can activate `with_input` to get a text input with autocompletion.\n        The options will be filtered as you type.\n    ')
    def search_as_you_type():
        if False:
            print('Hello World!')
        continents = ['Asia', 'Africa', 'Antarctica', 'Europe', 'Oceania', 'North America', 'South America']
        ui.select(options=continents, with_input=True, on_change=lambda e: ui.notify(e.value)).classes('w-40')

    @text_demo('Multi selection', '\n        You can activate `multiple` to allow the selection of more than one item.\n    ')
    def multi_select():
        if False:
            while True:
                i = 10
        names = ['Alice', 'Bob', 'Carol']
        ui.select(names, multiple=True, value=names[:2], label='comma-separated').classes('w-64')
        ui.select(names, multiple=True, value=names[:2], label='with chips').classes('w-64').props('use-chips')

    @text_demo('Update options', '\n        Options can be changed with the `options` property.\n        But then you also need to call `update()` afterwards to let the change take effect.\n        `set_options` is a shortcut that does both and works well for lambdas.\n    ')
    def update_selection():
        if False:
            i = 10
            return i + 15
        select = ui.select([1, 2, 3], value=1)
        with ui.row():
            ui.button('4, 5, 6', on_click=lambda : select.set_options([4, 5, 6], value=4))
            ui.button('1, 2, 3', on_click=lambda : select.set_options([1, 2, 3], value=1))