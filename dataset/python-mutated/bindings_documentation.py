from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Bindings\n\n    NiceGUI is able to directly bind UI elements to models.\n    Binding is possible for UI element properties like text, value or visibility and for model properties that are (nested) class attributes.\n    Each element provides methods like `bind_value` and `bind_visibility` to create a two-way binding with the corresponding property.\n    To define a one-way binding use the `_from` and `_to` variants of these methods.\n    Just pass a property of the model as parameter to these methods to create the binding.\n    '

    class Demo:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.number = 1
    demo = Demo()
    v = ui.checkbox('visible', value=True)
    with ui.column().bind_visibility_from(v, 'value'):
        ui.slider(min=1, max=3).bind_value(demo, 'number')
        ui.toggle({1: 'A', 2: 'B', 3: 'C'}).bind_value(demo, 'number')
        ui.number().bind_value(demo, 'number')
date = '2023-01-01'

def more() -> None:
    if False:
        i = 10
        return i + 15

    @text_demo('Bind to dictionary', '\n        Here we are binding the text of labels to a dictionary.\n    ')
    def bind_dictionary():
        if False:
            i = 10
            return i + 15
        data = {'name': 'Bob', 'age': 17}
        ui.label().bind_text_from(data, 'name', backward=lambda n: f'Name: {n}')
        ui.label().bind_text_from(data, 'age', backward=lambda a: f'Age: {a}')
        ui.button('Turn 18', on_click=lambda : data.update(age=18))

    @text_demo('Bind to variable', '\n        Here we are binding the value from the datepicker to a bare variable.\n        Therefore we use the dictionary `globals()` which contains all global variables.\n        This demo is based on the [official datepicker example](/documentation/date#input_element_with_date_picker).\n    ')
    def bind_variable():
        if False:
            for i in range(10):
                print('nop')
        with ui.input('Date').bind_value(globals(), 'date') as date_input:
            with ui.menu() as menu:
                ui.date(on_change=lambda : ui.notify(f'Date: {date}')).bind_value(date_input)
            with date_input.add_slot('append'):
                ui.icon('edit_calendar').on('click', menu.open).classes('cursor-pointer')

    @text_demo('Bind to storage', '\n        Bindings also work with [`app.storage`](/documentation/storage).\n        Here we are storing the value of a textarea between visits.\n        The note is also shared between all tabs of the same user.\n    ')
    def ui_state():
        if False:
            i = 10
            return i + 15
        from nicegui import app
        ui.textarea('This note is kept between visits').classes('w-full').bind_value(app.storage.user, 'note')