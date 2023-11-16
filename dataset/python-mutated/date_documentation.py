from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    ui.date(value='2023-01-01', on_change=lambda e: result.set_text(e.value))
    result = ui.label()

def more() -> None:
    if False:
        return 10

    @text_demo('Input element with date picker', "\n        This demo shows how to implement a date picker with an input element.\n        We place an icon in the input element's append slot.\n        When the icon is clicked, we open a menu with a date picker.\n\n        The date is bound to the input element's value.\n        So both the input element and the date picker will stay in sync whenever the date is changed.\n    ")
    def date():
        if False:
            i = 10
            return i + 15
        with ui.input('Date') as date:
            with date.add_slot('append'):
                ui.icon('edit_calendar').on('click', lambda : menu.open()).classes('cursor-pointer')
            with ui.menu() as menu:
                ui.date().bind_value(date)

    @text_demo('Date filter', '\n        This demo shows how to filter the dates in a date picker.\n        In order to pass a function to the date picker, we use the `:options` property.\n        The leading `:` tells NiceGUI that the value is a JavaScript expression.\n    ')
    def date_filter():
        if False:
            while True:
                i = 10
        ui.date().props('default-year-month=2023/01 :options="date => date <= \'2023/01/15\'"')