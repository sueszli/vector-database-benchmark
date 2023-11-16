from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        while True:
            i = 10
    with ui.card().tight():
        ui.image('https://picsum.photos/id/684/640/360')
        with ui.card_section():
            ui.label('Lorem ipsum dolor sit amet, consectetur adipiscing elit, ...')

def more() -> None:
    if False:
        i = 10
        return i + 15

    @text_demo('Card without shadow', '\n        You can remove the shadow from a card by adding the `no-shadow` class.\n        The following demo shows a 1 pixel wide border instead.\n    ')
    def card_without_shadow() -> None:
        if False:
            i = 10
            return i + 15
        with ui.card().classes('no-shadow border-[1px]'):
            ui.label('See, no shadow!')

    @text_demo('The issue with nested borders', "\n        The following example shows a table nested in a card.\n        Cards have a default padding in NiceGUI, so the table is not flush with the card's border.\n        The table has the `flat` and `bordered` props set, so it should have a border.\n        However, due to the way QCard is designed, the border is not visible (first card).\n        There are two ways to fix this:\n\n        - To get the original QCard behavior, use the `tight` method (second card).\n            It removes the padding and the table border collapses with the card border.\n        \n        - To preserve the padding _and_ the table border, move the table into another container like a `ui.row` (third card).\n\n        See https://github.com/zauberzeug/nicegui/issues/726 for more information.\n    ")
    def custom_context_menu() -> None:
        if False:
            return 10
        columns = [{'name': 'age', 'label': 'Age', 'field': 'age'}]
        rows = [{'age': '16'}, {'age': '18'}, {'age': '21'}]
        with ui.row():
            with ui.card():
                ui.table(columns, rows).props('flat bordered')
            with ui.card().tight():
                ui.table(columns, rows).props('flat bordered')
            with ui.card():
                with ui.row():
                    ui.table(columns, rows).props('flat bordered')