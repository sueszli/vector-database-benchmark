from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        print('Hello World!')
    ui.tree([{'id': 'numbers', 'children': [{'id': '1'}, {'id': '2'}]}, {'id': 'letters', 'children': [{'id': 'A'}, {'id': 'B'}]}], label_key='id', on_select=lambda e: ui.notify(e.value))

def more() -> None:
    if False:
        return 10

    @text_demo('Tree with custom header and body', '\n        Scoped slots can be used to insert custom content into the header and body of a tree node.\n        See the [Quasar documentation](https://quasar.dev/vue-components/tree#customize-content) for more information.\n    ')
    def tree_with_custom_header_and_body():
        if False:
            return 10
        tree = ui.tree([{'id': 'numbers', 'description': 'Just some numbers', 'children': [{'id': '1', 'description': 'The first number'}, {'id': '2', 'description': 'The second number'}]}, {'id': 'letters', 'description': 'Some latin letters', 'children': [{'id': 'A', 'description': 'The first letter'}, {'id': 'B', 'description': 'The second letter'}]}], label_key='id', on_select=lambda e: ui.notify(e.value))
        tree.add_slot('default-header', '\n            <span :props="props">Node <strong>{{ props.node.id }}</strong></span>\n        ')
        tree.add_slot('default-body', '\n            <span :props="props">Description: "{{ props.node.description }}"</span>\n        ')

    @text_demo('Expand and collapse programmatically', '\n        The whole tree or individual nodes can be toggled programmatically using the `expand()` and `collapse()` methods.\n        This even works if a node is disabled (e.g. not clickable by the user).\n    ')
    def expand_programmatically():
        if False:
            while True:
                i = 10
        t = ui.tree([{'id': 'A', 'children': [{'id': 'A1'}, {'id': 'A2'}], 'disabled': True}, {'id': 'B', 'children': [{'id': 'B1'}, {'id': 'B2'}]}], label_key='id').expand()
        with ui.row():
            ui.button('+ all', on_click=t.expand)
            ui.button('- all', on_click=t.collapse)
            ui.button('+ A', on_click=lambda : t.expand(['A']))
            ui.button('- A', on_click=lambda : t.collapse(['A']))

    @text_demo('Tree with checkboxes', '\n        The tree can be used with checkboxes by setting the "tick-strategy" prop.\n    ')
    def tree_with_checkboxes():
        if False:
            while True:
                i = 10
        ui.tree([{'id': 'A', 'children': [{'id': 'A1'}, {'id': 'A2'}]}, {'id': 'B', 'children': [{'id': 'B1'}, {'id': 'B2'}]}], label_key='id', tick_strategy='leaf', on_tick=lambda e: ui.notify(e.value))