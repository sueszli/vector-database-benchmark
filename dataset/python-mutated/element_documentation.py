from nicegui import ui
from ..documentation_tools import text_demo

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    with ui.element('div').classes('p-2 bg-blue-100'):
        ui.label('inside a colored div')

def more() -> None:
    if False:
        for i in range(10):
            print('nop')

    @text_demo('Move elements', '\n        This demo shows how to move elements between or within containers.\n    ')
    def move_elements() -> None:
        if False:
            while True:
                i = 10
        with ui.card() as a:
            ui.label('A')
            x = ui.label('X')
        with ui.card() as b:
            ui.label('B')
        ui.button('Move X to A', on_click=lambda : x.move(a))
        ui.button('Move X to B', on_click=lambda : x.move(b))
        ui.button('Move X to top', on_click=lambda : x.move(target_index=0))

    @text_demo('Default props', '\n        You can set default props for all elements of a certain class.\n        This way you can avoid repeating the same props over and over again.\n        \n        Default props only apply to elements created after the default props were set.\n        Subclasses inherit the default props of their parent class.\n    ')
    def default_props() -> None:
        if False:
            print('Hello World!')
        ui.button.default_props('rounded outline')
        ui.button('Button A')
        ui.button('Button B')
        ui.button.default_props(remove='rounded outline')

    @text_demo('Default classes', '\n        You can set default classes for all elements of a certain class.\n        This way you can avoid repeating the same classes over and over again.\n        \n        Default classes only apply to elements created after the default classes were set.\n        Subclasses inherit the default classes of their parent class.\n    ')
    def default_classes() -> None:
        if False:
            while True:
                i = 10
        ui.label.default_classes('bg-blue-100 p-2')
        ui.label('Label A')
        ui.label('Label B')
        ui.label.default_classes(remove='bg-blue-100 p-2')

    @text_demo('Default style', '\n        You can set a default style for all elements of a certain class.\n        This way you can avoid repeating the same style over and over again.\n        \n        A default style only applies to elements created after the default style was set.\n        Subclasses inherit the default style of their parent class.\n    ')
    def default_style() -> None:
        if False:
            print('Hello World!')
        ui.label.default_style('color: tomato')
        ui.label('Label A')
        ui.label('Label B')
        ui.label.default_style(remove='color: tomato')