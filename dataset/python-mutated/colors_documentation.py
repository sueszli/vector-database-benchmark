from nicegui import ui

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    b1 = ui.button('Default', on_click=lambda : [b.classes(replace='!bg-primary') for b in {b1, b2}])
    b2 = ui.button('Gray', on_click=lambda : [b.classes(replace='!bg-[#555]') for b in {b1, b2}])