from nicegui import ui

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    label = ui.label('Change my color!')
    ui.color_input(label='Color', value='#000000', on_change=lambda e: label.style(f'color:{e.value}'))