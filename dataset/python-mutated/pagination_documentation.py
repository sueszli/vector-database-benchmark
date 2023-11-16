from nicegui import ui

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    p = ui.pagination(1, 5, direction_links=True)
    ui.label().bind_text_from(p, 'value', lambda v: f'Page {v}')