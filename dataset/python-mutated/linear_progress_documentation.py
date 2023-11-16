from nicegui import ui

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    slider = ui.slider(min=0, max=1, step=0.01, value=0.5)
    ui.linear_progress().bind_value_from(slider, 'value')