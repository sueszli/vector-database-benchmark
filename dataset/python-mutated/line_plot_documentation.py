from nicegui import events, ui

def main_demo() -> None:
    if False:
        for i in range(10):
            print('nop')
    import math
    from datetime import datetime
    line_plot = ui.line_plot(n=2, limit=20, figsize=(3, 2), update_every=5).with_legend(['sin', 'cos'], loc='upper center', ncol=2)

    def update_line_plot() -> None:
        if False:
            for i in range(10):
                print('nop')
        now = datetime.now()
        x = now.timestamp()
        y1 = math.sin(x)
        y2 = math.cos(x)
        line_plot.push([now], [[y1], [y2]])
    line_updates = ui.timer(0.1, update_line_plot, active=False)
    line_checkbox = ui.checkbox('active').bind_value(line_updates, 'active')

    def handle_change(e: events.GenericEventArguments) -> None:
        if False:
            return 10

        def turn_off() -> None:
            if False:
                while True:
                    i = 10
            line_checkbox.set_value(False)
            ui.notify('Turning off that line plot to save resources on our live demo server. 😎')
        line_checkbox.value = e.args
        if line_checkbox.value:
            ui.timer(10.0, turn_off, once=True)
    line_checkbox.on('update:model-value', handle_change, args=[None])