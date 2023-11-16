from reactpy import component, html, run

@component
def Button():
    if False:
        i = 10
        return i + 15

    def handle_event(event):
        if False:
            for i in range(10):
                print('nop')
        print(event)
    return html.button({'on_click': handle_event}, 'Click me!')
run(Button)