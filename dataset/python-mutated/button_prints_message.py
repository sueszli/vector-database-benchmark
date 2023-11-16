from reactpy import component, html, run

@component
def PrintButton(display_text, message_text):
    if False:
        print('Hello World!')

    def handle_event(event):
        if False:
            return 10
        print(message_text)
    return html.button({'on_click': handle_event}, display_text)

@component
def App():
    if False:
        return 10
    return html.div(PrintButton('Play', 'Playing'), PrintButton('Pause', 'Paused'))
run(App)