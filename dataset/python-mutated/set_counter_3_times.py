from reactpy import component, html, run, use_state

@component
def Counter():
    if False:
        print('Hello World!')
    (number, set_number) = use_state(0)

    def handle_click(event):
        if False:
            while True:
                i = 10
        set_number(number + 1)
        set_number(number + 1)
        set_number(number + 1)
    return html.div(html.h1(number), html.button({'on_click': handle_click}, 'Increment'))
run(Counter)