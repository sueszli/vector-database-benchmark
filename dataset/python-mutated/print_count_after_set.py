from reactpy import component, html, run, use_state

@component
def Counter():
    if False:
        i = 10
        return i + 15
    (number, set_number) = use_state(0)

    def handle_click(event):
        if False:
            print('Hello World!')
        set_number(number + 5)
        print(number)
    return html.div(html.h1(number), html.button({'on_click': handle_click}, 'Increment'))
run(Counter)