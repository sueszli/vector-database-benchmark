from reactpy import component, html, run, use_state

def increment(old_number):
    if False:
        return 10
    new_number = old_number + 1
    return new_number

@component
def Counter():
    if False:
        for i in range(10):
            print('nop')
    (number, set_number) = use_state(0)

    def handle_click(event):
        if False:
            return 10
        set_number(increment)
        set_number(increment)
        set_number(increment)
    return html.div(html.h1(number), html.button({'on_click': handle_click}, 'Increment'))
run(Counter)