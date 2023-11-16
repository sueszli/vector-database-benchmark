from reactpy import component, html, run

@component
def Button():
    if False:
        for i in range(10):
            print('nop')
    return html.button("I don't do anything yet")
run(Button)