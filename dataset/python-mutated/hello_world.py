from reactpy import component, html, run

@component
def App():
    if False:
        i = 10
        return i + 15
    return html.h1('Hello, world!')
run(App)