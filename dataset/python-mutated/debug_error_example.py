from reactpy import component, html, run

@component
def App():
    if False:
        while True:
            i = 10
    return html.div(GoodComponent(), BadComponent())

@component
def GoodComponent():
    if False:
        i = 10
        return i + 15
    return html.p('This component rendered successfully')

@component
def BadComponent():
    if False:
        print('Hello World!')
    msg = 'This component raised an error'
    raise RuntimeError(msg)
run(App)