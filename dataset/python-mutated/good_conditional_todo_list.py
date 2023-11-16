from reactpy import component, html, run

@component
def Item(name, done):
    if False:
        return 10
    return html.li(name, ' ✔' if done else '')

@component
def TodoList():
    if False:
        while True:
            i = 10
    return html.section(html.h1('My Todo List'), html.ul(Item('Find a cool problem to solve', done=True), Item('Build an app to solve it', done=True), Item('Share that app with the world!', done=False)))
run(TodoList)