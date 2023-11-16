from reactpy import component, html, run

@component
def Item(name, done):
    if False:
        for i in range(10):
            print('nop')
    if done:
        return html.li(name, ' ✔')
    else:
        return html.li(name)

@component
def TodoList():
    if False:
        for i in range(10):
            print('nop')
    return html.section(html.h1('My Todo List'), html.ul(Item('Find a cool problem to solve', done=True), Item('Build an app to solve it', done=True), Item('Share that app with the world!', done=False)))
run(TodoList)