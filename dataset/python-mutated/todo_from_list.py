from reactpy import component, html, run

@component
def DataList(items):
    if False:
        for i in range(10):
            print('nop')
    list_item_elements = [html.li(text) for text in items]
    return html.ul(list_item_elements)

@component
def TodoList():
    if False:
        for i in range(10):
            print('nop')
    tasks = ['Make breakfast (important)', 'Feed the dog (important)', 'Do laundry', 'Go on a run (important)', 'Clean the house', 'Go to the grocery store', 'Do some coding', 'Read a book (important)']
    return html.section(html.h1('My Todo List'), DataList(tasks))
run(TodoList)