""" Demonstration of how to register document JS event callbacks. """
from bokeh.events import Event
from bokeh.io import curdoc
from bokeh.models import Button, CustomJS

def py_ready(event: Event):
    if False:
        while True:
            i = 10
    print('READY!')
js_ready = CustomJS(code='\nconst html = "<div>READY!</div>"\ndocument.body.insertAdjacentHTML("beforeend", html)\n')
curdoc().on_event('document_ready', py_ready)
curdoc().js_on_event('document_ready', js_ready)

def py_connection_lost(event: Event):
    if False:
        while True:
            i = 10
    print('CONNECTION LOST!')
js_connection_lost = CustomJS(code='\nconst html = "<div>DISCONNECTED!</div>"\ndocument.body.insertAdjacentHTML("beforeend", html)\n')
curdoc().on_event('connection_lost', py_connection_lost)
curdoc().js_on_event('connection_lost', js_connection_lost)

def py_clicked(event: Event):
    if False:
        while True:
            i = 10
    print('CLICKED!')
js_clicked = CustomJS(code='\nconst html = "<div>CLICKED!</div>"\ndocument.body.insertAdjacentHTML("beforeend", html)\n')
button = Button(label='Click me')
button.on_event('button_click', py_clicked)
button.js_on_event('button_click', js_clicked)
curdoc().add_root(button)