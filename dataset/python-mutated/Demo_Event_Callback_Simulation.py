import PySimpleGUI as sg
'\n    Event Callback Simulation\n    \n    This design pattern simulates callbacks for events.  \n    This is NOT the "normal" way things work in PySimpleGUI and is an architecture that is actively discouraged\n    Unlike tkinter, Qt, etc, PySimpleGUI does not utilize callback\n    functions as a mechanism for communicating when button presses or other events happen.\n    BUT, should you want to quickly convert some existing code that does use callback functions, then this\n    is one way to do a "quick and dirty" port to PySimpleGUI.\n'

def button1(event, values):
    if False:
        for i in range(10):
            print('nop')
    sg.popup_quick_message('Button 1 callback', background_color='red', text_color='white')

def button2(event, values):
    if False:
        i = 10
        return i + 15
    sg.popup_quick_message('Button 2 callback', background_color='green', text_color='white')

def catch_all(event, values):
    if False:
        print('Hello World!')
    sg.popup_quick_message(f'An unplanned event = "{event}" happend', background_color='blue', text_color='white', auto_close_duration=6)
func_dict = {'1': button1, '2': button2}
layout = [[sg.Text('Please click a button')], [sg.Button('1'), sg.Button('2'), sg.Button('Not defined', key='-MY-KEY-'), sg.Quit()]]
window = sg.Window('Button callback example', layout)
while True:
    (event, values) = window.read()
    try:
        func_dict[event](event, values)
    except:
        catch_all(event, values)
    if event in ('Quit', None):
        break
window.close()
sg.popup_auto_close('Done... this window auto closes')