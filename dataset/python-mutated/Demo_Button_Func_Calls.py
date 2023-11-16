import sys
import PySimpleGUI as sg
'\nDemo Button Function Calls\nTypically GUI packages in Python (tkinter, Qt, WxPython, etc) will call a user\'s function\nwhen a button is clicked.  This "Callback" model versus "Message Passing" model is a fundamental\ndifference between PySimpleGUI and all other GUI.\n\nThere are NO BUTTON CALLBACKS in the PySimpleGUI Architecture\n\nIt is quite easy to simulate these callbacks however.  The way to do this is to add the calls\nto your Event Loop\n'

def callback_function1():
    if False:
        i = 10
        return i + 15
    sg.popup('In Callback Function 1')
    print('In the callback function 1')

def callback_function2():
    if False:
        while True:
            i = 10
    sg.popup('In Callback Function 2')
    print('In the callback function 2')
layout = [[sg.Text('Demo of Button Callbacks')], [sg.Button('Button 1'), sg.Button('Button 2')]]
window = sg.Window('Button Callback Simulation', layout)
while True:
    (event, values) = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == 'Button 1':
        callback_function1()
    elif event == 'Button 2':
        callback_function2()
window.close()