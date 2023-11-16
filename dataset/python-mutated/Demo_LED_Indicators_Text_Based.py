import PySimpleGUI as sg
from random import randint as randint
'\n    Demo - LEDS using Text\n\n    A simple example of how you can use UNICODE characters as LED indicators in a window\n\n    Copyright 2020 PySimpleGUI.org\n'
sg.theme('Light Brown 4')
CIRCLE = '⚫'
CIRCLE_OUTLINE = '⚪'

def LED(color, key):
    if False:
        i = 10
        return i + 15
    '\n    A "user defined element".  In this case our LED is based on a Text element. This gives up 1 location to change how they look, size, etc.\n    :param color: (str) The color of the LED\n    :param key: (Any) The key used to look up the element\n    :return: (sg.Text) Returns a Text element that displays the circle\n    '
    return sg.Text(CIRCLE_OUTLINE, text_color=color, key=key)
layout = [[sg.Text('Status 1  '), LED('Green', '-LED0-')], [sg.Text('Status 2  '), LED('blue', '-LED1-')], [sg.Text('Status 3  '), LED('red', '-LED2-')], [sg.Button('Exit')]]
window = sg.Window('Window Title', layout, font='Any 16')
while True:
    (event, values) = window.read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    for i in range(3):
        window[f'-LED{i}-'].update(CIRCLE if randint(1, 100) < 25 else CIRCLE_OUTLINE)
window.close()