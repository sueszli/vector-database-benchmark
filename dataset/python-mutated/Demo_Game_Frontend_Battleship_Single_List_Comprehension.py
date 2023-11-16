import PySimpleGUI as sg
from random import randint

def Battleship():
    if False:
        for i in range(10):
            print('nop')
    sg.theme('Dark Blue 3')
    MAX_ROWS = MAX_COL = 10
    layout = [[sg.Text('BATTLESHIP', font='Default 25')], [sg.Text(size=(15, 1), key='-MESSAGE-', font='Default 20')]]
    board = []
    for row in range(MAX_ROWS):
        board.append([sg.Button(str('O'), size=(4, 2), pad=(0, 0), border_width=0, key=(row, col)) for col in range(MAX_COL)])
    layout += board
    layout += [[sg.Button('Exit', button_color=('white', 'red'))]]
    window = sg.Window('Battleship', layout)
    while True:
        (event, values) = window.read()
        print(event, values)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if randint(1, 10) < 5:
            window[event].update('H', button_color=('white', 'red'))
            window['-MESSAGE-'].update('Hit')
        else:
            window[event].update('M', button_color=('white', 'black'))
            window['-MESSAGE-'].update('Miss')
    window.close()
Battleship()