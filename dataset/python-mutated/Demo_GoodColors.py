import sys
import PySimpleGUI as sg

def main():
    if False:
        return 10
    window = sg.Window('GoodColors', default_element_size=(30, 2))
    window.AddRow(sg.Text('Having trouble picking good colors? Try this'))
    window.AddRow(sg.Text('Here come the good colors as defined by PySimpleGUI'))
    text_color = sg.YELLOWS[0]
    buttons = (sg.Button('BLUES[{}]\n{}'.format(j, c), button_color=(text_color, c), size=(10, 2)) for (j, c) in enumerate(sg.BLUES))
    window.AddRow(sg.Text('Button Colors Using PySimpleGUI.BLUES'))
    window.AddRow(*buttons)
    window.AddRow(sg.Text('_' * 100, size=(65, 1)))
    buttons = (sg.Button('PURPLES[{}]\n{}'.format(j, c), button_color=(text_color, c), size=(10, 2)) for (j, c) in enumerate(sg.PURPLES))
    window.AddRow(sg.Text('Button Colors Using PySimpleGUI.PURPLES'))
    window.AddRow(*buttons)
    window.AddRow(sg.Text('_' * 100, size=(65, 1)))
    buttons = (sg.Button('GREENS[{}]\n{}'.format(j, c), button_color=(text_color, c), size=(10, 2)) for (j, c) in enumerate(sg.GREENS))
    window.AddRow(sg.Text('Button Colors Using PySimpleGUI.GREENS'))
    window.AddRow(*buttons)
    window.AddRow(sg.Text('_' * 100, size=(65, 1)))
    text_color = sg.GREENS[0]
    buttons = (sg.Button('TANS[{}]\n{}'.format(j, c), button_color=(text_color, c), size=(10, 2)) for (j, c) in enumerate(sg.TANS))
    window.AddRow(sg.Text('Button Colors Using PySimpleGUI.TANS'))
    window.AddRow(*buttons)
    window.AddRow(sg.Text('_' * 100, size=(65, 1)))
    text_color = 'black'
    buttons = (sg.Button('YELLOWS[{}]\n{}'.format(j, c), button_color=(text_color, c), size=(10, 2)) for (j, c) in enumerate(sg.YELLOWS))
    window.AddRow(sg.Text('Button Colors Using PySimpleGUI.YELLOWS'))
    window.AddRow(*buttons)
    window.AddRow(sg.Text('_' * 100, size=(65, 1)))
    window.AddRow(sg.Button('Click ME!'))
    (event, values) = window.read()
    window.close()
if __name__ == '__main__':
    main()