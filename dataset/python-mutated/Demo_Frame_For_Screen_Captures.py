import PySimpleGUI as sg
'\n    Demo Frame For Screen Captures\n\n    This program can be used to help you record videos.\n\n    Because it relies on the "transparent color" feature that\'s only available on Windows, this Demo is only going\n    to work the indended way on Windows.\n\n    Some video recorders that record a portion of the screen do not show you, at all times, what portion of the screen\n    is being recorded.  This can make it difficult for you to stay within the bounds being recorded.\n    This demo program is meant to help the situation by showing a thin line that is 20 pixels larger than the area\n    being recorded.  \n\n    The top edge of the window has the controls.  There\'s an exit button, a solid "bar" for you to grab with your mouse to move\n    the frame around your window, and 2 inputs with a "resize" button that enables you to set the frame to the size you want to stay\n    within.\n\n\n    Copyright 2022 PySimpleGUI.org\n'

def main():
    if False:
        print('Hello World!')
    offset = (20, 20)
    default_size = (1920, 1080)
    location = (None, None)
    window = sg.Window('Window Title', [[sg.Button('Exit'), sg.T(sg.SYMBOL_SQUARE * 10, grab=True), sg.I(default_size[0], s=4, k='-W-'), sg.I(default_size[1], s=4, k='-H-'), sg.B('Resize')], [sg.Frame('', [[]], s=(default_size[0] + offset[0], default_size[1] + offset[1]), k='-FRAME-')]], transparent_color=sg.theme_background_color(), right_click_menu=['', ['Edit Me', 'Exit']], location=location, no_titlebar=True, keep_on_top=True)
    while True:
        (event, values) = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Edit Me':
            sg.execute_editor(__file__)
        elif event == 'Resize':
            window['-FRAME-'].set_size((int(values['-W-']) + offset[0], int(values['-H-']) + offset[1]))
    window.close()
if __name__ == '__main__':
    main()