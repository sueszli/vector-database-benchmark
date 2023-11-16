import PySimpleGUI as sg
'\n    Demo Status Bar\n    \n    This demo shows you how to create your statusbar in a way that will keep it at the bottom of\n    a resizeable window.  The key is the correct setting of the Expand settings for both the \n    StatusBar (done for you) and for a line above it that will keep it pushed to the bottom of the window.\n    It\'s possible to also "simulate" a statusbar (i.e. use a text element or something else) by also\n    configuring that element with the correct expand setting (X direction = True, expand row=True)\n    \n    Copyright 2020 PySimpleGUI.org\n'

def main():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Text('StatusBar Demo', font='ANY 15')], [sg.Text('This window has a status bar that is at the bottom of the window')], [sg.Text('The key to getting your bar to stay at the bottom of the window when')], [sg.Text('the window is resizeed is to insert a line of text (or some other element)')], [sg.Text('that is configured to expand.  ')], [sg.Text('This is accomplished by calling the "expand" method')], [sg.Text('')], [sg.Button('Ok'), sg.B('Quit')], [sg.Text(key='-EXPAND-', font='ANY 1', pad=(0, 0))], [sg.StatusBar('This is the statusbar')]]
    window = sg.Window('Vertical Layout Example', layout, resizable=True, finalize=True)
    window['-EXPAND-'].expand(True, True, True)
    while True:
        (event, values) = window.read()
        if event in (sg.WINDOW_CLOSED, 'Quit'):
            break
if __name__ == '__main__':
    main()