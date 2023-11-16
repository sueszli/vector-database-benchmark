import PySimpleGUI as sg

def MediaPlayerGUI():
    if False:
        return 10
    background = '#F0F0F0'
    sg.set_options(background_color=background, element_background_color=background)

    def ImageButton(title, key):
        if False:
            return 10
        return sg.Button(title, button_color=(background, background), border_width=0, key=key)
    layout = [[sg.Text('Media File Player', font=('Helvetica', 25))], [sg.Text('', size=(15, 2), font=('Helvetica', 14), key='output')], [ImageButton('restart', key='Restart Song'), sg.Text(' ' * 2), ImageButton('pause', key='Pause'), sg.Text(' ' * 2), ImageButton('next', key='Next'), sg.Text(' ' * 2), sg.Text(' ' * 2), ImageButton('exit', key='Exit')]]
    window = sg.Window('Media File Player', layout, default_element_size=(20, 1), font=('Helvetica', 25))
    while True:
        (event, values) = window.read(timeout=100)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        if event != sg.TIMEOUT_KEY:
            window['output'].update(event)
    window.close()
MediaPlayerGUI()