import PySimpleGUI as sg
"\n    IP Address entry window with digit validation and auto advance\n    If not a digit or ., the ignored\n    . will advance the focus to the next entry\n    On the last input, once it's complete the focus moves to the OK button\n    Pressing spacebar with focus on OK generates an -OK- event\n"

def MyInput(key):
    if False:
        for i in range(10):
            print('nop')
    return sg.I('', size=(3, 1), key=key, pad=(0, 2))
layout = [[sg.T('Your typed chars appear here:'), sg.T('', key='-OUTPUT-')], [MyInput(0), sg.T('.'), MyInput(1), sg.T('.'), MyInput(2), sg.T('.'), MyInput(3)], [sg.B('Ok', key='-OK-', bind_return_key=True), sg.B('Exit')]]
window = sg.Window('Window Title', layout, return_keyboard_events=True)
while True:
    (event, values) = window.read()
    print(event)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elem = window.find_element_with_focus()
    if elem is not None:
        key = elem.Key
        value = values[key]
        if event == '.' and key != '-OK-':
            elem.update(value[:-1])
            value = value[:-1]
            next_elem = window[key + 1]
            next_elem.set_focus()
        elif event not in '0123456789':
            elem.update(value[:-1])
        elif len(value) > 2 and key < 3:
            next_elem = window[key + 1]
            next_elem.set_focus()
        elif len(value) > 2 and key == 3:
            window['-OK-'].set_focus()
            print('You entered IP Address {}.{}.{}.{}'.format(*values.values()))
window.close()