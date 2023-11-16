import PySimpleGUI as sg
'\nA chatbot with history\nScroll up and down through prior commands using the arrow keys\nSpecial keyboard keys:\n    Up arrow - scroll up in commands\n    Down arrow - scroll down in commands\n    Escape - clear current command\n    Control C - exit form\n'

def ChatBotWithHistory():
    if False:
        while True:
            i = 10
    sg.theme('GreenTan')
    layout = [[sg.Text('Your output will go here', size=(40, 1))], [sg.Output(size=(127, 30), font='Helvetica 10')], [sg.Text('Command History'), sg.Text('', size=(20, 3), key='history')], [sg.ML(size=(85, 5), enter_submits=True, key='query', do_not_clear=False), sg.Button('SEND', button_color=(sg.YELLOWS[0], sg.BLUES[0]), bind_return_key=True), sg.Button('EXIT', button_color=(sg.YELLOWS[0], sg.GREENS[0]))]]
    window = sg.Window('Chat window with history', layout, default_element_size=(30, 2), font=('Helvetica', ' 13'), default_button_element_size=(8, 2), return_keyboard_events=True)
    command_history = []
    history_offset = 0
    while True:
        (event, value) = window.read()
        if event == 'SEND':
            query = value['query'].rstrip()
            print('The command you entered was {}'.format(query))
            command_history.append(query)
            history_offset = len(command_history) - 1
            window['query'].update('')
            window['history'].update('\n'.join(command_history[-3:]))
        elif event in (sg.WIN_CLOSED, 'EXIT'):
            break
        elif 'Up' in event and len(command_history):
            command = command_history[history_offset]
            history_offset -= 1 * (history_offset > 0)
            window['query'].update(command)
        elif 'Down' in event and len(command_history):
            history_offset += 1 * (history_offset < len(command_history) - 1)
            command = command_history[history_offset]
            window['query'].update(command)
        elif 'Escape' in event:
            window['query'].update('')
ChatBotWithHistory()