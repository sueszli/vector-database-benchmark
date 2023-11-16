import PySimpleGUIWeb as sg
import subprocess
import howdoi
HOW_DO_I_COMMAND = 'python -m howdoi.howdoi'

def HowDoI():
    if False:
        print('Hello World!')
    "\n    Make and show a window (PySimpleGUI form) that takes user input and sends to the HowDoI web oracle\n    Excellent example of 2 GUI concepts\n        1. Output Element that will show text in a scrolled window\n        2. Non-Window-Closing Buttons - These buttons will cause the form to return with the form's values, but doesn't close the form\n    :return: never returns\n    "
    sg.change_look_and_feel('GreenTan')
    layout = [[sg.Text('Ask and your answer will appear here....', size=(40, 1))], [sg.MLineOutput(size_px=(980, 400), key='_OUTPUT_')], [sg.CBox('Display Full Text', key='full text', font='Helvetica 15'), sg.Text('Command History', font='Helvetica 15'), sg.Text('', size=(40, 3), text_color=sg.BLUES[0], key='history')], [sg.MLine(size=(85, 5), enter_submits=True, key='query', do_not_clear=False), sg.ReadButton('SEND', button_color=(sg.YELLOWS[0], sg.BLUES[0]), bind_return_key=True), sg.Button('EXIT', button_color=(sg.YELLOWS[0], sg.GREENS[0]))]]
    window = sg.Window('How Do I?', layout, default_element_size=(30, 1), font=('Helvetica', ' 17'), default_button_element_size=(8, 2), return_keyboard_events=False)
    command_history = []
    history_offset = 0
    while True:
        (event, values) = window.Read()
        if type(event) is int:
            event = str(event)
        if event == 'SEND':
            query = values['query'].rstrip()
            window['_OUTPUT_'].update(query, append=True)
            print(query)
            QueryHowDoI(query, 1, values['full text'], window)
            command_history.append(query)
            history_offset = len(command_history) - 1
            window['query'].update('')
            window['history'].update('\n'.join(command_history[-3:]))
        elif event == None or event == 'EXIT':
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
    window.close()

def QueryHowDoI(Query, num_answers, full_text, window: sg.Window):
    if False:
        print('Hello World!')
    "\n    Kicks off a subprocess to send the 'Query' to HowDoI\n    Prints the result, which in this program will route to a gooeyGUI window\n    :param Query: text english question to ask the HowDoI web engine\n    :return: nothing\n    "
    howdoi_command = HOW_DO_I_COMMAND
    full_text_option = ' -a' if full_text else ''
    t = subprocess.Popen(howdoi_command + ' "' + Query + '" -n ' + str(num_answers) + full_text_option, stdout=subprocess.PIPE)
    (output, err) = t.communicate()
    window['_OUTPUT_'].update('{:^88}'.format(Query.rstrip()), append=True)
    window['_OUTPUT_'].update('_' * 60, append=True)
    window['_OUTPUT_'].update(output.decode('utf-8'), append=True)
    exit_code = t.wait()
if __name__ == '__main__':
    HowDoI()