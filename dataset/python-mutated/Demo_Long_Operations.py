import time
import PySimpleGUI as sg
"\n    Demo Long Operations\n    \n    How to make calls to your functions that take a very long time to complete.\n    \n    One of the classic GUI problems is when a function takes a long time to complete.\n    Normally these functions cause a GUI to appear to the operating system to have\n    hung and you'll see a message asking if you want to kill your program.\n    \n    PySimpleGUI has a Window method - perform_long_operation that can help in these situations\n    NOTE - because this method uses threads, it's important you do not make any PySimpleGUI calls\n    from your long function.  Also, some things simply cannot be safely run as a thread.  Just understand\n    that this function perform_long_operation utilizes threads.\n    \n    window.perform_long_operation takes 2 parameters:\n        * A lambda expression that represents your function call\n        * A key that is returned when you function completes\n        \n    When you function completes, you will receive an event when calling window.read() that\n    matches the key provided.\n\n    Copyright 2021 PySimpleGUI\n"
'\nM""MMMMM""M                            \nM  MMMMM  M                            \nM  MMMMM  M .d8888b. .d8888b. 88d888b. \nM  MMMMM  M Y8ooooo. 88ooood8 88\'  `88 \nM  `MMM\'  M       88 88.  ... 88       \nMb       dM `88888P\' `88888P\' dP       \nMMMMMMMMMMM                            \n                                       \nMM""""""""`M                            \nMM  mmmmmmmM                            \nM\'      MMMM dP    dP 88d888b. .d8888b. \nMM  MMMMMMMM 88    88 88\'  `88 88\'  `"" \nMM  MMMMMMMM 88.  .88 88    88 88.  ... \nMM  MMMMMMMM `88888P\' dP    dP `88888P\' \nMMMMMMMMMMMM\n'

def my_long_func(count, a=1, b=2):
    if False:
        i = 10
        return i + 15
    '\n    This is your function that takes a long time\n    :param count:\n    :param a:\n    :param b:\n    :return:\n    '
    for i in range(count):
        print(i, a, b)
        time.sleep(0.5)
    return 'DONE!'
'\n                    oo          \n                                \n88d8b.d8b. .d8888b. dP 88d888b. \n88\'`88\'`88 88\'  `88 88 88\'  `88 \n88  88  88 88.  .88 88 88    88 \ndP  dP  dP `88888P8 dP dP    dP \n                                \n                                \noo                dP oo                              dP                        dP dP \n                  88                                 88                        88 88 \ndP 88d888b. .d888b88 dP 88d888b. .d8888b. .d8888b. d8888P    .d8888b. .d8888b. 88 88 \n88 88\'  `88 88\'  `88 88 88\'  `88 88ooood8 88\'  `""   88      88\'  `"" 88\'  `88 88 88 \n88 88    88 88.  .88 88 88       88.  ... 88.  ...   88      88.  ... 88.  .88 88 88 \ndP dP    dP `88888P8 dP dP       `88888P\' `88888P\'   dP      `88888P\' `88888P8 dP dP\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    layout = [[sg.Text('Indirect Call Version')], [sg.Text('How many times to run the loop?'), sg.Input(s=(4, 1), key='-IN-')], [sg.Text(s=(30, 1), k='-STATUS-')], [sg.Button('Go', bind_return_key=True), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout)
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Go':
            window['-STATUS-'].update('Calling your function...')
            if values['-IN-'].isnumeric():
                window.perform_long_operation(lambda : my_long_func(int(values['-IN-']), a=10), '-END KEY-')
            else:
                window['-STATUS-'].update('Try again... how about an int?')
        elif event == '-END KEY-':
            return_value = values[event]
            window['-STATUS-'].update(f'Completed. Returned: {return_value}')
    window.close()
'\n                    oo          \n                                \n88d8b.d8b. .d8888b. dP 88d888b. \n88\'`88\'`88 88\'  `88 88 88\'  `88 \n88  88  88 88.  .88 88 88    88 \ndP  dP  dP `88888P8 dP dP    dP \n                                \n                                \n      dP oo                              dP                        dP dP \n      88                                 88                        88 88 \n.d888b88 dP 88d888b. .d8888b. .d8888b. d8888P    .d8888b. .d8888b. 88 88 \n88\'  `88 88 88\'  `88 88ooood8 88\'  `""   88      88\'  `"" 88\'  `88 88 88 \n88.  .88 88 88       88.  ... 88.  ...   88      88.  ... 88.  .88 88 88 \n`88888P8 dP dP       `88888P\' `88888P\'   dP      `88888P\' `88888P8 dP dP\n'

def old_main():
    if False:
        print('Hello World!')
    layout = [[sg.Text('Direct Call Version')], [sg.Text('How many times to run the loop?'), sg.Input(s=(4, 1), key='-IN-')], [sg.Text(s=(30, 1), k='-STATUS-')], [sg.Button('Go', bind_return_key=True), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout)
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Go':
            if values['-IN-'].isnumeric():
                window['-STATUS-'].update('Calling your function...')
                window.refresh()
                return_value = my_long_func(int(values['-IN-']), a=10)
                window['-STATUS-'].update(f'Completed. Returned: {return_value}')
            else:
                window['-STATUS-'].update('Try again... how about an int?')
    window.close()
if __name__ == '__main__':
    main()