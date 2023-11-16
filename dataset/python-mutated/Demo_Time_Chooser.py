import PySimpleGUI as sg
'\n    Demo Time Chooser\n\n    A sample window for choosing a time.\n\n    This particular implementation uses a Spin element.  Numerous possibilities exist for entering a time of day.  Instead\n    of Spin elements, Input or Combo Elements could be used.\n\n    If you do not want your user to be able to manually enter values using the keyboard, then set readonly=True in\n    the Spin elements.\n\n    Copyright 2023 PySimpleGUI\n'

def popup_get_time(title='Time Entry', starting_hour=1, starting_minute=0, allow_manual_input=True, font=None):
    if False:
        while True:
            i = 10
    '\n    Shows a window that will gather a time of day.\n\n    :param title:               The title that is shown on the window\n    :type title:                str\n    :param starting_hour:       Value to initially show in the hour field\n    :type starting_hour:        int\n    :param starting_minute:     Value to initially show in the minute field\n    :type starting_minute:      int\n    :param allow_manual_input:  If True, then the Spin elements can be manually edited\n    :type allow_manual_input:   bool\n    :param font:                Font to use for the window\n    :type font:                 str | tuple\n    :return:                    Tuple with format: (hour, minute, am-pm string)\n    :type:                      (int, int, str)\n    '
    max_value_dict = {'-HOUR-': (1, 12), '-MIN-': (0, 59)}
    hour_list = [i for i in range(0, 15)]
    minute_list = [i for i in range(-1, 62)]
    layout = [[sg.Spin(hour_list, initial_value=starting_hour, key='-HOUR-', s=3, enable_events=True, readonly=not allow_manual_input), sg.Text(':'), sg.Spin(minute_list, initial_value=starting_minute, key='-MIN-', s=3, enable_events=True, readonly=not allow_manual_input), sg.Combo(['AM', 'PM'], 'AM', readonly=True, key='-AMPM-')], [sg.Button('Ok'), sg.Button('Cancel')]]
    window = sg.Window(title, layout, font=font)
    while True:
        (event, values) = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            hours = minutes = ampm = None
            break
        if event == '-HOUR-' or event == '-MIN-':
            spin_value = values[event]
            if spin_value > max_value_dict[event][1]:
                values[event] = max_value_dict[event][0]
                window[event].update(values[event])
            elif spin_value < max_value_dict[event][0]:
                values[event] = max_value_dict[event][1]
                window[event].update(values[event])
        if event == 'Ok':
            try:
                hours = int(values['-HOUR-'])
                minutes = int(values['-MIN-'])
                ampm = values['-AMPM-']
            except:
                continue
            if 1 <= hours <= 12 and 0 <= minutes < 60:
                break
    window.close()
    return (hours, minutes, ampm)
print(popup_get_time(font='_ 15'))