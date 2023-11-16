import PySimpleGUI as sg
'\n    Demo - User Settings\n\n    Use the "user_settings" API calls to make a "settings window"\n\n    This demo is very basic. The user_settings functions are used directly without a lookup table\n    or some other mechanism to map between PySimpleGUI keys and user settings keys. \n    \n    Two windows are shown.  One is a super-simple "save previously entered filename"\n    The other is a larger "settings window" where multiple settings are saved/loaded\n    \n    Copyright 2020 PySimpleGUI.org\n'
SETTINGS_PATH = '.'

def make_window():
    if False:
        print('Hello World!')
    '\n    Creates a new window.  The default values for some elements are pulled directly from the\n    "User Settings" without the use of temp variables.\n\n    Some get_entry calls don\'t have a default value, such as theme, because there was an initial call\n    that would have set the default value if the setting wasn\'t present.  Could still put the default\n    value if you wanted but it would be 2 places to change if you wanted a different default value.\n\n    Use of a lookup table to map between element keys and user settings could be aded. This demo\n    is intentionally done without one to show how to use the settings APIs in the most basic,\n    straightforward way.\n\n    If your application allows changing the theme, then a make_window function is good to have\n    so that you can close and re-create a window easily.\n\n    :return: (sg.Window)  The window that was created\n    '
    sg.theme(sg.user_settings_get_entry('-theme-', 'DarkBlue2'))
    layout = [[sg.Text('Settings Window')], [sg.Input(sg.user_settings_get_entry('-input-', ''), k='-IN-')], [sg.Listbox(sg.theme_list(), default_values=[sg.user_settings_get_entry('theme')], size=(15, 10), k='-LISTBOX-')], [sg.CB('Option 1', sg.user_settings_get_entry('-option1-', True), k='-CB1-')], [sg.CB('Option 2', sg.user_settings_get_entry('-option2-', False), k='-CB2-')], [sg.T('Settings file = ' + sg.user_settings_filename())], [sg.Button('Save'), sg.Button('Exit without saving', k='Exit')]]
    return sg.Window('A Settings Window', layout)

def settings_window():
    if False:
        return 10
    '\n    Create and interact with a "settings window". You can a similar pair of functions to your\n    code to add a "settings" feature.\n    '
    window = make_window()
    current_theme = sg.theme()
    while True:
        (event, values) = window.read()
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            break
        if event == 'Save':
            sg.user_settings_set_entry('-input-', values['-IN-'])
            sg.user_settings_set_entry('-theme-', values['-LISTBOX-'][0])
            sg.user_settings_set_entry('-option1-', values['-CB1-'])
            sg.user_settings_set_entry('-option2-', values['-CB2-'])
        if values['-LISTBOX-'][0] != current_theme:
            current_theme = values['-LISTBOX-'][0]
            window.close()
            window = make_window()

def save_previous_filename_demo():
    if False:
        i = 10
        return i + 15
    '\n    Saving the previously selected filename....\n    A demo of one of the likely most popular use of user settings\n    * Use previous input as default for Input\n    * When a new filename is chosen, write the filename to user settings\n    '
    layout = [[sg.Text('Enter a filename:')], [sg.Input(sg.user_settings_get_entry('-filename-', ''), key='-IN-'), sg.FileBrowse()], [sg.B('Save'), sg.B('Exit Without Saving', key='Exit')]]
    window = sg.Window('Filename Example', layout)
    while True:
        (event, values) = window.read()
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            break
        elif event == 'Save':
            sg.user_settings_set_entry('-filename-', values['-IN-'])
    window.close()
if __name__ == '__main__':
    sg.user_settings_filename(path=SETTINGS_PATH)
    save_previous_filename_demo()
    settings_window()