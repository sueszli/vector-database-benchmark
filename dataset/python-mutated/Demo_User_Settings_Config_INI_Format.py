import PySimpleGUI as sg
'\n    Demo - User Settings - Config.ini format\n    \n    There are now 2 types of settings files available through the UserSettings APIs\n        1. JSON - .json files\n        2. INI - Config.ini files\n\n    The default is JSON files.\n    \n    If you wish to use .ini files, then you can do so using the UserSettings object.  The function interface\n    for the UserSettings API does not support .ini files, only the object interface at this time.  You\'ll see\n    why by looking at this demo.\n    \n    JSON settings:\n        settings[\'key\']\n        \n    CONFIG.INI settings:\n        settings[\'section\'][\'key\']\n    \n    NOTE - There is a setting (default is ON) that converts True", "False, "None" into Python values of True, False, None\n     \n    Copyright 2021 PySimpleGUI\n'

def show_settings_file(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Display the contents of any .INI file you wish to display\n    :param filename: full path and filename\n    '
    settings_obj = sg.UserSettings(filename, use_config_file=True)
    sg.popup_scrolled(settings_obj, title=f'INI File: {filename}')

def save_previous_filename_demo():
    if False:
        i = 10
        return i + 15
    '\n    Saving the previously selected filename....\n    A demo of one of the likely most popular use of user settings\n    * Use previous input as default for Input\n    * When a new filename is chosen, write the filename to user settings\n    '
    layout = [[sg.Text('The filename value below will be auto-filled with previously saved entry')], [sg.T('The format for this entry is:')], [sg.T('settings["My Section"]["filename"]', background_color=sg.theme_text_color(), text_color=sg.theme_background_color())], [sg.Input(settings['My Section'].get('filename', ''), key='-IN-'), sg.FileBrowse()], [sg.B('Save')], [sg.B('Display Settings'), sg.B('Display Section'), sg.B('Display filename setting')], [sg.B('Dump an INI File')], [sg.B('Exit Without Saving', key='Exit')]]
    window = sg.Window('Filename Example', layout)
    while True:
        (event, values) = window.read()
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            break
        elif event == 'Save':
            settings['My Section']['filename'] = values['-IN-']
        elif event == 'Display Settings':
            sg.popup_scrolled(settings, title='All settings')
        elif event == 'Display Section':
            sect = settings['My Section']
            sg.popup_scrolled(sect, title='Section Contents')
        elif event == 'Display filename setting':
            sg.popup_scrolled(f"filename = {settings['My Section']['filename']}", title='Filename Setting')
        elif event.startswith('Dump'):
            filename = sg.popup_get_file('What INI file would you like to display?', file_types=(('INI Files', '*.ini'),))
            if filename:
                show_settings_file(filename)
    window.close()
if __name__ == '__main__':
    sg.theme('dark green 7')
    SETTINGS_PATH = '.'
    settings = sg.UserSettings(path=SETTINGS_PATH, use_config_file=True, convert_bools_and_none=True)
    settings['Section 2'].set('var1', 'Default')
    settings['Section 2']['var'] = 'New Value'
    settings['NEW SECTION']['a'] = 'brand new section'
    save_previous_filename_demo()