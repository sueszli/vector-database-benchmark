import PySimpleGUI as sg
'\n    Demo of using User Settings to create and store your own personal PySimpleGUI themes\n\n    There are 2 operations\n    1. Initialize your settings file.  You would normally only do this once and perhaps\n        write a simple program to administer it or you can also edit the JSON file directly\n    2. Use your settings file.  Add the code to the top of all of your applications that you want\n        to have access to your selection of custom themes\n\n    Copyright 2022 PySimpleGUI.org\n'
MY_APPS_SETTING_FILENAME = 'my_awesome_apps.json'

def init_your_settings():
    if False:
        while True:
            i = 10
    DarkGrey20 = {'BACKGROUND': '#19232D', 'TEXT': '#ffffff', 'INPUT': '#32414B', 'TEXT_INPUT': '#ffffff', 'SCROLL': '#505F69', 'BUTTON': ('#ffffff', '#32414B'), 'PROGRESS': ('#505F69', '#32414B'), 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}
    DarkGrey21 = {'BACKGROUND': '#121212', 'TEXT': '#dddddd', 'INPUT': '#1e1e1e', 'TEXT_INPUT': '#dbdcd9', 'SCROLL': '#272727', 'BUTTON': ('#69b1ef', '#2e2e2e'), 'PROGRESS': ('#69b1ef', '#2e2e2e'), 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}
    sg.user_settings_delete_filename(MY_APPS_SETTING_FILENAME)
    sg.user_settings_filename(MY_APPS_SETTING_FILENAME)
    sg.user_settings_set_entry('Dark Gray 20', DarkGrey20)
    sg.user_settings_set_entry('Dark Gray 21', DarkGrey21)
    sg.user_settings_set_entry('-theme list-', ('Dark Gray 20', 'Dark Gray 21'))
    sg.popup_quick_message('Your settings file has been created and is ready to be used', background_color='#1c1e23', text_color='white', keep_on_top=True, font='_ 30', non_blocking=False)

def use_your_settings():
    if False:
        while True:
            i = 10
    sg.user_settings_filename(MY_APPS_SETTING_FILENAME)
    default_theme_name = sg.user_settings_get_entry('-theme default-', None)
    if default_theme_name is None:
        default_theme_list = sg.user_settings_get_entry('-theme list-', None)
        (event, values) = sg.Window('Choose a theme', [[sg.T('Your settings do not have a default theme chosen so please choose one')], [sg.Combo(default_theme_list, key='-THEME-', readonly=True, enable_events=True)]]).read(close=True)
        if event == sg.WIN_CLOSED:
            sg.popup_error('No theme chosen so exiting')
            exit()
        default_theme_name = values[event]
        sg.user_settings_set_entry('-theme default-', default_theme_name)
    my_theme = sg.user_settings_get_entry(default_theme_name, None)
    sg.theme_add_new(default_theme_name, my_theme)
    sg.theme(default_theme_name)
    sg.popup_get_text(f'My theme is {default_theme_name} looks like this.', image=sg.EMOJI_BASE64_HAPPY_THUMBS_UP)
if __name__ == '__main__':
    operations = ('Initialize your settings (must do first)', 'Use your settings')
    (event, values) = sg.Window('Choose an operation', [[sg.T('Choose an operation to perform')], [sg.Combo(operations, key='-OPERATION-', readonly=True, enable_events=True)]]).read(close=True)
    if event == sg.WIN_CLOSED:
        sg.popup_error('No operation chosen so exiting')
    elif values[event] == operations[0]:
        init_your_settings()
    elif values[event] == operations[1]:
        use_your_settings()