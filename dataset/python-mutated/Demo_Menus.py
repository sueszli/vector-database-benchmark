import PySimpleGUI as sg
'\n    Demo of Menu element, ButtonMenu element and right-click menus\n\n    The same basic structure is used for all menus in PySimpleGUI. \n    Each entry is a list of items to display.  If any of those items is a list, then a cancade menu is added.\n    \n    Copyright 2018, 2019, 2020, 2021, 2022 PySimpleGUI\n'

def second_window():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Text('The second form is small \nHere to show that opening a window using a window works')], [sg.OK()]]
    window = sg.Window('Second Form', layout)
    (event, values) = window.read()
    window.close()

def test_menus():
    if False:
        return 10
    sg.theme('LightGreen')
    sg.set_options(element_padding=(0, 0))
    menu_def = [['&File', ['&Open     Ctrl-O', '&Save       Ctrl-S', '&Properties', 'E&xit']], ['&Edit', ['&Paste', ['Special', 'Normal'], 'Undo', 'Options::this_is_a_menu_key']], ['&Toolbar', ['---', 'Command &1', 'Command &2', '---', 'Command &3', 'Command &4']], ['&Help', ['&About...']]]
    right_click_menu = ['Unused', ['Right', '!&Click', '&Menu', 'E&xit', 'Properties']]
    layout = [[sg.Menu(menu_def, tearoff=True, font='_ 12', key='-MENUBAR-')], [sg.Text('Right click me for a right click menu example')], [sg.Output(size=(60, 20))], [sg.ButtonMenu('ButtonMenu', right_click_menu, key='-BMENU-', text_color='red', disabled_text_color='green'), sg.Button('Plain Button')]]
    window = sg.Window('Windows-like program', layout, default_element_size=(12, 1), default_button_element_size=(12, 1), right_click_menu=right_click_menu)
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        print(event, values)
        if event == 'About...':
            window.disappear()
            sg.popup('About this program', 'Version 1.0', 'PySimpleGUI Version', sg.get_versions())
            window.reappear()
        elif event == 'Open':
            filename = sg.popup_get_file('file to open', no_window=True)
            print(filename)
        elif event == 'Properties':
            second_window()
    window.close()
test_menus()