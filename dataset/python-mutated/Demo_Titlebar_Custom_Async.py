import PySimpleGUI as sg
'\n    Custom Titlebar - Async Version\n\n    Demo showing how to remove the titlebar and replace with your own\n    Unlike previous demos that lacked a titlebar, this one provides a way for you\n    to "minimize" your window that does not have a titlebar.  This is done by faking\n    the window using a hidden window that is minimized.\n\n    While this demo uses the button colors for the titlebar color, you can use anything you want.\n    If possible it could be good to use combinations that are known to match like the input element colors\n    or the button colors.\n\n    This version of the demo allows for async execution of your code.  In another demo of this\n    custom titlebar idea, the user window would stop when the window is minimized.  This is OK\n    for most applications, but if you\'re running a window with a timeout value (an async window)\n    then stopping execution when the window is minimized is not good.\n\n    The way to achieve both async window and the custom titlebar is to use the "read_all_windows"\n    function call.  Using this function with a timeout has the same effect as running your\n    window.read with a timeout.\n\n    Additionally, if you right click and choose "close" on the minimized window on your\n    taskbar, now the program will exist rather than restoring the window like the other demo does.\n\n    Copyright 2020 PySimpleGUI.org\n'

def minimize_main_window(main_window):
    if False:
        return 10
    '\n    Creates an icon on the taskbar that represents your custom titlebar window.\n    The FocusIn event is set so that if the user restores the window from the taskbar.\n    If this window is closed by right clicking on the icon and choosing close, then the\n    program will exit just as if the "X" was clicked on the main window.\n    '
    main_window.hide()
    layout = [[sg.T('This is your window with a customized titlebar... you just cannot see it')]]
    window = sg.Window(main_window.Title, layout, finalize=True, alpha_channel=0)
    window.minimize()
    window.bind('<FocusIn>', '-RESTORE-')
    minimize_main_window.dummy_window = window

def restore_main_window(main_window):
    if False:
        return 10
    '\n    Call this function when you want to restore your main window\n\n    :param main_window:\n    :return:\n    '
    if hasattr(minimize_main_window, 'dummy_window'):
        minimize_main_window.dummy_window.close()
        minimize_main_window.dummy_window = None
    main_window.un_hide()

def title_bar(title, text_color, background_color):
    if False:
        print('Hello World!')
    '\n    Creates a "row" that can be added to a layout. This row looks like a titlebar\n    :param title: The "title" to show in the titlebar\n    :type title: str\n    :return: A list of elements (i.e. a "row" for a layout)\n    :type: List[sg.Element]\n    '
    bc = background_color
    tc = text_color
    return [sg.Col([[sg.T(title, text_color=tc, background_color=bc)]], pad=(0, 0), background_color=bc), sg.Col([[sg.T('_', text_color=tc, background_color=bc, enable_events=True, key='-MINIMIZE-'), sg.Text('❎', text_color=tc, background_color=bc, enable_events=True, key='Exit')]], element_justification='r', key='-TITLEBAR-', pad=(0, 0), background_color=bc)]

def main():
    if False:
        print('Hello World!')
    sg.theme('light brown 10')
    title = 'Customized Titlebar Window'
    layout = [title_bar(title, sg.theme_button_color()[0], sg.theme_button_color()[1]), [sg.T('This is normal window text.   The above is the fake "titlebar"')], [sg.T('Input something:')], [sg.Input(key='-IN-'), sg.Text(size=(12, 1), key='-OUT-')], [sg.Button('Go')]]
    window_main = sg.Window(title, layout, resizable=True, no_titlebar=True, grab_anywhere=True, keep_on_top=True, margins=(0, 0), finalize=True)
    window_main['-TITLEBAR-'].expand(True, False, False)
    counter = 0
    while True:
        (window, event, values) = sg.read_all_windows(timeout=100)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-MINIMIZE-':
            minimize_main_window(window_main)
            continue
        elif event == '-RESTORE-' or (event == sg.WINDOW_CLOSED and window != window_main):
            restore_main_window(window_main)
            continue
        window_main['-OUT-'].update(counter)
        counter += 1
    window.close()
if __name__ == '__main__':
    main()