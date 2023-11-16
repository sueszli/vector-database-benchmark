import PySimpleGUI as sg
'\n    Demo Theme Color Swatches\n    \n    Sometimes when working with themes, it\'s nice ot know all of the hex values\n    for the theme.  Or, maybe you want to scroll through the list of themes and\n    look at the colors in the theme as groups of color swatches.  Whatever thr\n    reason, this ia good candidate for you.\n    \n    Thie program is interactive.  In addition to showing you the swatches, you can\n    interact with them.  \n    * If you hover with your mouse, you\'ll get a tooltip popup  that tells you the hex value.  \n    * If you left click, then the value it posted to the clipboard.\n    * If you right click a swatch, then the right clip menu will show you the hex value.\n      If you then select that menu item, it\'s copied to the clipbard.\n    \n    The code has several examples you may want to try out in your prgorams.  Everything from\n    using "Symbols" to make the swatches, so generating layouts, integrating (optionally) other\n    packages like pyperclip, moving a window based on the size of the window\n    \n    This code\'s pattern is becoming more widespread lately:\n    * Have a "create_window\' function where the layout and Window is defined\n    * Use a "main" program function where the event loop also lives\n    \n    Copyright 2020  PySimpleGUI.org\n'
try:
    import pyperclip
    pyperclip_available = True
except:
    pyperclip_available = False

def create_window():
    if False:
        while True:
            i = 10
    layout = [[sg.Text('Themes as color swatches', text_color='white', background_color='black', font='Default 25')], [sg.Text('Tooltip and right click a color to get the value', text_color='white', background_color='black', font='Default 15')], [sg.Text('Left click a color to copy to clipboard (requires pyperclip)', text_color='white', background_color='black', font='Default 15')]]
    layout = [[sg.Column(layout, element_justification='c', background_color='black')]]
    for (i, theme) in enumerate(sg.theme_list()):
        sg.theme(theme)
        colors = [sg.theme_background_color(), sg.theme_text_color(), sg.theme_input_background_color(), sg.theme_input_text_color()]
        if sg.theme_button_color() != sg.COLOR_SYSTEM_DEFAULT:
            colors.append(sg.theme_button_color()[0])
            colors.append(sg.theme_button_color()[1])
        colors = list(set(colors))
        row = [sg.T(sg.theme(), background_color='black', text_color='white', size=(20, 1), justification='r')]
        for color in colors:
            if color != sg.COLOR_SYSTEM_DEFAULT:
                row.append(sg.T(sg.SYMBOL_SQUARE, text_color=color, background_color='black', pad=(0, 0), font='DEFAUlT 20', right_click_menu=['Nothing', [color]], tooltip=color, enable_events=True, key=(i, color)))
        layout += [row]
    layout += [[sg.B('Exit')]]
    layout = [[sg.Column(layout, scrollable=True, vertical_scroll_only=True, background_color='black')]]
    return sg.Window('Theme Color Swatches', layout, background_color='black', finalize=True)

def main():
    if False:
        return 10
    sg.popup_quick_message('This is going to take a minute...', text_color='white', background_color='red', font='Default 20')
    window = create_window()
    sg.theme(sg.OFFICIAL_PYSIMPLEGUI_THEME)
    if window.size[1] > 100:
        window.size = (window.size[0], 1000)
    window.move(window.get_screen_size()[0] // 2 - window.size[0] // 2, window.get_screen_size()[1] // 2 - 500)
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if isinstance(event, tuple):
            chosen_color = event[1]
        elif event[0] == '#':
            chosen_color = event
        else:
            chosen_color = ''
        if pyperclip_available:
            pyperclip.copy(chosen_color)
            sg.popup_auto_close(f'{chosen_color}\nColor copied to clipboard', auto_close_duration=1)
        else:
            sg.popup_auto_close(f'pyperclip not installed\nPlease install pyperclip', auto_close_duration=3)
    window.close()
if __name__ == '__main__':
    main()