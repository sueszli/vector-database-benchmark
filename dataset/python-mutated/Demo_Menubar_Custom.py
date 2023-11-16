import PySimpleGUI as sg
'\n    Demo - The MenubarCustom element\n    \n    With the addition of the Custom Titlebar came the addition of a problem with the \n    os provided menubar.  It\'s not possible to use the custom titlebar with the\n    normal menubar.  The menubar ends up being placed above the titlebar.\n    \n    Enter the MenubarCustom!\n    \n    This "Compound Element" is not really an element but rather a combination of\n    ButtonMenu elements and Column elements.  It\'s possible for users to create a similar\n    construct. In fact, this element started as a Demo Program and then migrated into PySimpleGUI.\n\n    At the moment, you cannot perform all of the same operations using this custom menubar\n    that you can with a traditional menubar.  Modifying the menu isn\'t possible yet.  In\n    other words, it\'s a great start, but more work is needed such as adding an update method, etc.\n    \n    For statically defined menus, it works great.  Shortuts are possible within the menus, but not\n    for the initial selection.\n    \n    You can use the same menu defition as the standard Menu element.\n\n    The Menubar will tbe themed according to your current theme\'s colors.  The theme\'s button\n    colors are used.  All of the colors can be changed from the menubar to the menu\'s text an background.\n    \n    The disabled color has a problem.  The default tkinter gray is used even though PySimpleGUI sets the value.\n    The color choice for the menubar background and text use the theme\'s button colors.\n    You can change these color choices by changing the Menubar in the layout.\n\n    Copyright 2021 PySimpleGUI\n'

def main():
    if False:
        return 10
    sg.theme('dark green 7')
    sg.theme('dark red')
    menu_def = [['&File', ['&Open     Ctrl-O', '&Save       Ctrl-S', '&Properties', 'E&xit']], ['&Edit', ['Edit Me', 'Special', 'Normal', ['Normal1', 'Normal2'], 'Undo']], ['!Disabled', ['Special', 'Normal', ['Normal1', 'Normal2'], 'Undo']], ['&Toolbar', ['---', 'Command &1::Command_Key', 'Command &2', '---', 'Command &3', 'Command &4']], ['&Help', ['&About...']]]
    layout = [[sg.MenubarCustom(menu_def, pad=(0, 0), k='-CUST MENUBAR-')], [sg.Multiline(size=(70, 20), reroute_cprint=True, write_only=True, no_scrollbar=True, k='-MLINE-')]]
    window = sg.Window('Custom Titlebar with Custom (Simulated) Menubar', layout, use_custom_titlebar=True, keep_on_top=True, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT)
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        sg.cprint(f'event = {event}', c=(sg.theme_background_color(), sg.theme_text_color()))
        sg.cprint(f'values = {values}', c=(sg.theme_input_text_color(), sg.theme_input_background_color()))
        if event == 'About...':
            window.disappear()
            sg.popup('About this program', 'Simulated Menubar to accompany a simulated Titlebar', 'PySimpleGUI Version', sg.get_versions(), grab_anywhere=True, keep_on_top=True)
            window.reappear()
        elif event == 'Edit Me':
            sg.execute_editor(__file__)
        elif event == 'Version':
            sg.popup_scrolled(__file__, sg.get_versions(), keep_on_top=True, non_blocking=True)
        elif event.startswith('Open'):
            filename = sg.popup_get_file('file to open', no_window=True)
            print(filename)
    window.close()
if __name__ == '__main__':
    main()