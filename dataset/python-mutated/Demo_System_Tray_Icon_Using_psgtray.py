import PySimpleGUI as sg
from psgtray import SystemTray
"\n    A System Tray Icon courtesy of pystray and your friends at PySimpleGUI\n    \n    Import the SystemTray object with this line of code:\n    from psgtray import SystemTray\n\n    Key for the system tray icon is: \n        tray = SystemTray()\n        tray.key\n        \n    values[key] contains the menu item chosen.\n    \n    One trick employed here is to change the window's event to be the event from the System Tray.\n    \n    \n    Copyright PySimpleGUI 2021\n"

def main():
    if False:
        print('Hello World!')
    menu = ['', ['Show Window', 'Hide Window', '---', '!Disabled Item', 'Change Icon', ['Happy', 'Sad', 'Plain'], 'Exit']]
    tooltip = 'Tooltip'
    layout = [[sg.Text('My PySimpleGUI Celebration Window - X will minimize to tray')], [sg.T('Double clip icon to restore or right click and choose Show Window')], [sg.T('Icon Tooltip:'), sg.Input(tooltip, key='-IN-', s=(20, 1)), sg.B('Change Tooltip')], [sg.Multiline(size=(60, 10), reroute_stdout=False, reroute_cprint=True, write_only=True, key='-OUT-')], [sg.Button('Go'), sg.B('Hide Icon'), sg.B('Show Icon'), sg.B('Hide Window'), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout, finalize=True, enable_close_attempted_event=True)
    tray = SystemTray(menu, single_click_events=False, window=window, tooltip=tooltip, icon=sg.DEFAULT_BASE64_ICON)
    tray.show_message('System Tray', 'System Tray Icon Started!')
    sg.cprint(sg.get_versions())
    while True:
        (event, values) = window.read()
        if event == tray.key:
            sg.cprint(f'System Tray Event = ', values[event], c='white on red')
            event = values[event]
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        sg.cprint(event, values)
        tray.show_message(title=event, message=values)
        if event in ('Show Window', sg.EVENT_SYSTEM_TRAY_ICON_DOUBLE_CLICKED):
            window.un_hide()
            window.bring_to_front()
        elif event in ('Hide Window', sg.WIN_CLOSE_ATTEMPTED_EVENT):
            window.hide()
            tray.show_icon()
        elif event == 'Happy':
            tray.change_icon(sg.EMOJI_BASE64_HAPPY_JOY)
        elif event == 'Sad':
            tray.change_icon(sg.EMOJI_BASE64_FRUSTRATED)
        elif event == 'Plain':
            tray.change_icon(sg.DEFAULT_BASE64_ICON)
        elif event == 'Hide Icon':
            tray.hide_icon()
        elif event == 'Show Icon':
            tray.show_icon()
        elif event == 'Change Tooltip':
            tray.set_tooltip(values['-IN-'])
    tray.close()
    window.close()
if __name__ == '__main__':
    main()