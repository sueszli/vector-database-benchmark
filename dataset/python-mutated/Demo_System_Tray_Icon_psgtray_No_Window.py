import PySimpleGUI as sg
from psgtray import SystemTray
'\n    A System Tray Icon using pystray - No visible window version\n\n    Import the SystemTray object with this line of code:\n    from psgtray import SystemTray\n\n    Key for the system tray icon is: \n        tray = SystemTray()\n        tray.key\n\n    values[key] contains the menu item chosen.\n\n    One trick employed here is to change the window\'s event to be the event from the System Tray.\n\n    This demo program keeps the Window hidden all the time so that it\'s a pure "System Tray" application.\n    Because the PySimpleGUI architecture implemented the tray icon using the psgtray package combined with the\n    overall window event loop, a Window object is still required.  The point of this demo is to show that this\n    window does not need to ever appear to the user.\n\n    Copyright PySimpleGUI 2022\n'

def main():
    if False:
        return 10
    menu = ['', ['---', '!Disabled Item', 'Change Icon', ['Happy', 'Sad', 'Plain'], 'Exit']]
    tooltip = 'Tooltip'
    layout = [[sg.T('Empty Window', key='-T-')]]
    window = sg.Window('Window Title', layout, finalize=True, enable_close_attempted_event=True, alpha_channel=0)
    window.hide()
    tray = SystemTray(menu, single_click_events=False, window=window, tooltip=tooltip, icon=sg.DEFAULT_BASE64_ICON, key='-TRAY-')
    tray.show_message('System Tray', 'System Tray Icon Started!')
    print(sg.get_versions())
    while True:
        (event, values) = window.read()
        if event == tray.key:
            event = values[event]
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        tray.show_message(title=event, message=values)
        if event == 'Happy':
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