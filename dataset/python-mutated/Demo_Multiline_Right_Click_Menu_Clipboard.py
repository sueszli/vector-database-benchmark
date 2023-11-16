import PySimpleGUI as sg
"\n    Demo - Adding a right click menu to perform multiline element common operations\n\n    Sometimes Multiline Elements can benefit from a right click menu. There are no default menu\n    that come with tkinter, so you'll need to create your own.\n\n    Some common clipboard types of operations\n        Select all\n        Copy\n        Paste\n        Cut\n\n    The underlying Widget is accessed several times in this code because setting selections,\n    getting their values, and clipboard operations are not currently exposed in the APIs\n\n    NOTE - With tkinter, if you use the built-in clipboard, you must keep your program\n    running in order to access the clipboard.  Upon exit, your clipboard will be deleted.\n    You can get around this by using other clipboard packages.\n\n    Copyright 2021 PySimpleGUI\n"
right_click_menu = ['', ['Copy', 'Paste', 'Select All', 'Cut']]
MLINE_KEY = '-MLINE-'

def do_clipboard_operation(event, window, element):
    if False:
        print('Hello World!')
    if event == 'Select All':
        element.Widget.selection_clear()
        element.Widget.tag_add('sel', '1.0', 'end')
    elif event == 'Copy':
        try:
            text = element.Widget.selection_get()
            window.TKroot.clipboard_clear()
            window.TKroot.clipboard_append(text)
        except:
            print('Nothing selected')
    elif event == 'Paste':
        element.Widget.insert(sg.tk.INSERT, window.TKroot.clipboard_get())
    elif event == 'Cut':
        try:
            text = element.Widget.selection_get()
            window.TKroot.clipboard_clear()
            window.TKroot.clipboard_append(text)
            element.update('')
        except:
            print('Nothing selected')

def main():
    if False:
        return 10
    layout = [[sg.Text('Using a custom right click menu with Multiline Element')], [sg.Multiline(size=(60, 20), key=MLINE_KEY, right_click_menu=right_click_menu)], [sg.B('Go'), sg.B('Exit')]]
    window = sg.Window('Right Click Menu Multiline', layout)
    mline: sg.Multiline = window[MLINE_KEY]
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event in right_click_menu[1]:
            do_clipboard_operation(event, window, mline)
    window.close()
if __name__ == '__main__':
    main()