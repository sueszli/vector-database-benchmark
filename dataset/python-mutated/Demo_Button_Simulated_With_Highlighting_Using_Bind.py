import PySimpleGUI as sg
'\n    Demo Program - Simulated Buttons with Mouseover Highlights\n\n    The purpose of this demo is to teach you 5 unique PySimpleGUI constructs that when combined\n    create a "Button" that highlights on mouseover regarless of the Operating System.\n    Because of how tktiner works, mouseover highlighting is inconsistent across operating systems for Buttons.\n    This is one (dare I say "clever") way to get this effect in your program\n\n    1. Binding the Enter and Leave tkinter events\n    2. Using Tuples as keys\n    3. Using List Comprehensions to build a layout\n    4. Using Text Elements to Simulate Buttons\n    5. Using a "User Defined Element" to make what appears to be a new type of Button in the layout\n\n    The KEY to making this work simply is these "Buttons" have a tuple as a key.\n        The format of the key is (\'-B-\', button_text)\n\n    An element\'s bind method will make a tuple if the original key is a tuple.\n        ((\'-B-\', button_text), \'ENTER\') will be the event when the mouse is moved over the "Button"\n\n    Copyright 2022 PySimpleGUI.org\n'

def TextButton(text):
    if False:
        return 10
    '\n    A User Defined Element.  It looks like a Button, but is a Text element\n    :param text:    The text that will be put on the "Button"\n    :return:        A Text element with a tuple as the key\n    '
    return sg.Text(text, key=('-B-', text), relief='raised', enable_events=True, font='_ 15', text_color=sg.theme_button_color_text(), background_color=sg.theme_button_color_background())

def do_binds(window, button_text):
    if False:
        i = 10
        return i + 15
    '\n    This is magic code that enables the mouseover highlighting to work.\n    '
    for btext in button_text:
        window['-B-', btext].bind('<Enter>', 'ENTER')
        window['-B-', btext].bind('<Leave>', 'EXIT')

def main():
    if False:
        for i in range(10):
            print('nop')
    button_text = ('Button 1', 'Button 2', 'Button 3')
    layout = [[TextButton(text) for text in button_text], [sg.Text(font='_ 14', k='-STATUS-')], [sg.Ok(), sg.Exit()]]
    window = sg.Window('Custom Mouseover Highlighting Buttons', layout, finalize=True)
    do_binds(window, button_text)
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if isinstance(event, tuple):
            if event[1] in ('ENTER', 'EXIT'):
                button_key = event[0]
                if event[1] == 'ENTER':
                    window[button_key].update(text_color=sg.theme_button_color_background(), background_color=sg.theme_button_color_text())
                if event[1] == 'EXIT':
                    window[button_key].update(text_color=sg.theme_button_color_text(), background_color=sg.theme_button_color_background())
            else:
                window['-STATUS-'].update(f'Button pressed = {event[1]}')
    window.close()
if __name__ == '__main__':
    main()