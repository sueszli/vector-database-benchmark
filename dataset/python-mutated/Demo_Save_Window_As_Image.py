import PySimpleGUI as sg
from PIL import ImageGrab
"\n    Demo - Saving the contents of a window as an image file\n    \n    This demo will teach you how to save any portion of your window to an image file.\n    You can save in JPG, GIF, or PNG format.\n    \n    In this example the entire window's layout is placed into a single Column Element.  This allows\n    us to save an image of the Column which saves the entire window layout\n    \n    Portions of windows can be saved, such as a Graph Element, by specifying the Graph Element instead of the Column\n"

def save_element_as_file(element, filename):
    if False:
        while True:
            i = 10
    '\n    Saves any element as an image file.  Element needs to have an underlyiong Widget available (almost if not all of them do)\n    :param element: The element to save\n    :param filename: The filename to save to. The extension of the filename determines the format (jpg, png, gif, ?)\n    '
    widget = element.Widget
    box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())
    grab = ImageGrab.grab(bbox=box)
    grab.save(filename)

def main():
    if False:
        while True:
            i = 10
    col = [[sg.Text('This is the first line')], [sg.In()], [sg.Button('Save'), sg.Button('Exit')]]
    layout = [[sg.Column(col, key='-COLUMN-')]]
    window = sg.Window('Drawing and Moving Stuff Around', layout)
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        elif event == 'Save':
            filename = sg.popup_get_file('Choose file (PNG, JPG, GIF) to save to', save_as=True)
            save_element_as_file(window['-COLUMN-'], filename)
    window.close()
main()