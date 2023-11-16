import PySimpleGUI as sg
import os.path
import PIL.Image
import io
import base64
'\n    Demo for displaying any format of image file.\n    \n    Normally tkinter only wants PNG and GIF files.  This program uses PIL to convert files\n    such as jpg files into a PNG format so that tkinter can use it.\n    \n    The key to the program is the function "convert_to_bytes" which takes a filename or a \n    bytes object and converts (with optional resize) into a PNG formatted bytes object that\n    can then be passed to an Image Element\'s update method.  This function can also optionally\n    resize the image.\n    \n    Copyright 2020 PySimpleGUI.org\n'

def convert_to_bytes(file_or_bytes, resize=None):
    if False:
        while True:
            i = 10
    '\n    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.\n    Turns into  PNG format in the process so that can be displayed by tkinter\n    :param file_or_bytes: either a string filename or a bytes base64 image object\n    :type file_or_bytes:  (Union[str, bytes])\n    :param resize:  optional new size\n    :type resize: (Tuple[int, int] or None)\n    :return: (bytes) a byte-string object\n    :rtype: (bytes)\n    '
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)
    (cur_width, cur_height) = img.size
    if resize:
        (new_width, new_height) = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.LANCZOS)
    with io.BytesIO() as bio:
        img.save(bio, format='PNG')
        del img
        return bio.getvalue()
left_col = [[sg.Text('Folder'), sg.In(size=(25, 1), enable_events=True, key='-FOLDER-'), sg.FolderBrowse()], [sg.Listbox(values=[], enable_events=True, size=(40, 20), key='-FILE LIST-')], [sg.Text('Resize to'), sg.In(key='-W-', size=(5, 1)), sg.In(key='-H-', size=(5, 1))]]
images_col = [[sg.Text('You choose from the list:')], [sg.Text(size=(40, 1), key='-TOUT-')], [sg.Image(key='-IMAGE-')]]
layout = [[sg.Column(left_col, element_justification='c'), sg.VSeperator(), sg.Column(images_col, element_justification='c')]]
window = sg.Window('Multiple Format Image Viewer', layout, resizable=True)
while True:
    (event, values) = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == '-FOLDER-':
        folder = values['-FOLDER-']
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [f for f in file_list if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.jpg', 'jpeg', '.tiff', '.bmp'))]
        window['-FILE LIST-'].update(fnames)
    elif event == '-FILE LIST-':
        try:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            window['-TOUT-'].update(filename)
            if values['-W-'] and values['-H-']:
                new_size = (int(values['-W-']), int(values['-H-']))
            else:
                new_size = None
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=new_size))
        except Exception as E:
            print(f'** Error {E} **')
            pass
window.close()