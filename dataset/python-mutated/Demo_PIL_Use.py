import PySimpleGUI as sg
import PIL
from PIL import Image
import io
import base64
import random
'\n    Using PIL with PySimpleGUI - for Images and Buttons\n    \n    The reason for this demo is to give you this nice PIL based function - convert_to_bytes\n    \n    This function is your gateway to using any format of image (not just PNG & GIF) and to \n    resize / convert it so that it can be used with the Button and Image elements.\n    \n    Copyright 2020, 2022 PySimpleGUI.org\n'

def make_square(im, fill_color=(0, 0, 0, 0)):
    if False:
        for i in range(10):
            print('nop')
    (x, y) = im.size
    size = max(x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def convert_to_bytes(source, size=(None, None), subsample=None, zoom=None, fill=False):
    if False:
        return 10
    '\n    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.\n    Turns into  PNG format in the process so that can be displayed by tkinter\n    :param source: either a string filename or a bytes base64 image object\n    :type source:  (Union[str, bytes])\n    :param size:  optional new size (width, height)\n    :type size: (Tuple[int, int] or None)\n    :param subsample: change the size by multiplying width and height by 1/subsample\n    :type subsample: (int)\n    :param zoom: change the size by multiplying width and height by zoom\n    :type zoom: (int)\n    :param fill: If True then the image is filled/padded so that the image is square\n    :type fill: (bool)\n    :return: (bytes) a byte-string object\n    :rtype: (bytes)\n    '
    if isinstance(source, str):
        image = Image.open(source)
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(base64.b64decode(source)))
    else:
        image = PIL.Image.open(io.BytesIO(source))
    (width, height) = image.size
    scale = None
    if size != (None, None):
        (new_width, new_height) = size
        scale = min(new_height / height, new_width / width)
    elif subsample is not None:
        scale = 1 / subsample
    elif zoom is not None:
        scale = zoom
    resized_image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS) if scale is not None else image
    if fill and scale is not None:
        resized_image = make_square(resized_image)
    with io.BytesIO() as bio:
        resized_image.save(bio, format='PNG')
        contents = bio.getvalue()
        encoded = base64.b64encode(contents)
    return encoded

def random_image():
    if False:
        return 10
    return random.choice(sg.EMOJI_BASE64_LIST)

def make_toolbar():
    if False:
        return 10
    layout = [[sg.T('❎', enable_events=True, key='Exit')]]
    for i in range(6):
        layout += [[sg.B(image_data=convert_to_bytes(random_image(), (30, 30))), sg.B(image_data=convert_to_bytes(random_image(), (30, 30)))]]
    return sg.Window('', layout, element_padding=(0, 0), margins=(0, 0), finalize=True, no_titlebar=True, grab_anywhere=True)

def main():
    if False:
        for i in range(10):
            print('nop')
    image = random_image()
    size = (60, 60)
    image = convert_to_bytes(image, size, fill=False)
    layout = [[sg.Button('+', size=(4, 2)), sg.Button('-', size=(4, 2)), sg.B('Next', size=(4, 2)), sg.T(size, size=(10, 1), k='-SIZE-')], [sg.Image(data=image, k='-IMAGE-')], [sg.Button(image_data=image, key='-BUTTON IMAGE-')]]
    window = sg.Window('Window Title', layout, finalize=True)
    toolbar = make_toolbar()
    while True:
        (event_window, event, values) = sg.read_all_windows()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '+':
            size = (size[0] + 20, size[1] + 20)
        elif event == '-':
            if size[0] > 20:
                size = (size[0] - 20, size[1] - 20)
        elif event in ('Next', '-BUTTON IMAGE-'):
            image = random.choice(sg.EMOJI_BASE64_LIST)
        elif event_window == toolbar:
            image = event_window[event].ImageData
        image = convert_to_bytes(image, size, fill=True)
        window['-IMAGE-'].update(data=image)
        window['-BUTTON IMAGE-'].update(image_data=image)
        window['-SIZE-'].update(size)
    window.close()
if __name__ == '__main__':
    main()