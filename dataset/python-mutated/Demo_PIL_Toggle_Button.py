from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import PySimpleGUI as sg
'\n    Demo Toggle Buttons created using PIL\n\n    A BIG thank you to @jason990420 for his talented work creating this demo.\n\n    Demonstrates how you can draw buttons rather than using the built-in buttons\n    with the help of the PIL module.\n\n    One advantage is that the buttons will match your window\'s theme.\n\n    You\'ll find more PySimpleGUI programs featured in \n    Mike Driscoll\'s PIL book - "Pillow: Image Processing with Python"\n\n    Copyright 2021 PySimpleGUI, @jason990420 \n'

def im_to_data(im):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a PIL.Image.Image to a bytes image\n    :param im: Image\n    :type: PIL.Image.Image object\n    :return image in bytes\n    :type: bytes\n    '
    with BytesIO() as buffer:
        im.save(buffer, format='PNG')
        data = buffer.getvalue()
    return data

def toggle_button(button_color=None, size=(100, 40), on=True):
    if False:
        while True:
            i = 10
    '\n\n    :return image in bytes\n    :type: bytes\n    '
    (pad, radius, spacing) = (5, 10, 5)
    (w, h) = size
    (c1, c2) = button_color if button_color else (sg.theme_input_background_color(), sg.theme_background_color())
    im = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), fill=c1, width=3, radius=radius)
    if on:
        draw.rounded_rectangle((2, 2, w - size[1], h - 3), fill=c2, width=3, radius=radius)
    else:
        draw.rounded_rectangle((size[1], 2, w - 2, h - 3), fill=c2, width=3, radius=radius)
    return im_to_data(im)
sg.theme('dark red')
button_on = toggle_button()
button_off = toggle_button(on=False)
layout = [[sg.T('PIL Made Toggle Buttons')], [sg.T('On'), sg.Button(image_data=button_on, button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0, k='-B1-', metadata=True), sg.T('Off')], [sg.Button(image_data=button_off, button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0, k='-B2-', metadata=False)]]
window = sg.Window('PIL Buttons', layout, finalize=True, element_justification='c', use_custom_titlebar=True, font='_ 18', keep_on_top=True)
while True:
    (event, values) = window.read()
    if event in (sg.WINDOW_CLOSED, 'Exit'):
        break
    window[event].metadata = not window[event].metadata
    window[event].update(image_data=button_on if window[event].metadata else button_off)
window.close()