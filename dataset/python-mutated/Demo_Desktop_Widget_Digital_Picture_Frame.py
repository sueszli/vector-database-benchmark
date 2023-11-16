import PySimpleGUI as sg
import datetime
import PIL
from PIL import Image
import random
import os
import io
import base64
'\n    Another simple Desktop Widget using PySimpleGUI\n\n    This one shows images from a folder of your choosing.\n    You can change the new "Standard Desktop Widget Settings"\n        * Theme, location, alpha channel, refresh info, \n\n    Specific to this Widget are\n        * Image size\n        * How long to show the image and if you wnt this time to vary semi-randomly\n        * Folder containing your images\n        \n    Copyright 2021, 2023 PySimpleGUI\n'
ALPHA = 0.9
refresh_font = sg.user_settings_get_entry('-refresh font-', 'Courier 8')

def make_square(im, fill_color=(0, 0, 0, 0)):
    if False:
        print('Hello World!')
    (x, y) = im.size
    size = max(x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def get_image_size(source):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(source, str):
        image = PIL.Image.open(source)
    elif isinstance(source, bytes):
        image = PIL.Image.open(io.BytesIO(base64.b64decode(source)))
    else:
        image = PIL.Image.open(io.BytesIO(source))
    (width, height) = image.size
    return (width, height)

def convert_to_bytes(source, size=(None, None), subsample=None, zoom=None, fill=False):
    if False:
        return 10
    '\n    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.\n    Turns into  PNG format in the process so that can be displayed by tkinter\n    :param source: either a string filename or a bytes base64 image object\n    :type source:  (Union[str, bytes])\n    :param size:  optional new size (width, height)\n    :type size: (Tuple[int, int] or None)\n    :param subsample: change the size by multiplying width and height by 1/subsample\n    :type subsample: (int)\n    :param zoom: change the size by multiplying width and height by zoom\n    :type zoom: (int)\n    :param fill: If True then the image is filled/padded so that the image is square\n    :type fill: (bool)\n    :return: (bytes) a byte-string object\n    :rtype: (bytes)\n    '
    if isinstance(source, str):
        image = PIL.Image.open(source)
    elif isinstance(source, bytes):
        image = PIL.Image.open(io.BytesIO(base64.b64decode(source)))
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

def choose_theme(location):
    if False:
        while True:
            i = 10
    layout = [[sg.Text(f'Current theme {sg.theme()}')], [sg.Listbox(values=sg.theme_list(), size=(20, 20), key='-LIST-')], [sg.OK(), sg.Cancel()]]
    (event, values) = sg.Window('Look and Feel Browser', layout, location=location, keep_on_top=True).read(close=True)
    if event == 'OK' and values['-LIST-']:
        sg.theme(values['-LIST-'][0])
        sg.user_settings_set_entry('-theme-', values['-LIST-'][0])
        return values['-LIST-'][0]
    else:
        return None

def reset_settings():
    if False:
        while True:
            i = 10
    sg.user_settings_set_entry('-time per image-', 60)
    sg.user_settings_set_entry('-random time-', False)
    sg.user_settings_set_entry('-image size-', (None, None))
    sg.user_settings_set_entry('-image_folder-', None)
    sg.user_settings_set_entry('-location-', (None, None))
    sg.user_settings_set_entry('-single image-', None)
    sg.user_settings_set_entry('-alpha-', ALPHA)

def make_window(location):
    if False:
        return 10
    alpha = sg.user_settings_get_entry('-alpha-', ALPHA)
    right_click_menu = [[''], ['Choose Image Folder', 'Choose Single Image', 'Edit Me', 'Change Theme', 'Set Image Size', 'Set Time Per Image', 'Save Location', 'Refresh', 'Show Refresh Info', 'Hide Refresh Info', 'Alpha', [str(x) for x in range(1, 11)], 'Exit']]
    refresh_info = [[sg.T(size=(25, 1), font=refresh_font, k='-REFRESHED-', justification='c')], [sg.T(size=(40, 1), justification='c', font=refresh_font, k='-FOLDER-')], [sg.T(size=(40, 1), justification='c', font=refresh_font, k='-FILENAME-')]]
    layout = [[sg.Image(k='-IMAGE-', enable_events=True)], [sg.pin(sg.Column(refresh_info, key='-REFRESH INFO-', element_justification='c', visible=sg.user_settings_get_entry('-show refresh-', True)))]]
    window = sg.Window('Photo Frame', layout, location=location, no_titlebar=True, grab_anywhere=True, margins=(0, 0), element_justification='c', element_padding=(0, 0), alpha_channel=alpha, finalize=True, right_click_menu=right_click_menu, keep_on_top=True, enable_close_attempted_event=True, enable_window_config_events=True)
    return window

def main():
    if False:
        return 10
    loc = sg.user_settings_get_entry('-location-', (None, None))
    sg.theme(sg.user_settings_get_entry('-theme-', None))
    time_per_image = sg.user_settings_get_entry('-time per image-', 60)
    vary_randomly = sg.user_settings_get_entry('-random time-', False)
    (width, height) = sg.user_settings_get_entry('-image size-', (None, None))
    image_folder = sg.user_settings_get_entry('-image_folder-', None)
    try:
        os.listdir(image_folder)
    except:
        image_folder = None
        sg.user_settings_set_entry('-image_folder-', None)
    image_name = single_image = sg.user_settings_get_entry('-single image-', None)
    if image_folder is None and single_image is None:
        image_name = single_image = sg.popup_get_file('Choose a starting image', keep_on_top=True)
        if not single_image:
            if sg.popup_yes_no('No folder entered', 'Go you want to exit the program entirely?', keep_on_top=True) == 'Yes':
                exit()
    if image_folder is not None and single_image is None:
        images = os.listdir(image_folder)
        images = [i for i in images if i.lower().endswith(('.png', '.jpg', '.gif'))]
        image_name = os.path.join(image_folder, random.choice(images))
    else:
        images = None
        image_name = single_image
    window = make_window(loc)
    window_size = window.size
    image_data = convert_to_bytes(image_name, (width, height))
    while True:
        timeout = time_per_image * 1000 + (random.randint(int(-time_per_image * 500), int(time_per_image * 500)) if vary_randomly else 0) if single_image is None else None
        (event, values) = window.read(timeout=timeout)
        if event == sg.WIN_CLOSED:
            break
        elif event in (sg.WIN_CLOSE_ATTEMPTED_EVENT, 'Exit'):
            sg.user_settings_set_entry('-location-', window.current_location())
            break
        if event == sg.TIMEOUT_EVENT:
            if single_image is None:
                image_name = random.choice(images)
                image_data = convert_to_bytes(os.path.join(image_folder, image_name))
                window['-FOLDER-'].update(image_folder)
            else:
                image_name = single_image
                image_data = convert_to_bytes(single_image, (width, height))
        window['-FILENAME-'].update(image_name)
        window['-IMAGE-'].update(data=image_data)
        window['-REFRESHED-'].update(datetime.datetime.now().strftime('%m/%d/%Y %I:%M:%S %p'))
        if event == sg.WINDOW_CONFIG_EVENT:
            new_size = window.size
            if new_size != window_size:
                print(f'resizing {new_size}')
                (width, height) = new_size
                image_data = convert_to_bytes(image_data, (width, height))
                window['-IMAGE-'].update(data=image_data)
                window.size = get_image_size(image_data)
                window_size = window.size
        if event == 'Edit Me':
            sg.execute_editor(__file__)
        elif event == 'Choose Image Folder':
            folder = sg.popup_get_folder('Choose location of your images', default_path=image_folder, location=window.current_location(), keep_on_top=True)
            if folder is not None:
                image_folder = folder
                window['-FOLDER-'].update(image_folder)
                sg.user_settings_set_entry('-image_folder-', image_folder)
                images = os.listdir(image_folder)
                images = [i for i in images if i.lower().endswith(('.png', '.jpg', '.gif'))]
                sg.user_settings_set_entry('-single image-', None)
                single_image = None
        elif event == 'Set Time Per Image':
            layout = [[sg.T('Enter number of seconds each image should be displayed')], [sg.I(time_per_image, size=(5, 1), k='-TIME PER IMAGE-')], [sg.CB('Use some randomness', vary_randomly, k='-RANDOM TIME-')], [sg.Ok(), sg.Cancel()]]
            (event, values) = sg.Window('Display duration', layout, location=window.current_location(), keep_on_top=True, no_titlebar=True).read(close=True)
            if event == 'Ok':
                try:
                    time_per_image = int(values['-TIME PER IMAGE-'])
                    vary_randomly = values['-RANDOM TIME-']
                    sg.user_settings_set_entry('-time per image-', time_per_image)
                    sg.user_settings_set_entry('-random time-', values['-RANDOM TIME-'])
                except:
                    sg.popup_error('Bad number of seconds entered', location=window.current_location(), keep_on_top=True)
        elif event == 'Set Image Size':
            layout = [[sg.T('Enter size should be shown at in pixels (width, height)')], [sg.I(width, size=(4, 1), k='-W-'), sg.I(height, size=(4, 1), k='-H-')], [sg.Ok(), sg.Cancel()]]
            (event, values) = sg.Window('Image Dimensions', layout, location=window.current_location(), keep_on_top=True, no_titlebar=True).read(close=True)
            if event == 'Ok':
                try:
                    (w, h) = (int(values['-W-']), int(values['-H-']))
                    sg.user_settings_set_entry('-image size-', (w, h))
                    (width, height) = (w, h)
                except:
                    sg.popup_error('Bad size specified. Use integers only', location=window.current_location(), keep_on_top=True)
        elif event == 'Show Refresh Info':
            window['-REFRESH INFO-'].update(visible=True)
            sg.user_settings_set_entry('-show refresh-', True)
        elif event == 'Save Location':
            sg.user_settings_set_entry('-location-', window.current_location())
        elif event == 'Hide Refresh Info':
            window['-REFRESH INFO-'].update(visible=False)
            sg.user_settings_set_entry('-show refresh-', False)
        elif event in [str(x) for x in range(1, 11)]:
            window.set_alpha(int(event) / 10)
            sg.user_settings_set_entry('-alpha-', int(event) / 10)
        elif event == 'Change Theme':
            loc = window.current_location()
            if choose_theme(loc) is not None:
                window.close()
                window = make_window(loc)
        elif event == 'Choose Single Image':
            image_name = single_image = sg.popup_get_file('Choose single image to show', history=True)
            sg.user_settings_set_entry('-single image-', single_image)
            (width, height) = get_image_size(single_image)
            sg.user_settings_set_entry('-image size-', (width, height))
            image_data = convert_to_bytes(image_name, (width, height))
            window['-IMAGE-'].update(data=image_data)
            window.size = window_size = (width, height)
    window.close()
if __name__ == '__main__':
    main()