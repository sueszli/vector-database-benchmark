import PySimpleGUI as sg
import cv2 as cv
"\n    Demo program to open and play a file using OpenCV\n    It's main purpose is to show you:\n    1. How to get a frame at a time from a video file using OpenCV\n    2. How to display an image in a PySimpleGUI Window\n    \n    For added fun, you can reposition the video using the slider.\n    \n    Copyright 2022 PySimpleGUI\n"

def main():
    if False:
        for i in range(10):
            print('nop')
    filename = sg.popup_get_file('Filename to play')
    if filename is None:
        return
    vidFile = cv.VideoCapture(filename)
    num_frames = vidFile.get(cv.CAP_PROP_FRAME_COUNT)
    fps = vidFile.get(cv.CAP_PROP_FPS)
    sg.theme('Black')
    layout = [[sg.Text('OpenCV Demo', size=(15, 1), font='Helvetica 20')], [sg.Image(key='-IMAGE-')], [sg.Slider(range=(0, num_frames), size=(60, 10), orientation='h', key='-SLIDER-')], [sg.Push(), sg.Button('Exit', font='Helvetica 14')]]
    window = sg.Window('Demo Application - OpenCV Integration', layout, no_titlebar=False, location=(0, 0))
    image_elem = window['-IMAGE-']
    slider_elem = window['-SLIDER-']
    timeout = 1000 // fps
    cur_frame = 0
    while vidFile.isOpened():
        (event, values) = window.read(timeout=timeout)
        if event in ('Exit', None):
            break
        (ret, frame) = vidFile.read()
        if not ret:
            break
        if int(values['-SLIDER-']) != cur_frame - 1:
            cur_frame = int(values['-SLIDER-'])
            vidFile.set(cv.CAP_PROP_POS_FRAMES, cur_frame)
        slider_elem.update(cur_frame)
        cur_frame += 1
        imgbytes = cv.imencode('.ppm', frame)[1].tobytes()
        image_elem.update(data=imgbytes)
main()
"         #############\n        # This was another way updates were being done, but seems slower than the above\n        img = Image.fromarray(frame)    # create PIL image from frame\n        bio = io.BytesIO()              # a binary memory resident stream\n        img.save(bio, format= 'PNG')    # save image as png to it\n        imgbytes = bio.getvalue()       # this can be used by OpenCV hopefully\n        image_elem.update(data=imgbytes)\n"