import PySimpleGUI as sg
import win32gui
from PIL import ImageGrab
'\n    Demo - Save Window screenshot\n    Works on WINDOWS only.\n    Saves a window as an image file.  Tested saving as PNG and JPG.  Input the title of the Window and it will be\n    saved in the format indicated by the filename.\n\n    Copyright 2020, 2022 PySimpleGUI.org\n'

def save_win(filename=None, title=None):
    if False:
        return 10
    '\n    Saves a window with the title provided as a file using the provided filename.\n    If one of them is missing, then a window is created and the information collected\n\n    :param filename:\n    :param title:\n    :return:\n    '
    C = 7
    if filename is None or title is None:
        layout = [[sg.T('Choose window to save', font='Any 18')], [sg.T('The extension you choose for filename will determine the image format')], [sg.T('Window Title:', size=(12, 1)), sg.I(title if title is not None else '', key='-T-')], [sg.T('Filename:', size=(12, 1)), sg.I(filename if filename is not None else '', key='-F-')], [sg.Button('Ok', bind_return_key=True), sg.Button('Cancel')]]
        (event, values) = sg.Window('Choose Win Title and Filename', layout).read(close=True)
        if event != 'Ok':
            print('Cancelling the save')
            return
        (filename, title) = (values['-F-'], values['-T-'])
    try:
        fceuxHWND = win32gui.FindWindow(None, title)
        rect = win32gui.GetWindowRect(fceuxHWND)
        rect_cropped = (rect[0] + C, rect[1], rect[2] - C, rect[3] - C)
        grab = ImageGrab.grab(bbox=rect_cropped)
        grab.save(filename)
        sg.popup('Wrote image to file:', filename)
    except Exception as e:
        sg.popup('Error trying to save screenshot file', e)
if __name__ == '__main__':
    save_win()