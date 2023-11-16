import PySimpleGUI as sg
import cv2
'\n    Demonstration of how to use a GRAPH ELEMENT to draw a webcam stream using OpenCV and PySimpleGUI.\n    Additionally, the thing this demo is really showcasing, is the ability to draw over the top of this\n    webcam stream, as it\'s being displayed.  To "Draw" simply move your mouse over the image, left click and hold, and\n    then drag your mouse.  You\'ll see a series of red circles on top of your image.\n    CURRENTLY ONLY WORKS WITH PySimpleGUI, NOT any of the other ports at this time.\n    \n    Note also that this demo is using ppm as the image format.  This worked fine on all PySimpleGUI ports except \n    the web port.  If you have trouble with the imencode statement, change "ppm" to "png"\n    \n    Copyright 2021 PySimpleGUI\n'

def main():
    if False:
        i = 10
        return i + 15
    layout = ([[sg.Graph((600, 450), (0, 450), (600, 0), key='-GRAPH-', enable_events=True, drag_submits=True)]],)
    window = sg.Window('Demo Application - OpenCV Integration', layout)
    graph_elem = window['-GRAPH-']
    a_id = None
    cap = cv2.VideoCapture(0)
    while True:
        (event, values) = window.read(timeout=0)
        if event in ('Exit', None):
            break
        (ret, frame) = cap.read()
        imgbytes = cv2.imencode('.ppm', frame)[1].tobytes()
        if a_id:
            graph_elem.delete_figure(a_id)
        a_id = graph_elem.draw_image(data=imgbytes, location=(0, 0))
        graph_elem.send_figure_to_back(a_id)
        if event == '-GRAPH-':
            graph_elem.draw_circle(values['-GRAPH-'], 5, fill_color='red', line_color='red')
    window.close()
main()