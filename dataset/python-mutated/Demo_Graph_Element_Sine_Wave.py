import PySimpleGUI as sg
import math
'\n    Demo - Graph Element used to plot a mathematical formula\n    \n    The Graph element has a flexible coordinate system that you define.\n    Thie makes is possible for you to work in your coordinates instead of an\n    arbitrary system.\n    \n    For example, in a typical mathematics graph, (0,0) is located at the center\n    of the graph / page / diagram.\n    This Demo Program shows a graph with (0,0) being at the center of the Graph\n    area rather than at one of the corners.\n    \n    It graphs the formula:\n        y = sine(x/x2) * x1\n        \n    The values of x1 and x2 can be changed using 2 sliders\n\n    Copyright 2018, 2019, 2020, 2021, 2022 PySimpleGUI\n    \n'
SIZE_X = 200
SIZE_Y = 200
NUMBER_MARKER_FREQUENCY = SIZE_X // 8

def draw_axis():
    if False:
        for i in range(10):
            print('nop')
    graph.draw_line((-SIZE_X, 0), (SIZE_X, 0))
    graph.draw_line((0, -SIZE_Y), (0, SIZE_Y))
    for x in range(-SIZE_X, SIZE_X + 1, NUMBER_MARKER_FREQUENCY):
        graph.draw_line((x, -SIZE_Y / 66), (x, SIZE_Y / 66))
        if x != 0:
            graph.draw_text(str(x), (x, -SIZE_Y / 15), color='green', font='courier 10')
    for y in range(-SIZE_Y, SIZE_Y + 1, NUMBER_MARKER_FREQUENCY):
        graph.draw_line((-SIZE_X / 66, y), (SIZE_X / 66, y))
        if y != 0:
            graph.draw_text(str(y), (-SIZE_X / 11, y), color='blue', font='courier 10')
graph = sg.Graph(canvas_size=(500, 500), graph_bottom_left=(-(SIZE_X + 5), -(SIZE_Y + 5)), graph_top_right=(SIZE_X + 5, SIZE_Y + 5), background_color='white', expand_x=True, expand_y=True, key='-GRAPH-')
layout = [[sg.Text('Graph Element Combined with Math!', justification='center', relief=sg.RELIEF_SUNKEN, expand_x=True, font='Courier 18')], [graph], [sg.Text('y = sin(x / x2) * x1', font='COURIER 18')], [sg.Text('x1', font='Courier 14'), sg.Slider((0, SIZE_Y), orientation='h', enable_events=True, key='-SLIDER-', expand_x=True)], [sg.Text('x2', font='Courier 14'), sg.Slider((1, SIZE_Y), orientation='h', enable_events=True, key='-SLIDER2-', expand_x=True)]]
window = sg.Window('Graph of Sine Function', layout, finalize=True)
draw_axis()
while True:
    (event, values) = window.read()
    if event == sg.WIN_CLOSED:
        break
    graph.erase()
    draw_axis()
    prev_x = prev_y = None
    for x in range(-SIZE_X, SIZE_X):
        y = math.sin(x / int(values['-SLIDER2-'])) * int(values['-SLIDER-'])
        if prev_x is not None:
            graph.draw_line((prev_x, prev_y), (x, y), color='red')
        (prev_x, prev_y) = (x, y)
window.close()