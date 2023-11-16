import PySimpleGUI as sg
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
'\nDemonstrates one way of embedding Matplotlib figures into a PySimpleGUI window.\n\nPaste your Pyplot code into the section marked below.\n\nDo all of your plotting as you normally would, but do NOT call plt.show(). \nStop just short of calling plt.show() and let the GUI do the rest.\n\nThe remainder of the program will convert your plot and display it in the GUI.\nIf you want to change the GUI, make changes to the GUI portion marked below.\n\n'
values_to_plot = (20, 35, 30, 35, 27)
ind = np.arange(len(values_to_plot))
width = 0.4
p1 = plt.bar(ind, values_to_plot, width)
plt.ylabel('Y-Axis Values')
plt.title('Plot Title')
plt.xticks(ind, ('Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0],), ('Data Group 1',))

def draw_figure(canvas, figure, loc=(0, 0)):
    if False:
        for i in range(10):
            print('nop')
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
sg.theme('Light Brown 3')
fig = plt.gcf()
(figure_x, figure_y, figure_w, figure_h) = fig.bbox.bounds
layout = [[sg.Text('Plot test', font='Any 18')], [sg.Canvas(size=(figure_w, figure_h), key='-CANVAS-')], [sg.OK(pad=((figure_w / 2, 0), 3), size=(4, 2))]]
window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, force_toplevel=True, finalize=True)
fig_photo = draw_figure(window['-CANVAS-'].TKCanvas, fig)
(event, values) = window.read()
window.close()