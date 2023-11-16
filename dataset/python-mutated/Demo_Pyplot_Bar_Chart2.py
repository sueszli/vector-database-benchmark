import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
'\nDemonstrates one way of embedding Matplotlib figures into a PySimpleGUI window.\n\nPaste your Pyplot code into the section marked below.\n\nDo all of your plotting as you normally would, but do NOT call plt.show().\nStop just short of calling plt.show() and let the GUI do the rest.\n\nThe remainder of the program will convert your plot and display it in the GUI.\nIf you want to change the GUI, make changes to the GUI portion marked below.\n\n'
label = ['Adventure', 'Action', 'Drama', 'Comedy', 'Thriller/Suspense', 'Horror', 'Romantic Comedy', 'Musical', 'Documentary', 'Black Comedy', 'Western', 'Concert/Performance', 'Multiple Genres', 'Reality']
no_movies = [941, 854, 4595, 2125, 942, 509, 548, 149, 1952, 161, 64, 61, 35, 5]
index = np.arange(len(label))
plt.bar(index, no_movies)
plt.xlabel('Genre', fontsize=5)
plt.ylabel('No of Movies', fontsize=5)
plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Market Share for Each Genre 1995-2017')

def draw_figure(canvas, figure, loc=(0, 0)):
    if False:
        print('Hello World!')
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