from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
import pylab
matplotlib.use('TkAgg')
'\nDemonstrates one way of embedding PyLab figures into a PySimpleGUI window.\n'
from numpy import sin
from numpy import cos
x = pylab.linspace(-3, 3, 30)
y = x ** 2
pylab.plot(x, sin(x))
pylab.plot(x, cos(x), 'r-')
pylab.plot(x, -sin(x), 'g--')
fig = pylab.gcf()
(figure_x, figure_y, figure_w, figure_h) = fig.bbox.bounds

def draw_figure(canvas, figure, loc=(0, 0)):
    if False:
        print('Hello World!')
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
layout = [[sg.Text('Plot test', font='Any 18')], [sg.Canvas(size=(figure_w, figure_h), key='canvas')], [sg.OK(pad=((figure_w / 2, 0), 3), size=(4, 2))]]
window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, finalize=True)
fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)
(event, values) = window.read()
window.close()