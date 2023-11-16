from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
matplotlib.use('TkAgg')
'\nDemonstrates one way of embedding Matplotlib figures into a PySimpleGUI window.\n\nBasic steps are:\n * Create a Canvas Element\n * Layout form\n * Display form (NON BLOCKING)\n * Draw plots onto convas\n * Display form (BLOCKING)\n \n Based on information from: https://matplotlib.org/3.1.0/gallery/user_interfaces/embedding_in_tk_sgskip.html\n (Thank you Em-Bo & dirck)\n'
fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, 0.01)
fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

def draw_figure(canvas, figure):
    if False:
        while True:
            i = 10
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
layout = [[sg.Text('Plot test')], [sg.Canvas(key='-CANVAS-')], [sg.Button('Ok')]]
window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, finalize=True, element_justification='center', font='Helvetica 18')
fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
(event, values) = window.read()
window.close()