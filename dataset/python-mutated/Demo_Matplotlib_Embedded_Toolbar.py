import PySimpleGUI as sg
import numpy as np
'\n    Embedding the Matplotlib toolbar into your application\n\n'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if False:
        while True:
            i = 10
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)

class Toolbar(NavigationToolbar2Tk):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(Toolbar, self).__init__(*args, **kwargs)
layout = [[sg.T('Graph: y=sin(x)')], [sg.B('Plot'), sg.B('Exit')], [sg.T('Controls:')], [sg.Canvas(key='controls_cv')], [sg.T('Figure:')], [sg.Column(layout=[[sg.Canvas(key='fig_cv', size=(400 * 2, 400))]], background_color='#DAE0E6', pad=(0, 0))], [sg.B('Alive?')]]
window = sg.Window('Graph with controls', layout)
while True:
    (event, values) = window.read()
    print(event, values)
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event is 'Plot':
        plt.figure(1)
        fig = plt.gcf()
        DPI = fig.get_dpi()
        fig.set_size_inches(404 * 2 / float(DPI), 404 / float(DPI))
        x = np.linspace(0, 2 * np.pi)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title('y=sin(x)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        draw_figure_w_toolbar(window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)
window.close()