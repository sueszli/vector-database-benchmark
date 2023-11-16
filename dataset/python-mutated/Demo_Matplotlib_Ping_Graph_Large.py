from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import matplotlib.backends.tkagg as tkagg
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import tkinter as tk
import ping

class MyGlobals:
    axis_pings = None
    ping_x_array = []
    ping_y_array = []
g_my_globals = MyGlobals()

def run_a_ping_and_graph():
    if False:
        i = 10
        return i + 15
    global g_my_globals
    response = ping.quiet_ping('google.com', timeout=1000)
    if response[0] == 0:
        ping_time = 1000
    else:
        ping_time = response[0]
    g_my_globals.ping_x_array.append(len(g_my_globals.ping_x_array))
    g_my_globals.ping_y_array.append(ping_time)
    if len(g_my_globals.ping_x_array) > 100:
        x_array = g_my_globals.ping_x_array[-100:]
        y_array = g_my_globals.ping_y_array[-100:]
    else:
        x_array = g_my_globals.ping_x_array
        y_array = g_my_globals.ping_y_array
    g_my_globals.axis_ping.clear()
    g_my_globals.axis_ping.plot(x_array, y_array)

def set_chart_labels():
    if False:
        print('Hello World!')
    global g_my_globals
    g_my_globals.axis_ping.set_xlabel('Time')
    g_my_globals.axis_ping.set_ylabel('Ping (ms)')
    g_my_globals.axis_ping.set_title('Current Ping Duration', fontsize=12)

def draw(fig, canvas):
    if False:
        while True:
            i = 10
    (figure_x, figure_y, figure_w, figure_h) = fig.bbox.bounds
    (figure_w, figure_h) = (int(figure_w), int(figure_h))
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)
    canvas.create_image(640 / 2, 480 / 2, image=photo)
    figure_canvas_agg = FigureCanvasAgg(fig)
    figure_canvas_agg.draw()
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)
    return photo

def main():
    if False:
        while True:
            i = 10
    global g_my_globals
    layout = [[sg.Text('Animated Ping', size=(40, 1), justification='center', font='Helvetica 20')], [sg.Canvas(size=(640, 480), key='canvas')], [sg.Button('Exit', size=(10, 2), pad=((280, 0), 3), font='Helvetica 14')]]
    window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, finalize=True)
    canvas_elem = window['canvas']
    canvas = canvas_elem.TKCanvas
    fig = plt.figure()
    g_my_globals.axis_ping = fig.add_subplot(1, 1, 1)
    set_chart_labels()
    plt.tight_layout()
    while True:
        (event, values) = window.read(timeout=0)
        if event in ('Exit', None):
            break
        run_a_ping_and_graph()
        photo = draw(fig, canvas)
if __name__ == '__main__':
    main()