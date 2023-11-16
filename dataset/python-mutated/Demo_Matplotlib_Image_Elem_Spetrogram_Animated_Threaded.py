import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.pyplot as plt
import io
import time
"\n    Demo_Matplotlib_Image_Elem_Spetrogram_Animated_Threaded Demo\n\n    Demo to show\n        * How to use an Image element to show a Matplotlib figure\n        * How to draw a Spectrogram\n        * How to animate the drawing by simply erasing and drawing the entire figure\n        * How to communicate between a thread and the GUI\n\n    The point here is to keep things simple to enable you to get started.\n\n    NOTE:\n        This threaded technique with matplotlib hasn't been thoroughly tested.\n        There may be resource leaks for example.  Have run for several hundred seconds\n        without problems so it's perhaps safe as written.\n\n    The example static graph can be found in the matplotlib gallery:\n    https://matplotlib.org/stable/gallery/images_contours_and_fields/specgram_demo.html        \n\n    Copyright 2021, 2022 PySimpleGUI\n"
np.random.seed(19801)

def the_thread(window: sg.Window):
    if False:
        while True:
            i = 10
    "\n    The thread that communicates with the application through the window's events.\n\n    Because the figure creation time is greater than the GUI drawing time, it's safe\n    to send a non-regulated stream of events without fear of overrunning the communication queue\n    "
    while True:
        fig = your_matplotlib_code()
        buf = draw_figure(fig)
        window.write_event_value('-THREAD-', buf)

def your_matplotlib_code():
    if False:
        print('Hello World!')
    if not hasattr(your_matplotlib_code, 't_lower'):
        your_matplotlib_code.t_lower = 10
        your_matplotlib_code.t_upper = 12
    else:
        your_matplotlib_code.t_lower = (your_matplotlib_code.t_lower + 0.5) % 18
        your_matplotlib_code.t_upper = (your_matplotlib_code.t_upper + 0.5) % 18
    dt = 0.0005
    t = np.arange(0.0, 20.0, dt)
    s1 = np.sin(2 * np.pi * 100 * t)
    s2 = 2 * np.sin(2 * np.pi * 400 * t)
    s2[t <= your_matplotlib_code.t_lower] = s2[your_matplotlib_code.t_upper <= t] = 0
    nse = 0.01 * np.random.random(size=len(t))
    x = s1 + s2 + nse
    NFFT = 1024
    Fs = int(1.0 / dt)
    (fig, ax2) = plt.subplots(nrows=1)
    (Pxx, freqs, bins, im) = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
    return fig

def draw_figure(figure):
    if False:
        i = 10
        return i + 15
    '\n    Draws the previously created "figure" in the supplied Image Element\n\n    :param figure: a Matplotlib figure\n    :return: BytesIO object\n    '
    plt.close('all')
    canv = FigureCanvasAgg(figure)
    buf = io.BytesIO()
    canv.print_figure(buf, format='png')
    if buf is not None:
        buf.seek(0)
        return buf
    else:
        return None

def main():
    if False:
        while True:
            i = 10
    layout = [[sg.Text('Spectrogram Animated - Threaded', font='Helvetica 24')], [sg.pin(sg.Image(key='-IMAGE-'))], [sg.T(k='-STATS-')], [sg.B('Animate', focus=True, k='-ANIMATE-')]]
    window = sg.Window('Animated Spectrogram', layout, element_justification='c', font='Helvetica 14')
    counter = start_time = delta = 0
    while True:
        (event, values) = window.read()
        if event == sg.WIN_CLOSED:
            break
        sg.timer_start()
        if event == '-ANIMATE-':
            window['-IMAGE-'].update(visible=True)
            start_time = time.time()
            window.start_thread(lambda : the_thread(window), '-THEAD FINISHED-')
        elif event == '-THREAD-':
            plt.close('all')
            window['-IMAGE-'].update(data=values[event].read())
            counter += 1
            seconds_elapsed = int(time.time() - start_time)
            fps = counter / seconds_elapsed if seconds_elapsed != 0 else 1.0
            window['-STATS-'].update(f'Frame {counter} Write Time {delta} FPS = {fps:2.2} seconds = {seconds_elapsed}')
        delta = sg.timer_stop()
    window.close()
if __name__ == '__main__':
    sg.Window.start_thread = sg.Window.perform_long_operation
    main()