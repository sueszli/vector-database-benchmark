import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.pyplot as plt
import io
import time
'\n    Demo_Matplotlib_Image_Elem_Spetrogram_Animated Demo\n\n    Demo to show\n        * How to use an Image element to show a Matplotlib figure\n        * How to draw a Spectrogram\n        * How to animate the drawing by simply erasing and drawing the entire figure\n\n    The point here is to keep things simple to enable you to get started.\n    \n    The example static graph can be found in the matplotlib gallery:\n    https://matplotlib.org/stable/gallery/images_contours_and_fields/specgram_demo.html        \n\n    Copyright 2021 PySimpleGUI\n'
np.random.seed(19801)

def your_matplotlib_code():
    if False:
        return 10
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

def draw_figure(element, figure):
    if False:
        print('Hello World!')
    '\n    Draws the previously created "figure" in the supplied Image Element\n\n    :param element: an Image Element\n    :param figure: a Matplotlib figure\n    :return: The figure canvas\n    '
    plt.close('all')
    canv = FigureCanvasAgg(figure)
    buf = io.BytesIO()
    canv.print_figure(buf, format='png')
    if buf is not None:
        buf.seek(0)
        element.update(data=buf.read())
        return canv
    else:
        return None

def main():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Text('Spectrogram Animated - Not Threaded', font='Helvetica 24')], [sg.pin(sg.Image(key='-IMAGE-'))], [sg.T(size=(50, 1), k='-STATS-')], [sg.B('Animate', focus=True, k='-ANIMATE-')]]
    window = sg.Window('Animated Spectrogram', layout, element_justification='c', font='Helvetica 14')
    counter = delta = start_time = 0
    timeout = None
    while True:
        (event, values) = window.read(timeout=timeout)
        if event == sg.WIN_CLOSED:
            break
        sg.timer_start()
        if event == '-ANIMATE-':
            timeout = 0
            window['-IMAGE-'].update(visible=True)
            start_time = time.time()
        elif event == sg.TIMEOUT_EVENT:
            plt.close('all')
            window['-IMAGE-'].update()
            draw_figure(window['-IMAGE-'], your_matplotlib_code())
            seconds_elapsed = int(time.time() - start_time)
            fps = counter / seconds_elapsed if seconds_elapsed != 0 else 1.0
            window['-STATS-'].update(f'Frame {counter} Write Time {delta} FPS = {fps:2.2} seconds = {seconds_elapsed}')
            counter += 1
        delta = sg.timer_stop()
    window.close()
if __name__ == '__main__':
    main()