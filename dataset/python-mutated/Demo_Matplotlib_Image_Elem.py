import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.pyplot as plt
import io
"\n    Demo_Matplotlib_Image_Elem Demo\n    \n    Demo to show\n        * How to use an Image element to show a Matplotlib figure\n        * How to draw a Spectrogram\n        * Hide the Image when a figure isn't present (shrinks the window automatically)\n        \n    The example graph can be found in the matplotlib gallery:\n    https://matplotlib.org/stable/gallery/images_contours_and_fields/specgram_demo.html        \n    \n    Copyright 2021 PySimpleGUI\n"

def your_matplotlib_code():
    if False:
        while True:
            i = 10
    np.random.seed(19680801)
    dt = 0.0005
    t = np.arange(0.0, 20.0, dt)
    s1 = np.sin(2 * np.pi * 100 * t)
    s2 = 2 * np.sin(2 * np.pi * 400 * t)
    s2[t <= 10] = s2[12 <= t] = 0
    nse = 0.01 * np.random.random(size=len(t))
    x = s1 + s2 + nse
    NFFT = 1024
    Fs = int(1.0 / dt)
    (fig, (ax1, ax2)) = plt.subplots(nrows=2)
    ax1.plot(t, x)
    (Pxx, freqs, bins, im) = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
    return fig

def draw_figure(element, figure):
    if False:
        for i in range(10):
            print('nop')
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
        while True:
            i = 10
    layout = [[sg.Text('Spectrogram test')], [sg.pin(sg.Image(key='-IMAGE-'))], [sg.Button('Ok'), sg.B('Clear')]]
    window = sg.Window('Spectrogram', layout, element_justification='c', font='Helvetica 14')
    while True:
        (event, values) = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Ok':
            draw_figure(window['-IMAGE-'], your_matplotlib_code())
            window['-IMAGE-'].update(visible=True)
        elif event == 'Clear':
            plt.close('all')
            window['-IMAGE-'].update()
            window['-IMAGE-'].update(visible=False)
    window.close()
if __name__ == '__main__':
    main()