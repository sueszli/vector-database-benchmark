import PySimpleGUIWeb as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.figure
import matplotlib.pyplot as plt
import io
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.ticker import NullFormatter

def create_axis_grid():
    if False:
        i = 10
        return i + 15
    from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
    plt.close('all')

    def get_demo_image():
        if False:
            while True:
                i = 10
        delta = 0.5
        extent = (-3, 4, -4, 3)
        x = np.arange(-3.0, 4.001, delta)
        y = np.arange(-4.0, 3.001, delta)
        (X, Y) = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
        Z = (Z1 - Z2) * 2
        return (Z, extent)

    def get_rgb():
        if False:
            for i in range(10):
                print('nop')
        (Z, extent) = get_demo_image()
        Z[Z < 0] = 0.0
        Z = Z / Z.max()
        R = Z[:13, :13]
        G = Z[2:, 2:]
        B = Z[:13, 2:]
        return (R, G, B)
    fig = plt.figure(1)
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])
    (r, g, b) = get_rgb()
    kwargs = dict(origin='lower', interpolation='nearest')
    ax.imshow_rgb(r, g, b, **kwargs)
    ax.RGB.set_xlim(0.0, 9.5)
    ax.RGB.set_ylim(0.9, 10.6)
    plt.draw()
    return plt.gcf()

def create_figure():
    if False:
        return 10
    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    t = np.arange(0, 3, 0.01)
    fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
    return fig

def create_subplot_3d():
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    (X, Y) = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.set_zlim3d(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    (X, Y, Z) = get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    return fig

def create_pyplot_scales():
    if False:
        print('Hello World!')
    plt.close('all')
    np.random.seed(19680801)
    y = np.random.normal(loc=0.5, scale=0.4, size=1000)
    y = y[(y > 0) & (y < 1)]
    y.sort()
    x = np.arange(len(y))
    plt.figure(1)
    plt.subplot(221)
    plt.plot(x, y)
    plt.yscale('linear')
    plt.title('linear')
    plt.grid(True)
    plt.subplot(222)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('log')
    plt.grid(True)
    plt.subplot(223)
    plt.plot(x, y - y.mean())
    plt.yscale('symlog', linthreshy=0.01)
    plt.title('symlog')
    plt.grid(True)
    plt.subplot(224)
    plt.plot(x, y)
    plt.yscale('logit')
    plt.title('logit')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25, wspace=0.35)
    return plt.gcf()

def draw_figure(fig, element):
    if False:
        print('Hello World!')
    '\n    Draws the previously created "figure" in the supplied Image Element\n\n    :param fig: a Matplotlib figure\n    :param element: an Image Element\n    :return: The figure canvas\n    '
    plt.close('all')
    canv = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canv.print_figure(buf, format='png')
    if buf is None:
        return None
    buf.seek(0)
    element.update(data=buf.read())
    return canv

def main():
    if False:
        while True:
            i = 10
    dictionary_of_figures = {'Axis Grid': create_axis_grid, 'Subplot 3D': create_subplot_3d, 'Scales': create_pyplot_scales, 'Basic Figure': create_figure}
    layout = [[sg.T('Matplotlib Example', font='Any 20')], [sg.Listbox(dictionary_of_figures.keys(), size=(15, 5), key='-LB-'), sg.Image(key='-IMAGE-')], [sg.B('Draw'), sg.B('Exit')]]
    window = sg.Window('Title', layout, finalize=True)
    image_element = window['-IMAGE-']
    while True:
        (event, values) = window.read()
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        if event == 'Draw':
            func = dictionary_of_figures[values['-LB-'][0]]
            draw_figure(func(), image_element)
    window.close()
if __name__ == '__main__':
    main()