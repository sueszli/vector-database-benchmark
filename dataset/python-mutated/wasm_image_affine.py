import matplotlib
import numpy as np
matplotlib.use('module://matplotlib_pyodide.wasm_backend')
import matplotlib.transforms as mtransforms
from matplotlib import pyplot as plt

def wasm_image_affine():
    if False:
        while True:
            i = 10

    def get_image():
        if False:
            while True:
                i = 10
        delta = 0.25
        x = y = np.arange(-3.0, 3.0, delta)
        (X, Y) = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
        Z = Z1 - Z2
        return Z

    def do_plot(ax, Z, transform):
        if False:
            return 10
        im = ax.imshow(Z, interpolation='none', origin='lower', extent=[-2, 4, -3, 2], clip_on=True)
        trans_data = transform + ax.transData
        im.set_transform(trans_data)
        (x1, x2, y1, y2) = im.get_extent()
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'y--', transform=trans_data)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4, 4)
    (fig, ((ax1, ax2), (ax3, ax4))) = plt.subplots(2, 2)
    Z = get_image()
    do_plot(ax1, Z, mtransforms.Affine2D().rotate_deg(30))
    do_plot(ax2, Z, mtransforms.Affine2D().skew_deg(30, 15))
    do_plot(ax3, Z, mtransforms.Affine2D().scale(-1, 0.5))
    do_plot(ax4, Z, mtransforms.Affine2D().rotate_deg(30).skew_deg(30, 15).scale(-1, 0.5).translate(0.5, -1))
    plt.show()
    plt.close('all')
    plt.clf()