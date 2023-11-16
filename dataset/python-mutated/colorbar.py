"""
Plot different styles of ColorBar
=================================
"""
from vispy import plot as vp
import numpy as np

def exp_z_inv(x, y):
    if False:
        i = 10
        return i + 15
    z = complex(x, y)
    f = np.exp(1.0 / z)
    return np.angle(f, deg=True)

def gen_image(width, height):
    if False:
        while True:
            i = 10
    x_vals = np.linspace(-0.5, 0.5, width)
    y_vals = np.linspace(-0.5, 0.5, height)
    grid = np.meshgrid(x_vals, y_vals)
    v_fn = np.vectorize(exp_z_inv)
    return v_fn(*grid).astype(np.float32)
fig = vp.Fig(size=(800, 600), show=False)
plot = fig[0, 0]
plot.bgcolor = '#efefef'
img = gen_image(500, 500)
plot.image(img, cmap='hsl')
plot.camera.set_range((100, 400), (100, 400))
positions = ['top', 'bottom', 'left', 'right']
for position in positions:
    plot.colorbar(position=position, label='argument of e^(1/z)', clim=('0°', '180°'), cmap='hsl', border_width=1, border_color='#aeaeae')
if __name__ == '__main__':
    fig.show(run=True)