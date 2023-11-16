"""
============
Axes divider
============

Axes divider to calculate location of axes and
create a divider for them using existing axes instances.
"""
import matplotlib.pyplot as plt
from matplotlib import cbook

def get_demo_image():
    if False:
        i = 10
        return i + 15
    z = cbook.get_sample_data('axes_grid/bivariate_normal.npy')
    return (z, (-3, 4, -4, 3))

def demo_simple_image(ax):
    if False:
        i = 10
        return i + 15
    (Z, extent) = get_demo_image()
    im = ax.imshow(Z, extent=extent)
    cb = plt.colorbar(im)
    cb.ax.yaxis.set_tick_params(labelright=False)

def demo_locatable_axes_hard(fig):
    if False:
        for i in range(10):
            print('nop')
    from mpl_toolkits.axes_grid1 import Size, SubplotDivider
    divider = SubplotDivider(fig, 2, 2, 2, aspect=True)
    ax = fig.add_subplot(axes_locator=divider.new_locator(nx=0, ny=0))
    ax_cb = fig.add_subplot(axes_locator=divider.new_locator(nx=2, ny=0))
    divider.set_horizontal([Size.AxesX(ax), Size.Fixed(0.05), Size.Fixed(0.2)])
    divider.set_vertical([Size.AxesY(ax)])
    (Z, extent) = get_demo_image()
    im = ax.imshow(Z, extent=extent)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.set_tick_params(labelright=False)

def demo_locatable_axes_easy(ax):
    if False:
        for i in range(10):
            print('nop')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes('right', size='5%', pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    (Z, extent) = get_demo_image()
    im = ax.imshow(Z, extent=extent)
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=False)

def demo_images_side_by_side(ax):
    if False:
        return 10
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    (Z, extent) = get_demo_image()
    ax2 = divider.append_axes('right', size='100%', pad=0.05)
    fig1 = ax.get_figure()
    fig1.add_axes(ax2)
    ax.imshow(Z, extent=extent)
    ax2.imshow(Z, extent=extent)
    ax2.yaxis.set_tick_params(labelleft=False)

def demo():
    if False:
        return 10
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(2, 2, 1)
    demo_simple_image(ax)
    demo_locatable_axes_hard(fig)
    ax = fig.add_subplot(2, 2, 3)
    demo_locatable_axes_easy(ax)
    ax = fig.add_subplot(2, 2, 4)
    demo_images_side_by_side(ax)
    plt.show()
demo()