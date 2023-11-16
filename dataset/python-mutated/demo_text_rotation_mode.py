"""
==================
Text Rotation Mode
==================

This example illustrates the effect of ``rotation_mode`` on the positioning
of rotated text.

Rotated `.Text`\\s are created by passing the parameter ``rotation`` to
the constructor or the axes' method `~.axes.Axes.text`.

The actual positioning depends on the additional parameters
``horizontalalignment``, ``verticalalignment`` and ``rotation_mode``.
``rotation_mode`` determines the order of rotation and alignment:

- ``rotation_mode='default'`` (or None) first rotates the text and then aligns
  the bounding box of the rotated text.
- ``rotation_mode='anchor'`` aligns the unrotated text and then rotates the
  text around the point of alignment.

.. redirect-from:: /gallery/text_labels_and_annotations/text_rotation
"""
import matplotlib.pyplot as plt

def test_rotation_mode(fig, mode):
    if False:
        while True:
            i = 10
    ha_list = ['left', 'center', 'right']
    va_list = ['top', 'center', 'baseline', 'bottom']
    axs = fig.subplots(len(va_list), len(ha_list), sharex=True, sharey=True, subplot_kw=dict(aspect=1), gridspec_kw=dict(hspace=0, wspace=0))
    for (ha, ax) in zip(ha_list, axs[-1, :]):
        ax.set_xlabel(ha)
    for (va, ax) in zip(va_list, axs[:, 0]):
        ax.set_ylabel(va)
    axs[0, 1].set_title(f"rotation_mode='{mode}'", size='large')
    kw = {} if mode == 'default' else {'bbox': dict(boxstyle='square,pad=0.', ec='none', fc='C1', alpha=0.3)}
    texts = {}
    for (i, va) in enumerate(va_list):
        for (j, ha) in enumerate(ha_list):
            ax = axs[i, j]
            ax.set(xticks=[], yticks=[])
            ax.axvline(0.5, color='skyblue', zorder=0)
            ax.axhline(0.5, color='skyblue', zorder=0)
            ax.plot(0.5, 0.5, color='C0', marker='o', zorder=1)
            tx = ax.text(0.5, 0.5, 'Tpg', size='x-large', rotation=40, horizontalalignment=ha, verticalalignment=va, rotation_mode=mode, **kw)
            texts[ax] = tx
    if mode == 'default':
        fig.canvas.draw()
        for (ax, text) in texts.items():
            bb = text.get_window_extent().transformed(ax.transData.inverted())
            rect = plt.Rectangle((bb.x0, bb.y0), bb.width, bb.height, facecolor='C1', alpha=0.3, zorder=2)
            ax.add_patch(rect)
fig = plt.figure(figsize=(8, 5))
subfigs = fig.subfigures(1, 2)
test_rotation_mode(subfigs[0], 'default')
test_rotation_mode(subfigs[1], 'anchor')
plt.show()