"""
=====================================
Resizing axes with constrained layout
=====================================

*Constrained layout* attempts to resize subplots in
a figure so that there are no overlaps between axes objects and labels
on the axes.

See :ref:`constrainedlayout_guide` for more details and
:ref:`tight_layout_guide` for an alternative.

"""
import matplotlib.pyplot as plt

def example_plot(ax):
    if False:
        for i in range(10):
            print('nop')
    ax.plot([1, 2])
    ax.set_xlabel('x-label', fontsize=12)
    ax.set_ylabel('y-label', fontsize=12)
    ax.set_title('Title', fontsize=14)
(fig, axs) = plt.subplots(nrows=2, ncols=2, layout=None)
for ax in axs.flat:
    example_plot(ax)
(fig, axs) = plt.subplots(nrows=2, ncols=2, layout='constrained')
for ax in axs.flat:
    example_plot(ax)
fig = plt.figure(layout='constrained')
import matplotlib.gridspec as gridspec
gs0 = gridspec.GridSpec(1, 2, figure=fig)
gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0])
for n in range(3):
    ax = fig.add_subplot(gs1[n])
    example_plot(ax)
gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1])
for n in range(2):
    ax = fig.add_subplot(gs2[n])
    example_plot(ax)
plt.show()