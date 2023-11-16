"""
===================
Butterworth Filters
===================

The Butterworth filter is implemented in the frequency domain and is designed
to have no passband or stopband ripple. It can be used in either a lowpass or
highpass variant. The ``cutoff_frequency_ratio`` parameter is used to set the
cutoff frequency as a fraction of the sampling frequency. Given that the
Nyquist frequency is half the sampling frequency, this means that this
parameter should be a positive floating point value < 0.5. The ``order`` of the
filter can be adjusted to control the transition width, with higher values
leading to a sharper transition between the passband and stopband.

"""
import matplotlib.pyplot as plt
from skimage import data, filters
image = data.camera()
cutoffs = [0.02, 0.08, 0.16]

def get_filtered(image, cutoffs, squared_butterworth=True, order=3.0, npad=0):
    if False:
        while True:
            i = 10
    'Lowpass and highpass butterworth filtering at all specified cutoffs.\n\n    Parameters\n    ----------\n    image : ndarray\n        The image to be filtered.\n    cutoffs : sequence of int\n        Both lowpass and highpass filtering will be performed for each cutoff\n        frequency in `cutoffs`.\n    squared_butterworth : bool, optional\n        Whether the traditional Butterworth filter or its square is used.\n    order : float, optional\n        The order of the Butterworth filter\n\n    Returns\n    -------\n    lowpass_filtered : list of ndarray\n        List of images lowpass filtered at the frequencies in `cutoffs`.\n    highpass_filtered : list of ndarray\n        List of images highpass filtered at the frequencies in `cutoffs`.\n    '
    lowpass_filtered = []
    highpass_filtered = []
    for cutoff in cutoffs:
        lowpass_filtered.append(filters.butterworth(image, cutoff_frequency_ratio=cutoff, order=order, high_pass=False, squared_butterworth=squared_butterworth, npad=npad))
        highpass_filtered.append(filters.butterworth(image, cutoff_frequency_ratio=cutoff, order=order, high_pass=True, squared_butterworth=squared_butterworth, npad=npad))
    return (lowpass_filtered, highpass_filtered)

def plot_filtered(lowpass_filtered, highpass_filtered, cutoffs):
    if False:
        for i in range(10):
            print('nop')
    'Generate plots for paired lists of lowpass and highpass images.'
    (fig, axes) = plt.subplots(2, 1 + len(cutoffs), figsize=(12, 8))
    fontdict = dict(fontsize=14, fontweight='bold')
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('original', fontdict=fontdict)
    axes[1, 0].set_axis_off()
    for (i, c) in enumerate(cutoffs):
        axes[0, i + 1].imshow(lowpass_filtered[i], cmap='gray')
        axes[0, i + 1].set_title(f'lowpass, c={c}', fontdict=fontdict)
        axes[1, i + 1].imshow(highpass_filtered[i], cmap='gray')
        axes[1, i + 1].set_title(f'highpass, c={c}', fontdict=fontdict)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return (fig, axes)
(lowpasses, highpasses) = get_filtered(image, cutoffs, squared_butterworth=True)
(fig, axes) = plot_filtered(lowpasses, highpasses, cutoffs)
titledict = dict(fontsize=18, fontweight='bold')
fig.text(0.5, 0.95, '(squared) Butterworth filtering (order=3.0, npad=0)', fontdict=titledict, horizontalalignment='center')
(lowpasses, highpasses) = get_filtered(image, cutoffs, squared_butterworth=True, npad=32)
(fig, axes) = plot_filtered(lowpasses, highpasses, cutoffs)
fig.text(0.5, 0.95, '(squared) Butterworth filtering (order=3.0, npad=32)', fontdict=titledict, horizontalalignment='center')
(lowpasses, highpasses) = get_filtered(image, cutoffs, squared_butterworth=False, npad=32)
(fig, axes) = plot_filtered(lowpasses, highpasses, cutoffs)
fig.text(0.5, 0.95, 'Butterworth filtering (order=3.0, npad=32)', fontdict=titledict, horizontalalignment='center')
plt.show()