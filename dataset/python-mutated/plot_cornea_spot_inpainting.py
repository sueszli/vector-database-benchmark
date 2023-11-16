"""
============================================
Restore spotted cornea image with inpainting
============================================

Optical coherence tomography (OCT) is a non-invasive imaging technique used by
ophthalmologists to take pictures of the back of a patient's eye [1]_.
When performing OCT,
dust may stick to the reference mirror of the equipment, causing dark spots to
appear on the images. The problem is that these dirt spots cover areas of
in-vivo tissue, hence hiding data of interest. Our goal here is to restore
(reconstruct) the hidden areas based on the pixels near their boundaries.

This tutorial is adapted from an application shared by Jules Scholler [2]_.
The images were acquired by Viacheslav Mazlin (see
:func:`skimage.data.palisades_of_vogt`).

.. [1] David Turbert, reviewed by Ninel Z Gregori, MD (2023)
       `What Is Optical Coherence Tomography?
       <https://www.aao.org/eye-health/treatments/what-is-optical-coherence-tomography>`_,
       American Academy of Ophthalmology.
.. [2] Jules Scholler (2019) "Image denoising using inpainting"
       https://www.jscholler.com/2019-02-28-remove-dots/
"""
import matplotlib.pyplot as plt
import numpy as np
import plotly.io
import plotly.express as px
import skimage as ski
image_seq = ski.data.palisades_of_vogt()
print(f'number of dimensions: {image_seq.ndim}')
print(f'shape: {image_seq.shape}')
print(f'dtype: {image_seq.dtype}')
fig = px.imshow(image_seq[::6, :, :], animation_frame=0, binary_string=True, labels={'animation_frame': '6-step time point'}, title='Sample of in-vivo human cornea')
fig.update_layout(autosize=False, minreducedwidth=250, minreducedheight=250)
plotly.io.show(fig)
image_med = np.median(image_seq, axis=0)
image_var = np.var(image_seq, axis=0)
assert image_var.shape == image_med.shape
print(f'shape: {image_med.shape}')
(fig, ax) = plt.subplots(ncols=2, figsize=(12, 6))
ax[0].imshow(image_med, cmap='gray')
ax[0].set_title('Image median over time')
ax[1].imshow(image_var, cmap='gray')
ax[1].set_title('Image variance over time')
fig.tight_layout()
thresh_1 = ski.filters.threshold_local(image_med, block_size=21, offset=15)
thresh_2 = ski.filters.threshold_local(image_med, block_size=43, offset=15)
mask_1 = image_med < thresh_1
mask_2 = image_med < thresh_2

def plot_comparison(plot1, plot2, title1, title2):
    if False:
        while True:
            i = 10
    (fig, (ax1, ax2)) = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)
    ax1.imshow(plot1, cmap='gray')
    ax1.set_title(title1)
    ax2.imshow(plot2, cmap='gray')
    ax2.set_title(title2)
plot_comparison(mask_1, mask_2, 'block_size = 21', 'block_size = 43')
thresh_0 = ski.filters.threshold_local(image_med, block_size=43)
mask_0 = image_med < thresh_0
plot_comparison(mask_0, mask_2, 'No offset', 'offset = 15')
footprint = ski.morphology.diamond(3)
mask_open = ski.morphology.opening(mask_2, footprint)
plot_comparison(mask_2, mask_open, 'mask before', 'after opening')
mask_dilate = ski.morphology.dilation(mask_open, footprint)
plot_comparison(mask_open, mask_dilate, 'before', 'after dilation')
image_seq_inpainted = np.zeros(image_seq.shape)
for i in range(image_seq.shape[0]):
    image_seq_inpainted[i] = ski.restoration.inpaint_biharmonic(image_seq[i], mask_dilate)
contours = ski.measure.find_contours(mask_dilate)
x = []
y = []
for contour in contours:
    x.append(contour[:, 0])
    y.append(contour[:, 1])
x_flat = np.concatenate(x).ravel().round().astype(int)
y_flat = np.concatenate(y).ravel().round().astype(int)
contour_mask = np.zeros(mask_dilate.shape, dtype=bool)
contour_mask[x_flat, y_flat] = 1
sample_result = image_seq_inpainted[12]
sample_result /= sample_result.max()
color_contours = ski.color.label2rgb(contour_mask, image=sample_result, alpha=0.4, bg_color=(1, 1, 1))
(fig, ax) = plt.subplots(figsize=(6, 6))
ax.imshow(color_contours)
ax.set_title('Segmented spots over restored image')
fig.tight_layout()
plt.show()