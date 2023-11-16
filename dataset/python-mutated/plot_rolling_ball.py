"""
================================================================
Use rolling-ball algorithm for estimating background intensity
================================================================

The rolling-ball algorithm estimates the background intensity of a grayscale
image in case of uneven exposure. It is frequently used in biomedical
image processing and was first proposed by Stanley R. Sternberg in
1983 [1]_.

The algorithm works as a filter and is quite intuitive. We think of the image
as a surface that has unit-sized blocks stacked on top of each other in place
of each pixel. The number of blocks, and hence surface height, is determined
by the intensity of the pixel. To get the intensity of the background at a
desired (pixel) position, we imagine submerging a ball under the surface at the
desired position. Once it is completely covered by the blocks, the apex of
the ball determines the intensity of the background at that position. We can
then *roll* this ball around below the surface to get the background values for
the entire image.

Scikit-image implements a general version of this rolling-ball algorithm, which
allows you to not just use balls, but arbitrary shapes as kernel and works on
n-dimensional ndimages. This allows you to directly filter RGB images or filter
image stacks along any (or all) spacial dimensions.

.. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1 (1983):
    22-34. :DOI:`10.1109/MC.1983.1654163`


Classic rolling ball
-------------------------------

In scikit-image, the rolling ball algorithm assumes that your background has
low intensity (black), whereas the features have high intensity (white). If
this is the case for your image, you can directly use the filter like so:

"""
import matplotlib.pyplot as plt
import numpy as np
import pywt
from skimage import data, restoration, util

def plot_result(image, background):
    if False:
        while True:
            i = 10
    (fig, ax) = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')
    ax[1].imshow(background, cmap='gray')
    ax[1].set_title('Background')
    ax[1].axis('off')
    ax[2].imshow(image - background, cmap='gray')
    ax[2].set_title('Result')
    ax[2].axis('off')
    fig.tight_layout()
image = data.coins()
background = restoration.rolling_ball(image)
plot_result(image, background)
plt.show()
image = data.page()
image_inverted = util.invert(image)
background_inverted = restoration.rolling_ball(image_inverted, radius=45)
filtered_image_inverted = image_inverted - background_inverted
filtered_image = util.invert(filtered_image_inverted)
background = util.invert(background_inverted)
(fig, ax) = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')
ax[0].axis('off')
ax[1].imshow(background, cmap='gray')
ax[1].set_title('Background')
ax[1].axis('off')
ax[2].imshow(filtered_image, cmap='gray')
ax[2].set_title('Result')
ax[2].axis('off')
fig.tight_layout()
plt.show()
image = data.page()
image_inverted = util.invert(image)
background_inverted = restoration.rolling_ball(image_inverted, radius=45)
background = util.invert(background_inverted)
underflow_image = image - background
correct_image = util.invert(image_inverted - background_inverted)
(fig, ax) = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(underflow_image, cmap='gray')
ax[0].set_title('Background Removal with Underflow')
ax[0].axis('off')
ax[1].imshow(correct_image, cmap='gray')
ax[1].set_title('Correct Background Removal')
ax[1].axis('off')
fig.tight_layout()
plt.show()
image = data.coins()[:200, :200].astype(np.uint16)
background = restoration.rolling_ball(image, radius=70.5)
plot_result(image, background)
plt.show()
image = util.img_as_float(data.coins()[:200, :200])
background = restoration.rolling_ball(image, radius=70.5)
plot_result(image, background)
plt.show()
normalized_radius = 70.5 / 255
image = util.img_as_float(data.coins())
kernel = restoration.ellipsoid_kernel((70.5 * 2, 70.5 * 2), normalized_radius * 2)
background = restoration.rolling_ball(image, kernel=kernel)
plot_result(image, background)
plt.show()
image = data.coins()
kernel = restoration.ellipsoid_kernel((70.5 * 2, 70.5 * 2), 70.5 * 2)
background = restoration.rolling_ball(image, kernel=kernel)
plot_result(image, background)
plt.show()
image = data.coins()
kernel = restoration.ellipsoid_kernel((10 * 2, 10 * 2), 255 * 2)
background = restoration.rolling_ball(image, kernel=kernel)
plot_result(image, background)
plt.show()
image = data.cells3d()[:, 1, ...]
background = restoration.rolling_ball(image, kernel=restoration.ellipsoid_kernel((1, 21, 21), 0.1))
plot_result(image[30, ...], background[30, ...])
plt.show()
image = data.cells3d()[:, 1, ...]
background = restoration.rolling_ball(image, kernel=restoration.ellipsoid_kernel((5, 21, 21), 0.1))
plot_result(image[30, ...], background[30, ...])
plt.show()
image = data.cells3d()[:, 1, ...]
background = restoration.rolling_ball(image, kernel=restoration.ellipsoid_kernel((100, 1, 1), 0.1))
plot_result(image[30, ...], background[30, ...])
plt.show()
x = pywt.data.ecg()
background = restoration.rolling_ball(x, radius=80)
background2 = restoration.rolling_ball(x, radius=10)
plt.figure()
plt.plot(x, label='original')
plt.plot(x - background, label='radius=80')
plt.plot(x - background2, label='radius=10')
plt.legend()
plt.show()