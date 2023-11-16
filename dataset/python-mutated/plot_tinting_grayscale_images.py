"""
=========================
Tinting gray-scale images
=========================

It can be useful to artificially tint an image with some color, either to
highlight particular regions of an image or maybe just to liven up a grayscale
image. This example demonstrates image-tinting by scaling RGB values and by
adjusting colors in the HSV color-space.

In 2D, color images are often represented in RGB---3 layers of 2D arrays, where
the 3 layers represent (R)ed, (G)reen and (B)lue channels of the image. The
simplest way of getting a tinted image is to set each RGB channel to the
grayscale image scaled by a different multiplier for each channel. For example,
multiplying the green and blue channels by 0 leaves only the red channel and
produces a bright red image. Similarly, zeroing-out the blue channel leaves
only the red and green channels, which combine to form yellow.
"""
import matplotlib.pyplot as plt
from skimage import data
from skimage import color
from skimage import img_as_float, img_as_ubyte
grayscale_image = img_as_float(data.camera()[::2, ::2])
image = color.gray2rgb(grayscale_image)
red_multiplier = [1, 0, 0]
yellow_multiplier = [1, 1, 0]
(fig, (ax1, ax2)) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
ax1.imshow(red_multiplier * image)
ax2.imshow(yellow_multiplier * image)
plt.show()
import numpy as np
hue_gradient = np.linspace(0, 1)
hsv = np.ones(shape=(1, len(hue_gradient), 3), dtype=float)
hsv[:, :, 0] = hue_gradient
all_hues = color.hsv2rgb(hsv)
(fig, ax) = plt.subplots(figsize=(5, 2))
ax.imshow(all_hues, extent=(0 - 0.5 / len(hue_gradient), 1 + 0.5 / len(hue_gradient), 0, 0.2))
ax.set_axis_off()

def colorize(image, hue, saturation=1):
    if False:
        print('Hello World!')
    'Add color of the given hue to an RGB image.\n\n    By default, set the saturation to 1 so that the colors pop!\n    '
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return color.hsv2rgb(hsv)
hue_rotations = np.linspace(0, 1, 6)
(fig, axes) = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
for (ax, hue) in zip(axes.flat, hue_rotations):
    tinted_image = colorize(image, hue, saturation=0.3)
    ax.imshow(tinted_image, vmin=0, vmax=1)
    ax.set_axis_off()
fig.tight_layout()
from skimage.filters import rank
top_left = (slice(25),) * 2
bottom_right = (slice(-25, None),) * 2
sliced_image = image.copy()
sliced_image[top_left] = colorize(image[top_left], 0.82, saturation=0.5)
sliced_image[bottom_right] = colorize(image[bottom_right], 0.5, saturation=0.5)
noisy = rank.entropy(img_as_ubyte(grayscale_image), np.ones((9, 9)))
textured_regions = noisy > 4.25
masked_image = image.copy()
masked_image[textured_regions, :] *= red_multiplier
(fig, (ax1, ax2)) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True)
ax1.imshow(sliced_image)
ax2.imshow(masked_image)
plt.show()