"""
=========================================
Adapting gray-scale filters to RGB images
=========================================

There are many filters that are designed to work with gray-scale images but not
with color images. To simplify the process of creating functions that can adapt
to RGB images, scikit-image provides the ``adapt_rgb`` decorator.

To actually use the ``adapt_rgb`` decorator, you have to decide how you want to
adapt the RGB image for use with the gray-scale filter. There are two
pre-defined handlers:

``each_channel``
    Pass each of the RGB channels to the filter one-by-one, and stitch the
    results back into an RGB image.
``hsv_value``
    Convert the RGB image to HSV and pass the value channel to the filter.
    The filtered result is inserted back into the HSV image and converted
    back to RGB.

Below, we demonstrate the use of ``adapt_rgb`` on a couple of gray-scale
filters:
"""
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters

@adapt_rgb(each_channel)
def sobel_each(image):
    if False:
        for i in range(10):
            print('nop')
    return filters.sobel(image)

@adapt_rgb(hsv_value)
def sobel_hsv(image):
    if False:
        return 10
    return filters.sobel(image)
from skimage import data
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
image = data.astronaut()
(fig, (ax_each, ax_hsv)) = plt.subplots(ncols=2, figsize=(14, 7))
ax_each.imshow(rescale_intensity(1 - sobel_each(image)))
(ax_each.set_xticks([]), ax_each.set_yticks([]))
ax_each.set_title('Sobel filter computed\n on individual RGB channels')
ax_hsv.imshow(rescale_intensity(1 - sobel_hsv(image)))
(ax_hsv.set_xticks([]), ax_hsv.set_yticks([]))
ax_hsv.set_title('Sobel filter computed\n on (V)alue converted image (HSV)')
from skimage.color import rgb2gray

def as_gray(image_filter, image, *args, **kwargs):
    if False:
        return 10
    gray_image = rgb2gray(image)
    return image_filter(gray_image, *args, **kwargs)

@adapt_rgb(as_gray)
def sobel_gray(image):
    if False:
        for i in range(10):
            print('nop')
    return filters.sobel(image)
(fig, ax) = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
ax.imshow(rescale_intensity(1 - sobel_gray(image)), cmap=plt.cm.gray)
(ax.set_xticks([]), ax.set_yticks([]))
ax.set_title('Sobel filter computed\n on the converted grayscale image')
plt.show()