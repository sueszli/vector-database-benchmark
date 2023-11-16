"""
=======================
Morphological Filtering
=======================

Morphological image processing is a collection of non-linear operations related
to the shape or morphology of features in an image, such as boundaries,
skeletons, etc. In any given technique, we probe an image with a small shape or
template called a structuring element, which defines the region of interest or
neighborhood around a pixel.

In this document we outline the following basic morphological operations:

1. Erosion
2. Dilation
3. Opening
4. Closing
5. White Tophat
6. Black Tophat
7. Skeletonize
8. Convex Hull


To get started, let's load an image using ``io.imread``. Note that morphology
functions only work on gray-scale or binary images, so we set ``as_gray=True``.
"""
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_ubyte
orig_phantom = img_as_ubyte(data.shepp_logan_phantom())
(fig, ax) = plt.subplots()
ax.imshow(orig_phantom, cmap=plt.cm.gray)

def plot_comparison(original, filtered, filter_name):
    if False:
        i = 10
        return i + 15
    (fig, (ax1, ax2)) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
footprint = disk(6)
eroded = erosion(orig_phantom, footprint)
plot_comparison(orig_phantom, eroded, 'erosion')
dilated = dilation(orig_phantom, footprint)
plot_comparison(orig_phantom, dilated, 'dilation')
opened = opening(orig_phantom, footprint)
plot_comparison(orig_phantom, opened, 'opening')
phantom = orig_phantom.copy()
phantom[10:30, 200:210] = 0
closed = closing(phantom, footprint)
plot_comparison(phantom, closed, 'closing')
phantom = orig_phantom.copy()
phantom[340:350, 200:210] = 255
phantom[100:110, 200:210] = 0
w_tophat = white_tophat(phantom, footprint)
plot_comparison(phantom, w_tophat, 'white tophat')
b_tophat = black_tophat(phantom, footprint)
plot_comparison(phantom, b_tophat, 'black tophat')
horse = data.horse()
sk = skeletonize(horse == 0)
plot_comparison(horse, sk, 'skeletonize')
hull1 = convex_hull_image(horse == 0)
plot_comparison(horse, hull1, 'convex hull')
horse_mask = horse == 0
horse_mask[45:50, 75:80] = 1
hull2 = convex_hull_image(horse_mask)
plot_comparison(horse_mask, hull2, 'convex hull')
plt.show()