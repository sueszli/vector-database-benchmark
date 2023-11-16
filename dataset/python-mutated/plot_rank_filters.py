"""
============
Rank filters
============

Rank filters are non-linear filters using local gray-level ordering to
compute the filtered value. This ensemble of filters share a common base:
the local gray-level histogram is computed on the neighborhood of a pixel
(defined by a 2D structuring element). If the filtered value is taken as the
middle value of the histogram, we get the classical median filter.

Rank filters can be used for several purposes, such as:

* image quality enhancement,
  e.g., image smoothing, sharpening

* image pre-processing,
  e.g., noise reduction, contrast enhancement

* feature extraction,
  e.g., border detection, isolated point detection

* image post-processing,
  e.g., small object removal, object grouping, contour smoothing

Some well-known filters (e.g., morphological dilation and morphological
erosion) are specific cases of rank filters [1]_.

In this example, we will see how to filter a gray-level image using some of the
linear and non-linear filters available in skimage. We use the ``camera`` image
from `skimage.data` for all comparisons.

.. [1] Pierre Soille, On morphological operators based on rank filters, Pattern
       Recognition 35 (2002) 527-535, :DOI:`10.1016/S0031-3203(01)00047-4`
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage import data
from skimage.exposure import histogram
noisy_image = img_as_ubyte(data.camera())
(hist, hist_centers) = histogram(noisy_image)
(fig, ax) = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[1].plot(hist_centers, hist, lw=2)
ax[1].set_title('Gray-level histogram')
plt.tight_layout()
from skimage.filters.rank import median
from skimage.morphology import disk, ball
rng = np.random.default_rng()
noise = rng.random(noisy_image.shape)
noisy_image = img_as_ubyte(data.camera())
noisy_image[noise > 0.99] = 255
noisy_image[noise < 0.01] = 0
(fig, axes) = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(noisy_image, vmin=0, vmax=255, cmap=plt.cm.gray)
ax[0].set_title('Noisy image')
ax[1].imshow(median(noisy_image, disk(1)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax[1].set_title('Median $r=1$')
ax[2].imshow(median(noisy_image, disk(5)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax[2].set_title('Median $r=5$')
ax[3].imshow(median(noisy_image, disk(20)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax[3].set_title('Median $r=20$')
for a in ax:
    a.axis('off')
plt.tight_layout()
from skimage.filters.rank import mean
loc_mean = mean(noisy_image, disk(10))
(fig, ax) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
ax[0].imshow(noisy_image, vmin=0, vmax=255, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(loc_mean, vmin=0, vmax=255, cmap=plt.cm.gray)
ax[1].set_title('Local mean $r=10$')
for a in ax:
    a.axis('off')
plt.tight_layout()
from skimage.filters.rank import mean_bilateral
noisy_image = img_as_ubyte(data.camera())
bilat = mean_bilateral(noisy_image.astype(np.uint16), disk(20), s0=10, s1=10)
(fig, axes) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex='row', sharey='row')
ax = axes.ravel()
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(bilat, cmap=plt.cm.gray)
ax[1].set_title('Bilateral mean')
ax[2].imshow(noisy_image[100:250, 350:450], cmap=plt.cm.gray)
ax[3].imshow(bilat[100:250, 350:450], cmap=plt.cm.gray)
for a in ax:
    a.axis('off')
plt.tight_layout()
from skimage import exposure
from skimage.filters import rank
noisy_image = img_as_ubyte(data.camera())
glob = exposure.equalize_hist(noisy_image) * 255
loc = rank.equalize(noisy_image, disk(20))
hist = np.histogram(noisy_image, bins=np.arange(0, 256))
glob_hist = np.histogram(glob, bins=np.arange(0, 256))
loc_hist = np.histogram(loc, bins=np.arange(0, 256))
(fig, axes) = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
ax = axes.ravel()
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[1].plot(hist[1][:-1], hist[0], lw=2)
ax[1].set_title('Histogram of gray values')
ax[2].imshow(glob, cmap=plt.cm.gray)
ax[2].axis('off')
ax[3].plot(glob_hist[1][:-1], glob_hist[0], lw=2)
ax[3].set_title('Histogram of gray values')
ax[4].imshow(loc, cmap=plt.cm.gray)
ax[4].axis('off')
ax[5].plot(loc_hist[1][:-1], loc_hist[0], lw=2)
ax[5].set_title('Histogram of gray values')
plt.tight_layout()
from skimage.filters.rank import autolevel
noisy_image = img_as_ubyte(data.camera())
auto = autolevel(noisy_image.astype(np.uint16), disk(20))
(fig, ax) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(auto, cmap=plt.cm.gray)
ax[1].set_title('Local autolevel')
for a in ax:
    a.axis('off')
plt.tight_layout()
from skimage.filters.rank import autolevel_percentile
image = data.camera()
footprint = disk(20)
loc_autolevel = autolevel(image, footprint=footprint)
loc_perc_autolevel0 = autolevel_percentile(image, footprint=footprint, p0=0.01, p1=0.99)
loc_perc_autolevel1 = autolevel_percentile(image, footprint=footprint, p0=0.05, p1=0.95)
loc_perc_autolevel2 = autolevel_percentile(image, footprint=footprint, p0=0.1, p1=0.9)
loc_perc_autolevel3 = autolevel_percentile(image, footprint=footprint, p0=0.15, p1=0.85)
(fig, axes) = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
ax = axes.ravel()
title_list = ['Original', 'auto_level', 'auto-level 1%', 'auto-level 5%', 'auto-level 10%', 'auto-level 15%']
image_list = [image, loc_autolevel, loc_perc_autolevel0, loc_perc_autolevel1, loc_perc_autolevel2, loc_perc_autolevel3]
for i in range(0, len(image_list)):
    ax[i].imshow(image_list[i], cmap=plt.cm.gray, vmin=0, vmax=255)
    ax[i].set_title(title_list[i])
    ax[i].axis('off')
plt.tight_layout()
from skimage.filters.rank import enhance_contrast
noisy_image = img_as_ubyte(data.camera())
enh = enhance_contrast(noisy_image, disk(5))
(fig, axes) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex='row', sharey='row')
ax = axes.ravel()
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(enh, cmap=plt.cm.gray)
ax[1].set_title('Local morphological contrast enhancement')
ax[2].imshow(noisy_image[100:250, 350:450], cmap=plt.cm.gray)
ax[3].imshow(enh[100:250, 350:450], cmap=plt.cm.gray)
for a in ax:
    a.axis('off')
plt.tight_layout()
from skimage.filters.rank import enhance_contrast_percentile
noisy_image = img_as_ubyte(data.camera())
penh = enhance_contrast_percentile(noisy_image, disk(5), p0=0.1, p1=0.9)
(fig, axes) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex='row', sharey='row')
ax = axes.ravel()
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(penh, cmap=plt.cm.gray)
ax[1].set_title('Local percentile morphological\n contrast enhancement')
ax[2].imshow(noisy_image[100:250, 350:450], cmap=plt.cm.gray)
ax[3].imshow(penh[100:250, 350:450], cmap=plt.cm.gray)
for a in ax:
    a.axis('off')
plt.tight_layout()
from skimage.filters.rank import otsu
from skimage.filters import threshold_otsu
from skimage import exposure
p8 = data.page()
radius = 10
footprint = disk(radius)
t_loc_otsu = otsu(p8, footprint)
loc_otsu = p8 >= t_loc_otsu
t_glob_otsu = threshold_otsu(p8)
glob_otsu = p8 >= t_glob_otsu
(fig, axes) = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex=True, sharey=True)
ax = axes.ravel()
fig.colorbar(ax[0].imshow(p8, cmap=plt.cm.gray), ax=ax[0])
ax[0].set_title('Original')
fig.colorbar(ax[1].imshow(t_loc_otsu, cmap=plt.cm.gray), ax=ax[1])
ax[1].set_title(f'Local Otsu ($r={radius}$)')
ax[2].imshow(p8 >= t_loc_otsu, cmap=plt.cm.gray)
ax[2].set_title('Original >= local Otsu')
ax[3].imshow(glob_otsu, cmap=plt.cm.gray)
ax[3].set_title(f'Global Otsu ($t={t_glob_otsu}$)')
for a in ax:
    a.axis('off')
plt.tight_layout()
brain = exposure.rescale_intensity(data.brain().astype(float))
radius = 5
neighborhood = ball(radius)
t_loc_otsu = rank.otsu(brain, neighborhood)
loc_otsu = brain >= t_loc_otsu
t_glob_otsu = threshold_otsu(brain)
glob_otsu = brain >= t_glob_otsu
(fig, axes) = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex=True, sharey=True)
ax = axes.ravel()
slice_index = 3
fig.colorbar(ax[0].imshow(brain[slice_index], cmap=plt.cm.gray), ax=ax[0])
ax[0].set_title('Original')
fig.colorbar(ax[1].imshow(t_loc_otsu[slice_index], cmap=plt.cm.gray), ax=ax[1])
ax[1].set_title(f'Local Otsu ($r={radius}$)')
ax[2].imshow(brain[slice_index] >= t_loc_otsu[slice_index], cmap=plt.cm.gray)
ax[2].set_title('Original >= local Otsu')
ax[3].imshow(glob_otsu[slice_index], cmap=plt.cm.gray)
ax[3].set_title(f'Global Otsu ($t={t_glob_otsu}$)')
for a in ax:
    a.axis('off')
fig.tight_layout()
n = 100
theta = np.linspace(0, 10 * np.pi, n)
x = np.sin(theta)
m = (np.tile(x, (n, 1)) * np.linspace(0.1, 1, n) * 128 + 128).astype(np.uint8)
radius = 10
t = rank.otsu(m, disk(radius))
(fig, ax) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
ax[0].imshow(m, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(m >= t, cmap=plt.cm.gray)
ax[1].set_title(f'Local Otsu ($r={radius}$)')
for a in ax:
    a.axis('off')
plt.tight_layout()
from skimage.filters.rank import maximum, minimum, gradient
noisy_image = img_as_ubyte(data.camera())
opening = maximum(minimum(noisy_image, disk(5)), disk(5))
closing = minimum(maximum(noisy_image, disk(5)), disk(5))
grad = gradient(noisy_image, disk(5))
(fig, axes) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(closing, cmap=plt.cm.gray)
ax[1].set_title('Gray-level closing')
ax[2].imshow(opening, cmap=plt.cm.gray)
ax[2].set_title('Gray-level opening')
ax[3].imshow(grad, cmap=plt.cm.gray)
ax[3].set_title('Morphological gradient')
for a in ax:
    a.axis('off')
plt.tight_layout()
from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
image = data.camera()
(fig, ax) = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)
fig.colorbar(ax[0].imshow(image, cmap=plt.cm.gray), ax=ax[0])
ax[0].set_title('Image')
fig.colorbar(ax[1].imshow(entropy(image, disk(5)), cmap=plt.cm.gray), ax=ax[1])
ax[1].set_title('Entropy')
for a in ax:
    a.axis('off')
plt.tight_layout()
from time import time
from scipy.ndimage import percentile_filter
from skimage.morphology import dilation
from skimage.filters.rank import median, maximum

def exec_and_timeit(func):
    if False:
        print('Hello World!')
    'Decorator that returns both function results and execution time.'

    def wrapper(*arg):
        if False:
            print('Hello World!')
        t1 = time()
        res = func(*arg)
        t2 = time()
        ms = (t2 - t1) * 1000.0
        return (res, ms)
    return wrapper

@exec_and_timeit
def cr_med(image, footprint):
    if False:
        return 10
    return median(image=image, footprint=footprint)

@exec_and_timeit
def cr_max(image, footprint):
    if False:
        print('Hello World!')
    return maximum(image=image, footprint=footprint)

@exec_and_timeit
def cm_dil(image, footprint):
    if False:
        for i in range(10):
            print('nop')
    return dilation(image=image, footprint=footprint)

@exec_and_timeit
def ndi_med(image, n):
    if False:
        while True:
            i = 10
    return percentile_filter(image, 50, size=n * 2 - 1)
a = data.camera()
rec = []
e_range = range(1, 20, 2)
for r in e_range:
    elem = disk(r + 1)
    (rc, ms_rc) = cr_max(a, elem)
    (rcm, ms_rcm) = cm_dil(a, elem)
    rec.append((ms_rc, ms_rcm))
rec = np.asarray(rec)
(fig, ax) = plt.subplots(figsize=(10, 10), sharey=True)
ax.set_title('Performance with respect to element size')
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Element radius')
ax.plot(e_range, rec)
ax.legend(['filters.rank.maximum', 'morphology.dilate'])
plt.tight_layout()
r = 9
elem = disk(r + 1)
rec = []
s_range = range(100, 1000, 100)
for s in s_range:
    a = (rng.random((s, s)) * 256).astype(np.uint8)
    (rc, ms_rc) = cr_max(a, elem)
    (rcm, ms_rcm) = cm_dil(a, elem)
    rec.append((ms_rc, ms_rcm))
rec = np.asarray(rec)
(fig, ax) = plt.subplots()
ax.set_title('Performance with respect to image size')
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Image size')
ax.plot(s_range, rec)
ax.legend(['filters.rank.maximum', 'morphology.dilate'])
plt.tight_layout()
a = data.camera()
rec = []
e_range = range(2, 30, 4)
for r in e_range:
    elem = disk(r + 1)
    (rc, ms_rc) = cr_med(a, elem)
    (rndi, ms_ndi) = ndi_med(a, r)
    rec.append((ms_rc, ms_ndi))
rec = np.asarray(rec)
(fig, ax) = plt.subplots()
ax.set_title('Performance with respect to element size')
ax.plot(e_range, rec)
ax.legend(['filters.rank.median', 'scipy.ndimage.percentile'])
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Element radius')
(fig, ax) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
ax[0].set_title('filters.rank.median')
ax[0].imshow(rc, cmap=plt.cm.gray)
ax[1].set_title('scipy.ndimage.percentile')
ax[1].imshow(rndi, cmap=plt.cm.gray)
for a in ax:
    a.axis('off')
plt.tight_layout()
r = 9
elem = disk(r + 1)
rec = []
s_range = [100, 200, 500, 1000]
for s in s_range:
    a = (rng.random((s, s)) * 256).astype(np.uint8)
    (rc, ms_rc) = cr_med(a, elem)
    (rndi, ms_ndi) = ndi_med(a, r)
    rec.append((ms_rc, ms_ndi))
rec = np.asarray(rec)
(fig, ax) = plt.subplots()
ax.set_title('Performance with respect to image size')
ax.plot(s_range, rec)
ax.legend(['filters.rank.median', 'scipy.ndimage.percentile'])
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Image size')
plt.tight_layout()
plt.show()