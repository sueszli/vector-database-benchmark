"""
==================================
Color Quantization using K-Means
==================================

Performs a pixel-wise Vector Quantization (VQ) of an image of the summer palace
(China), reducing the number of colors required to show the image from 96,615
unique colors to 64, while preserving the overall appearance quality.

In this example, pixels are represented in a 3D-space and K-means is used to
find 64 color clusters. In the image processing literature, the codebook
obtained from K-means (the cluster centers) is called the color palette. Using
a single byte, up to 256 colors can be addressed, whereas an RGB encoding
requires 3 bytes per pixel. The GIF file format, for example, uses such a
palette.

For comparison, a quantized image using a random codebook (colors picked up
randomly) is also shown.

"""
from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
n_colors = 64
china = load_sample_image('china.jpg')
china = np.array(china, dtype=np.float64) / 255
(w, h, d) = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))
print('Fitting model on a small sub-sample of the data')
t0 = time()
image_array_sample = shuffle(image_array, random_state=0, n_samples=1000)
kmeans = KMeans(n_clusters=n_colors, n_init='auto', random_state=0).fit(image_array_sample)
print(f'done in {time() - t0:0.3f}s.')
print('Predicting color indices on the full image (k-means)')
t0 = time()
labels = kmeans.predict(image_array)
print(f'done in {time() - t0:0.3f}s.')
codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
print('Predicting color indices on the full image (random)')
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
print(f'done in {time() - t0:0.3f}s.')

def recreate_image(codebook, labels, w, h):
    if False:
        i = 10
        return i + 15
    'Recreate the (compressed) image from the code book & labels'
    return codebook[labels].reshape(w, h, -1)
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)
plt.figure(2)
plt.clf()
plt.axis('off')
plt.title(f'Quantized image ({n_colors} colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.figure(3)
plt.clf()
plt.axis('off')
plt.title(f'Quantized image ({n_colors} colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()