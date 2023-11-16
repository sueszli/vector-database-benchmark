"""Python implementation of MS-SSIM.

Usage:

python msssim.py --original_image=original.png --compared_image=distorted.png
"""
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow as tf
tf.flags.DEFINE_string('original_image', None, 'Path to PNG image.')
tf.flags.DEFINE_string('compared_image', None, 'Path to PNG image.')
FLAGS = tf.flags.FLAGS

def _FSpecialGauss(size, sigma):
    if False:
        i = 10
        return i + 15
    "Function to mimic the 'fspecial' gaussian MATLAB function."
    radius = size // 2
    offset = 0.0
    (start, stop) = (-radius, radius + 1)
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    (x, y) = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    if False:
        i = 10
        return i + 15
    "Return the Structural Similarity Map between `img1` and `img2`.\n\n  This function attempts to match the functionality of ssim_index_new.m by\n  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip\n\n  Arguments:\n    img1: Numpy array holding the first RGB image batch.\n    img2: Numpy array holding the second RGB image batch.\n    max_val: the dynamic range of the images (i.e., the difference between the\n      maximum the and minimum allowed values).\n    filter_size: Size of blur kernel to use (will be reduced for small images).\n    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced\n      for small images).\n    k1: Constant used to maintain stability in the SSIM calculation (0.01 in\n      the original paper).\n    k2: Constant used to maintain stability in the SSIM calculation (0.03 in\n      the original paper).\n\n  Returns:\n    Pair containing the mean SSIM and contrast sensitivity between `img1` and\n    `img2`.\n\n  Raises:\n    RuntimeError: If input images don't have the same shape or don't have four\n      dimensions: [batch_size, height, width, depth].\n  "
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).', img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d', img1.ndim)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    (_, height, width, _) = img1.shape
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        (mu1, mu2) = (img1, img2)
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2
    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((2.0 * mu12 + c1) * v1 / ((mu11 + mu22 + c1) * v2))
    cs = np.mean(v1 / v2)
    return (ssim, cs)

def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, weights=None):
    if False:
        for i in range(10):
            print('nop')
    'Return the MS-SSIM score between `img1` and `img2`.\n\n  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image\n  Quality Assessment according to Zhou Wang\'s paper, "Multi-scale structural\n  similarity for image quality assessment" (2003).\n  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf\n\n  Author\'s MATLAB implementation:\n  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip\n\n  Arguments:\n    img1: Numpy array holding the first RGB image batch.\n    img2: Numpy array holding the second RGB image batch.\n    max_val: the dynamic range of the images (i.e., the difference between the\n      maximum the and minimum allowed values).\n    filter_size: Size of blur kernel to use (will be reduced for small images).\n    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced\n      for small images).\n    k1: Constant used to maintain stability in the SSIM calculation (0.01 in\n      the original paper).\n    k2: Constant used to maintain stability in the SSIM calculation (0.03 in\n      the original paper).\n    weights: List of weights for each level; if none, use five levels and the\n      weights from the original paper.\n\n  Returns:\n    MS-SSIM score between `img1` and `img2`.\n\n  Raises:\n    RuntimeError: If input images don\'t have the same shape or don\'t have four\n      dimensions: [batch_size, height, width, depth].\n  '
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).', img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d', img1.ndim)
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    (im1, im2) = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        (ssim, cs) = _SSIMForMultiScale(im1, im2, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [convolve(im, downsample_filter, mode='reflect') for im in [im1, im2]]
        (im1, im2) = [x[:, ::2, ::2, :] for x in filtered]
    return np.prod(mcs[0:levels - 1] ** weights[0:levels - 1]) * mssim[levels - 1] ** weights[levels - 1]

def main(_):
    if False:
        print('Hello World!')
    if FLAGS.original_image is None or FLAGS.compared_image is None:
        print('\nUsage: python msssim.py --original_image=original.png --compared_image=distorted.png\n\n')
        return
    if not tf.gfile.Exists(FLAGS.original_image):
        print('\nCannot find --original_image.\n')
        return
    if not tf.gfile.Exists(FLAGS.compared_image):
        print('\nCannot find --compared_image.\n')
        return
    with tf.gfile.FastGFile(FLAGS.original_image) as image_file:
        img1_str = image_file.read('rb')
    with tf.gfile.FastGFile(FLAGS.compared_image) as image_file:
        img2_str = image_file.read('rb')
    input_img = tf.placeholder(tf.string)
    decoded_image = tf.expand_dims(tf.image.decode_png(input_img, channels=3), 0)
    with tf.Session() as sess:
        img1 = sess.run(decoded_image, feed_dict={input_img: img1_str})
        img2 = sess.run(decoded_image, feed_dict={input_img: img2_str})
    print(MultiScaleSSIM(img1, img2, max_val=255))
if __name__ == '__main__':
    tf.app.run()