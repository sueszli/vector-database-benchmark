"""Networks for GAN compression example using TFGAN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from slim.nets import dcgan
from slim.nets import pix2pix

def _last_conv_layer(end_points):
    if False:
        i = 10
        return i + 15
    '"Returns the last convolutional layer from an endpoints dictionary.'
    conv_list = [k if k[:4] == 'conv' else None for k in end_points.keys()]
    conv_list.sort()
    return end_points[conv_list[-1]]

def _encoder(img_batch, is_training=True, bits=64, depth=64):
    if False:
        i = 10
        return i + 15
    'Maps images to internal representation.\n\n  Args:\n    img_batch: Stuff\n    is_training: Stuff\n    bits: Number of bits per patch.\n    depth: Stuff\n\n  Returns:\n    Real-valued 2D Tensor of size [batch_size, bits].\n  '
    (_, end_points) = dcgan.discriminator(img_batch, depth=depth, is_training=is_training, scope='Encoder')
    net = _last_conv_layer(end_points)
    with tf.variable_scope('EncoderTransformer'):
        encoded = tf.contrib.layers.conv2d(net, bits, kernel_size=1, stride=1, padding='VALID', normalizer_fn=None, activation_fn=None)
    encoded = tf.squeeze(encoded, [1, 2])
    encoded.shape.assert_has_rank(2)
    return tf.nn.softsign(encoded)

def _binarizer(prebinary_codes, is_training):
    if False:
        return 10
    "Binarize compression logits.\n\n  During training, add noise, as in https://arxiv.org/pdf/1611.01704.pdf. During\n  eval, map [-1, 1] -> {-1, 1}.\n\n  Args:\n    prebinary_codes: Floating-point tensors corresponding to pre-binary codes.\n      Shape is [batch, code_length].\n    is_training: A python bool. If True, add noise. If false, binarize.\n\n  Returns:\n    Binarized codes. Shape is [batch, code_length].\n\n  Raises:\n    ValueError: If the shape of `prebinary_codes` isn't static.\n  "
    if is_training:
        noise = tf.random_uniform(prebinary_codes.shape, minval=-1.0, maxval=1.0)
        return prebinary_codes + noise
    else:
        return tf.sign(prebinary_codes)

def _decoder(codes, final_size, is_training, depth=64):
    if False:
        return 10
    'Compression decoder.'
    (decoded_img, _) = dcgan.generator(codes, depth=depth, final_size=final_size, num_outputs=3, is_training=is_training, scope='Decoder')
    return tf.nn.softsign(decoded_img)

def _validate_image_inputs(image_batch):
    if False:
        return 10
    image_batch.shape.assert_has_rank(4)
    image_batch.shape[1:].assert_is_fully_defined()

def compression_model(image_batch, num_bits=64, depth=64, is_training=True):
    if False:
        i = 10
        return i + 15
    'Image compression model.\n\n  Args:\n    image_batch: A batch of images to compress and reconstruct. Images should\n      be normalized already. Shape is [batch, height, width, channels].\n    num_bits: Desired number of bits per image in the compressed representation.\n    depth: The base number of filters for the encoder and decoder networks.\n    is_training: A python bool. If False, run in evaluation mode.\n\n  Returns:\n    uncompressed images, binary codes, prebinary codes\n  '
    image_batch = tf.convert_to_tensor(image_batch)
    _validate_image_inputs(image_batch)
    final_size = image_batch.shape.as_list()[1]
    prebinary_codes = _encoder(image_batch, is_training, num_bits, depth)
    binary_codes = _binarizer(prebinary_codes, is_training)
    uncompressed_imgs = _decoder(binary_codes, final_size, is_training, depth)
    return (uncompressed_imgs, binary_codes, prebinary_codes)

def discriminator(image_batch, unused_conditioning=None, depth=64):
    if False:
        for i in range(10):
            print('nop')
    'A thin wrapper around the pix2pix discriminator to conform to TFGAN API.'
    (logits, _) = pix2pix.pix2pix_discriminator(image_batch, num_filters=[depth, 2 * depth, 4 * depth, 8 * depth])
    return tf.layers.flatten(logits)