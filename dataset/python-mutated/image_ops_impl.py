"""Implementation of image ops."""
import functools
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
ops.NotDifferentiable('RandomCrop')
ops.NotDifferentiable('HSVToRGB')
ops.NotDifferentiable('DrawBoundingBoxes')
ops.NotDifferentiable('SampleDistortedBoundingBox')
ops.NotDifferentiable('SampleDistortedBoundingBoxV2')
ops.NotDifferentiable('ExtractGlimpse')
ops.NotDifferentiable('NonMaxSuppression')
ops.NotDifferentiable('NonMaxSuppressionV2')
ops.NotDifferentiable('NonMaxSuppressionWithOverlaps')
ops.NotDifferentiable('GenerateBoundingBoxProposals')

def _assert(cond, ex_type, msg):
    if False:
        i = 10
        return i + 15
    'A polymorphic assert, works with tensors and boolean expressions.\n\n  If `cond` is not a tensor, behave like an ordinary assert statement, except\n  that a empty list is returned. If `cond` is a tensor, return a list\n  containing a single TensorFlow assert op.\n\n  Args:\n    cond: Something evaluates to a boolean value. May be a tensor.\n    ex_type: The exception class to use.\n    msg: The error message.\n\n  Returns:\n    A list, containing at most one assert op.\n  '
    if _is_tensor(cond):
        return [control_flow_assert.Assert(cond, [msg])]
    elif not cond:
        raise ex_type(msg)
    else:
        return []

def _is_tensor(x):
    if False:
        for i in range(10):
            print('nop')
    'Returns `True` if `x` is a symbolic tensor-like object.\n\n  Args:\n    x: A python object to check.\n\n  Returns:\n    `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.\n  '
    return isinstance(x, (tensor_lib.Tensor, variables.Variable))

def _ImageDimensions(image, rank):
    if False:
        while True:
            i = 10
    'Returns the dimensions of an image tensor.\n\n  Args:\n    image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.\n    rank: The expected rank of the image\n\n  Returns:\n    A list of corresponding to the dimensions of the\n    input image.  Dimensions that are statically known are python integers,\n    otherwise, they are integer scalar tensors.\n  '
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = array_ops_stack.unstack(array_ops.shape(image), rank)
        return [s if s is not None else d for (s, d) in zip(static_shape, dynamic_shape)]

def _Check3DImage(image, require_static=True):
    if False:
        i = 10
        return i + 15
    'Assert that we are working with a properly shaped image.\n\n  Args:\n    image: 3-D Tensor of shape [height, width, channels]\n    require_static: If `True`, requires that all dimensions of `image` are known\n      and non-zero.\n\n  Raises:\n    ValueError: if `image.shape` is not a 3-vector.\n\n  Returns:\n    An empty list, if `image` has fully defined dimensions. Otherwise, a list\n    containing an assert op is returned.\n  '
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' (shape %s) must be three-dimensional." % image.shape)
    if require_static and (not image_shape.is_fully_defined()):
        raise ValueError("'image' (shape %s) must be fully defined." % image_shape)
    if any((x == 0 for x in image_shape)):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" % image_shape)
    if not image_shape.is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image), ["all dims of 'image.shape' must be > 0."])]
    else:
        return []

def _Assert3DImage(image):
    if False:
        print('Hello World!')
    'Assert that we are working with a properly shaped image.\n\n  Performs the check statically if possible (i.e. if the shape\n  is statically known). Otherwise adds a control dependency\n  to an assert op that checks the dynamic shape.\n\n  Args:\n    image: 3-D Tensor of shape [height, width, channels]\n\n  Raises:\n    ValueError: if `image.shape` is not a 3-vector.\n\n  Returns:\n    If the shape of `image` could be verified statically, `image` is\n    returned unchanged, otherwise there will be a control dependency\n    added that asserts the correct dynamic shape.\n  '
    return control_flow_ops.with_dependencies(_Check3DImage(image, require_static=False), image)

def _AssertAtLeast3DImage(image):
    if False:
        i = 10
        return i + 15
    'Assert that we are working with a properly shaped image.\n\n  Performs the check statically if possible (i.e. if the shape\n  is statically known). Otherwise adds a control dependency\n  to an assert op that checks the dynamic shape.\n\n  Args:\n    image: >= 3-D Tensor of size [*, height, width, depth]\n\n  Raises:\n    ValueError: if image.shape is not a [>= 3] vector.\n\n  Returns:\n    If the shape of `image` could be verified statically, `image` is\n    returned unchanged, otherwise there will be a control dependency\n    added that asserts the correct dynamic shape.\n  '
    return control_flow_ops.with_dependencies(_CheckAtLeast3DImage(image, require_static=False), image)

def _CheckAtLeast3DImage(image, require_static=True):
    if False:
        for i in range(10):
            print('nop')
    'Assert that we are working with a properly shaped image.\n\n  Args:\n    image: >= 3-D Tensor of size [*, height, width, depth]\n    require_static: If `True`, requires that all dimensions of `image` are known\n      and non-zero.\n\n  Raises:\n    ValueError: if image.shape is not a [>= 3] vector.\n\n  Returns:\n    An empty list, if `image` has fully defined dimensions. Otherwise, a list\n    containing an assert op is returned.\n  '
    try:
        if image.get_shape().ndims is None:
            image_shape = image.get_shape().with_rank(3)
        else:
            image_shape = image.get_shape().with_rank_at_least(3)
    except ValueError:
        raise ValueError("'image' (shape %s) must be at least three-dimensional." % image.shape)
    if require_static and (not image_shape.is_fully_defined()):
        raise ValueError("'image' must be fully defined.")
    if any((x == 0 for x in image_shape[-3:])):
        raise ValueError("inner 3 dims of 'image.shape' must be > 0: %s" % image_shape)
    if not image_shape[-3:].is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image)[-3:], ["inner 3 dims of 'image.shape' must be > 0."]), check_ops.assert_greater_equal(array_ops.rank(image), 3, message="'image' must be at least three-dimensional.")]
    else:
        return []

def _AssertGrayscaleImage(image):
    if False:
        print('Hello World!')
    'Assert that we are working with a properly shaped grayscale image.\n\n  Performs the check statically if possible (i.e. if the shape\n  is statically known). Otherwise adds a control dependency\n  to an assert op that checks the dynamic shape.\n\n  Args:\n    image: >= 2-D Tensor of size [*, 1]\n\n  Raises:\n    ValueError: if image.shape is not a [>= 2] vector or if\n              last dimension is not size 1.\n\n  Returns:\n    If the shape of `image` could be verified statically, `image` is\n    returned unchanged, otherwise there will be a control dependency\n    added that asserts the correct dynamic shape.\n  '
    return control_flow_ops.with_dependencies(_CheckGrayscaleImage(image, require_static=False), image)

def _CheckGrayscaleImage(image, require_static=True):
    if False:
        while True:
            i = 10
    'Assert that we are working with properly shaped grayscale image.\n\n  Args:\n    image: >= 2-D Tensor of size [*, 1]\n    require_static: Boolean, whether static shape is required.\n\n  Raises:\n    ValueError: if image.shape is not a [>= 2] vector or if\n              last dimension is not size 1.\n\n  Returns:\n    An empty list, if `image` has fully defined dimensions. Otherwise, a list\n    containing an assert op is returned.\n  '
    try:
        if image.get_shape().ndims is None:
            image_shape = image.get_shape().with_rank(2)
        else:
            image_shape = image.get_shape().with_rank_at_least(2)
    except ValueError:
        raise ValueError('A grayscale image (shape %s) must be at least two-dimensional.' % image.shape)
    if require_static and (not image_shape.is_fully_defined()):
        raise ValueError("'image' must be fully defined.")
    if image_shape.is_fully_defined():
        if image_shape[-1] != 1:
            raise ValueError('Last dimension of a grayscale image should be size 1.')
    if not image_shape.is_fully_defined():
        return [check_ops.assert_equal(array_ops.shape(image)[-1], 1, message='Last dimension of a grayscale image should be size 1.'), check_ops.assert_greater_equal(array_ops.rank(image), 3, message='A grayscale image must be at least two-dimensional.')]
    else:
        return []

def fix_image_flip_shape(image, result):
    if False:
        for i in range(10):
            print('nop')
    "Set the shape to 3 dimensional if we don't know anything else.\n\n  Args:\n    image: original image size\n    result: flipped or transformed image\n\n  Returns:\n    An image whose shape is at least (None, None, None).\n  "
    image_shape = image.get_shape()
    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None])
    else:
        result.set_shape(image_shape)
    return result

@tf_export('image.random_flip_up_down')
@dispatch.add_dispatch_support
def random_flip_up_down(image, seed=None):
    if False:
        while True:
            i = 10
    'Randomly flips an image vertically (upside down).\n\n  With a 1 in 2 chance, outputs the contents of `image` flipped along the first\n  dimension, which is `height`.  Otherwise, output the image as-is.\n  When passing a batch of images, each image will be randomly flipped\n  independent of other images.\n\n  Example usage:\n\n  >>> image = np.array([[[1], [2]], [[3], [4]]])\n  >>> tf.image.random_flip_up_down(image, 3).numpy().tolist()\n  [[[3], [4]], [[1], [2]]]\n\n  Randomly flip multiple images.\n\n  >>> images = np.array(\n  ... [\n  ...     [[[1], [2]], [[3], [4]]],\n  ...     [[[5], [6]], [[7], [8]]]\n  ... ])\n  >>> tf.image.random_flip_up_down(images, 4).numpy().tolist()\n  [[[[3], [4]], [[1], [2]]], [[[5], [6]], [[7], [8]]]]\n\n  For producing deterministic results given a `seed` value, use\n  `tf.image.stateless_random_flip_up_down`. Unlike using the `seed` param\n  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the\n  same results given the same seed independent of how many times the function is\n  called, and independent of global seed settings (e.g. tf.random.set_seed).\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    seed: A Python integer. Used to create a random seed. See\n      `tf.compat.v1.set_random_seed` for behavior.\n\n  Returns:\n    A tensor of the same type and shape as `image`.\n  Raises:\n    ValueError: if the shape of `image` not supported.\n  '
    random_func = functools.partial(random_ops.random_uniform, seed=seed)
    return _random_flip(image, 0, random_func, 'random_flip_up_down')

@tf_export('image.random_flip_left_right')
@dispatch.add_dispatch_support
def random_flip_left_right(image, seed=None):
    if False:
        while True:
            i = 10
    'Randomly flip an image horizontally (left to right).\n\n  With a 1 in 2 chance, outputs the contents of `image` flipped along the\n  second dimension, which is `width`.  Otherwise output the image as-is.\n  When passing a batch of images, each image will be randomly flipped\n  independent of other images.\n\n  Example usage:\n\n  >>> image = np.array([[[1], [2]], [[3], [4]]])\n  >>> tf.image.random_flip_left_right(image, 5).numpy().tolist()\n  [[[2], [1]], [[4], [3]]]\n\n  Randomly flip multiple images.\n\n  >>> images = np.array(\n  ... [\n  ...     [[[1], [2]], [[3], [4]]],\n  ...     [[[5], [6]], [[7], [8]]]\n  ... ])\n  >>> tf.image.random_flip_left_right(images, 6).numpy().tolist()\n  [[[[2], [1]], [[4], [3]]], [[[5], [6]], [[7], [8]]]]\n\n  For producing deterministic results given a `seed` value, use\n  `tf.image.stateless_random_flip_left_right`. Unlike using the `seed` param\n  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the\n  same results given the same seed independent of how many times the function is\n  called, and independent of global seed settings (e.g. tf.random.set_seed).\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    seed: A Python integer. Used to create a random seed. See\n      `tf.compat.v1.set_random_seed` for behavior.\n\n  Returns:\n    A tensor of the same type and shape as `image`.\n\n  Raises:\n    ValueError: if the shape of `image` not supported.\n  '
    random_func = functools.partial(random_ops.random_uniform, seed=seed)
    return _random_flip(image, 1, random_func, 'random_flip_left_right')

@tf_export('image.stateless_random_flip_left_right', v1=[])
@dispatch.add_dispatch_support
def stateless_random_flip_left_right(image, seed):
    if False:
        i = 10
        return i + 15
    'Randomly flip an image horizontally (left to right) deterministically.\n\n  Guarantees the same results given the same `seed` independent of how many\n  times the function is called, and independent of global seed settings (e.g.\n  `tf.random.set_seed`).\n\n  Example usage:\n\n  >>> image = np.array([[[1], [2]], [[3], [4]]])\n  >>> seed = (2, 3)\n  >>> tf.image.stateless_random_flip_left_right(image, seed).numpy().tolist()\n  [[[2], [1]], [[4], [3]]]\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n\n  Returns:\n    A tensor of the same type and shape as `image`.\n  '
    random_func = functools.partial(stateless_random_ops.stateless_random_uniform, seed=seed)
    return _random_flip(image, 1, random_func, 'stateless_random_flip_left_right')

@tf_export('image.stateless_random_flip_up_down', v1=[])
@dispatch.add_dispatch_support
def stateless_random_flip_up_down(image, seed):
    if False:
        for i in range(10):
            print('nop')
    'Randomly flip an image vertically (upside down) deterministically.\n\n  Guarantees the same results given the same `seed` independent of how many\n  times the function is called, and independent of global seed settings (e.g.\n  `tf.random.set_seed`).\n\n  Example usage:\n\n  >>> image = np.array([[[1], [2]], [[3], [4]]])\n  >>> seed = (2, 3)\n  >>> tf.image.stateless_random_flip_up_down(image, seed).numpy().tolist()\n  [[[3], [4]], [[1], [2]]]\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n\n  Returns:\n    A tensor of the same type and shape as `image`.\n  '
    random_func = functools.partial(stateless_random_ops.stateless_random_uniform, seed=seed)
    return _random_flip(image, 0, random_func, 'stateless_random_flip_up_down')

def _random_flip(image, flip_index, random_func, scope_name):
    if False:
        i = 10
        return i + 15
    'Randomly (50% chance) flip an image along axis `flip_index`.\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    flip_index: Dimension along which to flip the image.\n      Vertical is 0, Horizontal is 1.\n    random_func: partial function for calling either stateful or stateless\n      random ops with `seed` parameter specified.\n    scope_name: Name of the scope in which the ops are added.\n\n  Returns:\n    A tensor of the same type and shape as `image`.\n\n  Raises:\n    ValueError: if the shape of `image` not supported.\n  '
    with ops.name_scope(None, scope_name, [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        image = _AssertAtLeast3DImage(image)
        shape = image.get_shape()

        def f_rank3():
            if False:
                print('Hello World!')
            uniform_random = random_func(shape=[], minval=0, maxval=1.0)
            mirror_cond = math_ops.less(uniform_random, 0.5)
            result = tf_cond.cond(mirror_cond, lambda : array_ops.reverse(image, [flip_index]), lambda : image, name=scope)
            return fix_image_flip_shape(image, result)

        def f_rank4():
            if False:
                print('Hello World!')
            batch_size = array_ops.shape(image)[0]
            uniform_random = random_func(shape=[batch_size], minval=0, maxval=1.0)
            flips = math_ops.round(array_ops.reshape(uniform_random, [batch_size, 1, 1, 1]))
            flips = math_ops.cast(flips, image.dtype)
            flipped_input = array_ops.reverse(image, [flip_index + 1])
            return flips * flipped_input + (1 - flips) * image
        if shape.ndims is None:
            rank = array_ops.rank(image)
            return tf_cond.cond(math_ops.equal(rank, 3), f_rank3, f_rank4)
        if shape.ndims == 3:
            return f_rank3()
        elif shape.ndims == 4:
            return f_rank4()
        else:
            raise ValueError("'image' (shape %s) must have either 3 or 4 dimensions." % shape)

@tf_export('image.flip_left_right')
@dispatch.add_dispatch_support
def flip_left_right(image):
    if False:
        while True:
            i = 10
    'Flip an image horizontally (left to right).\n\n  Outputs the contents of `image` flipped along the width dimension.\n\n  See also `tf.reverse`.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.flip_left_right(x)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 4.,  5.,  6.],\n          [ 1.,  2.,  3.]],\n         [[10., 11., 12.],\n          [ 7.,  8.,  9.]]], dtype=float32)>\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n\n  Returns:\n    A tensor of the same type and shape as `image`.\n\n  Raises:\n    ValueError: if the shape of `image` not supported.\n  '
    return _flip(image, 1, 'flip_left_right')

@tf_export('image.flip_up_down')
@dispatch.add_dispatch_support
def flip_up_down(image):
    if False:
        print('Hello World!')
    'Flip an image vertically (upside down).\n\n  Outputs the contents of `image` flipped along the height dimension.\n\n  See also `reverse()`.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.flip_up_down(x)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 7.,  8.,  9.],\n          [10., 11., 12.]],\n         [[ 1.,  2.,  3.],\n          [ 4.,  5.,  6.]]], dtype=float32)>\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n\n  Returns:\n    A `Tensor` of the same type and shape as `image`.\n\n  Raises:\n    ValueError: if the shape of `image` not supported.\n  '
    return _flip(image, 0, 'flip_up_down')

def _flip(image, flip_index, scope_name):
    if False:
        for i in range(10):
            print('nop')
    'Flip an image either horizontally or vertically.\n\n  Outputs the contents of `image` flipped along the dimension `flip_index`.\n\n  See also `reverse()`.\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    flip_index: 0 For vertical, 1 for horizontal.\n    scope_name: string, scope name.\n\n  Returns:\n    A `Tensor` of the same type and shape as `image`.\n\n  Raises:\n    ValueError: if the shape of `image` not supported.\n  '
    with ops.name_scope(None, scope_name, [image]):
        image = ops.convert_to_tensor(image, name='image')
        image = _AssertAtLeast3DImage(image)
        shape = image.get_shape()

        def f_rank3():
            if False:
                return 10
            return fix_image_flip_shape(image, array_ops.reverse(image, [flip_index]))

        def f_rank4():
            if False:
                for i in range(10):
                    print('nop')
            return array_ops.reverse(image, [flip_index + 1])
        if shape.ndims is None:
            rank = array_ops.rank(image)
            return tf_cond.cond(math_ops.equal(rank, 3), f_rank3, f_rank4)
        elif shape.ndims == 3:
            return f_rank3()
        elif shape.ndims == 4:
            return f_rank4()
        else:
            raise ValueError("'image' (shape %s)must have either 3 or 4 dimensions." % shape)

@tf_export('image.rot90')
@dispatch.add_dispatch_support
def rot90(image, k=1, name=None):
    if False:
        i = 10
        return i + 15
    'Rotate image(s) by 90 degrees.\n\n\n  For example:\n\n  >>> a=tf.constant([[[1],[2]],\n  ...                [[3],[4]]])\n  >>> # rotating `a` counter clockwise by 90 degrees\n  >>> a_rot=tf.image.rot90(a)\n  >>> print(a_rot[...,0].numpy())\n  [[2 4]\n   [1 3]]\n  >>> # rotating `a` counter clockwise by 270 degrees\n  >>> a_rot=tf.image.rot90(a, k=3)\n  >>> print(a_rot[...,0].numpy())\n  [[3 1]\n   [4 2]]\n  >>> # rotating `a` clockwise by 180 degrees\n  >>> a_rot=tf.image.rot90(a, k=-2)\n  >>> print(a_rot[...,0].numpy())\n  [[4 3]\n   [2 1]]\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    k: A scalar integer tensor. The number of times the image(s) are rotated by\n      90 degrees.\n    name: A name for this operation (optional).\n\n  Returns:\n    A rotated tensor of the same type and shape as `image`.\n\n  Raises:\n    ValueError: if the shape of `image` not supported.\n  '
    with ops.name_scope(name, 'rot90', [image, k]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        image = _AssertAtLeast3DImage(image)
        k = ops.convert_to_tensor(k, dtype=dtypes.int32, name='k')
        k.get_shape().assert_has_rank(0)
        k = math_ops.mod(k, 4)
        shape = image.get_shape()
        if shape.ndims is None:
            rank = array_ops.rank(image)

            def f_rank3():
                if False:
                    i = 10
                    return i + 15
                return _rot90_3D(image, k, scope)

            def f_rank4():
                if False:
                    return 10
                return _rot90_4D(image, k, scope)
            return tf_cond.cond(math_ops.equal(rank, 3), f_rank3, f_rank4)
        elif shape.ndims == 3:
            return _rot90_3D(image, k, scope)
        elif shape.ndims == 4:
            return _rot90_4D(image, k, scope)
        else:
            raise ValueError("'image' (shape %s) must have either 3 or 4 dimensions." % shape)

def _rot90_3D(image, k, name_scope):
    if False:
        for i in range(10):
            print('nop')
    'Rotate image counter-clockwise by 90 degrees `k` times.\n\n  Args:\n    image: 3-D Tensor of shape `[height, width, channels]`.\n    k: A scalar integer. The number of times the image is rotated by 90 degrees.\n    name_scope: A valid TensorFlow name scope.\n\n  Returns:\n    A 3-D tensor of the same type and shape as `image`.\n\n  '

    def _rot90():
        if False:
            return 10
        return array_ops.transpose(array_ops.reverse_v2(image, [1]), [1, 0, 2])

    def _rot180():
        if False:
            return 10
        return array_ops.reverse_v2(image, [0, 1])

    def _rot270():
        if False:
            i = 10
            return i + 15
        return array_ops.reverse_v2(array_ops.transpose(image, [1, 0, 2]), [1])
    cases = [(math_ops.equal(k, 1), _rot90), (math_ops.equal(k, 2), _rot180), (math_ops.equal(k, 3), _rot270)]
    result = control_flow_case.case(cases, default=lambda : image, exclusive=True, name=name_scope)
    result.set_shape([None, None, image.get_shape()[2]])
    return result

def _rot90_4D(images, k, name_scope):
    if False:
        i = 10
        return i + 15
    'Rotate batch of images counter-clockwise by 90 degrees `k` times.\n\n  Args:\n    images: 4-D Tensor of shape `[height, width, channels]`.\n    k: A scalar integer. The number of times the images are rotated by 90\n      degrees.\n    name_scope: A valid TensorFlow name scope.\n\n  Returns:\n    A 4-D `Tensor` of the same type and shape as `images`.\n  '

    def _rot90():
        if False:
            while True:
                i = 10
        return array_ops.transpose(array_ops.reverse_v2(images, [2]), [0, 2, 1, 3])

    def _rot180():
        if False:
            i = 10
            return i + 15
        return array_ops.reverse_v2(images, [1, 2])

    def _rot270():
        if False:
            print('Hello World!')
        return array_ops.reverse_v2(array_ops.transpose(images, [0, 2, 1, 3]), [2])
    cases = [(math_ops.equal(k, 1), _rot90), (math_ops.equal(k, 2), _rot180), (math_ops.equal(k, 3), _rot270)]
    result = control_flow_case.case(cases, default=lambda : images, exclusive=True, name=name_scope)
    shape = result.get_shape()
    result.set_shape([shape[0], None, None, shape[3]])
    return result

@tf_export('image.transpose', v1=['image.transpose', 'image.transpose_image'])
@dispatch.add_dispatch_support
def transpose(image, name=None):
    if False:
        return 10
    'Transpose image(s) by swapping the height and width dimension.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.transpose(x)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 1.,  2.,  3.],\n          [ 7.,  8.,  9.]],\n         [[ 4.,  5.,  6.],\n          [10., 11., 12.]]], dtype=float32)>\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    name: A name for this operation (optional).\n\n  Returns:\n    If `image` was 4-D, a 4-D float Tensor of shape\n   `[batch, width, height, channels]`\n    If `image` was 3-D, a 3-D float Tensor of shape\n   `[width, height, channels]`\n\n  Raises:\n    ValueError: if the shape of `image` not supported.\n\n  Usage Example:\n\n  >>> image = [[[1, 2], [3, 4]],\n  ...         [[5, 6], [7, 8]],\n  ...         [[9, 10], [11, 12]]]\n  >>> image = tf.constant(image)\n  >>> tf.image.transpose(image)\n  <tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=\n  array([[[ 1,  2],\n         [ 5,  6],\n         [ 9, 10]],\n        [[ 3,  4],\n         [ 7,  8],\n         [11, 12]]], dtype=int32)>\n  '
    with ops.name_scope(name, 'transpose', [image]):
        image = ops.convert_to_tensor(image, name='image')
        image = _AssertAtLeast3DImage(image)
        shape = image.get_shape()
        if shape.ndims is None:
            rank = array_ops.rank(image)

            def f_rank3():
                if False:
                    i = 10
                    return i + 15
                return array_ops.transpose(image, [1, 0, 2], name=name)

            def f_rank4():
                if False:
                    return 10
                return array_ops.transpose(image, [0, 2, 1, 3], name=name)
            return tf_cond.cond(math_ops.equal(rank, 3), f_rank3, f_rank4)
        elif shape.ndims == 3:
            return array_ops.transpose(image, [1, 0, 2], name=name)
        elif shape.ndims == 4:
            return array_ops.transpose(image, [0, 2, 1, 3], name=name)
        else:
            raise ValueError("'image' (shape %s) must have either 3 or 4 dimensions." % shape)

@tf_export('image.central_crop')
@dispatch.add_dispatch_support
def central_crop(image, central_fraction):
    if False:
        while True:
            i = 10
    'Crop the central region of the image(s).\n\n  Remove the outer parts of an image but retain the central region of the image\n  along each dimension. If we specify `central_fraction = 0.5`, this function\n  returns the region marked with "X" in the below diagram. The larger the value\n  of `central_fraction`, the larger the dimension of the region to be cropped\n  and retained.\n\n       --------\n      |        |\n      |  XXXX  |\n      |  XXXX  |\n      |        |   where "X" is the central 50% of the image.\n       --------\n\n  This function works on either a single image (`image` is a 3-D Tensor), or a\n  batch of images (`image` is a 4-D Tensor).\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0],\n  ...       [7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]],\n  ...     [[13.0, 14.0, 15.0],\n  ...       [16.0, 17.0, 18.0],\n  ...       [19.0, 20.0, 21.0],\n  ...       [22.0, 23.0, 24.0]],\n  ...     [[25.0, 26.0, 27.0],\n  ...       [28.0, 29.0, 30.0],\n  ...       [31.0, 32.0, 33.0],\n  ...       [34.0, 35.0, 36.0]],\n  ...     [[37.0, 38.0, 39.0],\n  ...       [40.0, 41.0, 42.0],\n  ...       [43.0, 44.0, 45.0],\n  ...       [46.0, 47.0, 48.0]]]\n  >>> tf.image.central_crop(x, 0.5)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[16., 17., 18.],\n          [19., 20., 21.]],\n         [[28., 29., 30.],\n          [31., 32., 33.]]], dtype=float32)>\n\n  Args:\n    image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D\n      Tensor of shape [batch_size, height, width, depth].\n    central_fraction: float (0, 1], fraction of size to crop\n\n  Raises:\n    ValueError: if central_crop_fraction is not within (0, 1].\n\n  Returns:\n    3-D / 4-D float Tensor, as per the input.\n  '
    with ops.name_scope(None, 'central_crop', [image]):
        image = ops.convert_to_tensor(image, name='image')
        central_fraction_static = tensor_util.constant_value(central_fraction)
        if central_fraction_static is not None:
            if central_fraction_static <= 0.0 or central_fraction_static > 1.0:
                raise ValueError('central_fraction must be within (0, 1]')
            if central_fraction_static == 1.0:
                return image
        else:
            assert_ops = _assert(math_ops.logical_or(central_fraction > 0.0, central_fraction <= 1.0), ValueError, 'central_fraction must be within (0, 1]')
            image = control_flow_ops.with_dependencies(assert_ops, image)
        _AssertAtLeast3DImage(image)
        rank = image.get_shape().ndims
        if rank != 3 and rank != 4:
            raise ValueError('`image` should either be a Tensor with rank = 3 or rank = 4. Had rank = {}.'.format(rank))

        def _get_dim(tensor, idx):
            if False:
                i = 10
                return i + 15
            static_shape = tensor.get_shape().dims[idx].value
            if static_shape is not None:
                return (static_shape, False)
            return (array_ops.shape(tensor)[idx], True)
        if rank == 3:
            (img_h, dynamic_h) = _get_dim(image, 0)
            (img_w, dynamic_w) = _get_dim(image, 1)
            img_d = image.get_shape()[2]
        else:
            img_bs = image.get_shape()[0]
            (img_h, dynamic_h) = _get_dim(image, 1)
            (img_w, dynamic_w) = _get_dim(image, 2)
            img_d = image.get_shape()[3]
        dynamic_h = dynamic_h or central_fraction_static is None
        dynamic_w = dynamic_w or central_fraction_static is None
        if dynamic_h:
            img_hd = math_ops.cast(img_h, dtypes.float64)
            bbox_h_start = math_ops.cast((img_hd - img_hd * math_ops.cast(central_fraction, dtypes.float64)) / 2, dtypes.int32)
        else:
            img_hd = float(img_h)
            bbox_h_start = int((img_hd - img_hd * central_fraction_static) / 2)
        if dynamic_w:
            img_wd = math_ops.cast(img_w, dtypes.float64)
            bbox_w_start = math_ops.cast((img_wd - img_wd * math_ops.cast(central_fraction, dtypes.float64)) / 2, dtypes.int32)
        else:
            img_wd = float(img_w)
            bbox_w_start = int((img_wd - img_wd * central_fraction_static) / 2)
        bbox_h_size = img_h - bbox_h_start * 2
        bbox_w_size = img_w - bbox_w_start * 2
        if rank == 3:
            bbox_begin = array_ops_stack.stack([bbox_h_start, bbox_w_start, 0])
            bbox_size = array_ops_stack.stack([bbox_h_size, bbox_w_size, -1])
        else:
            bbox_begin = array_ops_stack.stack([0, bbox_h_start, bbox_w_start, 0])
            bbox_size = array_ops_stack.stack([-1, bbox_h_size, bbox_w_size, -1])
        image = array_ops.slice(image, bbox_begin, bbox_size)
        if rank == 3:
            image.set_shape([None if dynamic_h else bbox_h_size, None if dynamic_w else bbox_w_size, img_d])
        else:
            image.set_shape([img_bs, None if dynamic_h else bbox_h_size, None if dynamic_w else bbox_w_size, img_d])
        return image

@tf_export('image.pad_to_bounding_box')
@dispatch.add_dispatch_support
def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    if False:
        while True:
            i = 10
    'Pad `image` with zeros to the specified `height` and `width`.\n\n  Adds `offset_height` rows of zeros on top, `offset_width` columns of\n  zeros on the left, and then pads the image on the bottom and right\n  with zeros until it has dimensions `target_height`, `target_width`.\n\n  This op does nothing if `offset_*` is zero and the image already has size\n  `target_height` by `target_width`.\n\n  Usage Example:\n\n  >>> x = [[[1., 2., 3.],\n  ...       [4., 5., 6.]],\n  ...       [[7., 8., 9.],\n  ...       [10., 11., 12.]]]\n  >>> padded_image = tf.image.pad_to_bounding_box(x, 1, 1, 4, 4)\n  >>> padded_image\n  <tf.Tensor: shape=(4, 4, 3), dtype=float32, numpy=\n  array([[[ 0.,  0.,  0.],\n  [ 0.,  0.,  0.],\n  [ 0.,  0.,  0.],\n  [ 0.,  0.,  0.]],\n  [[ 0.,  0.,  0.],\n  [ 1.,  2.,  3.],\n  [ 4.,  5.,  6.],\n  [ 0.,  0.,  0.]],\n  [[ 0.,  0.,  0.],\n  [ 7.,  8.,  9.],\n  [10., 11., 12.],\n  [ 0.,  0.,  0.]],\n  [[ 0.,  0.,  0.],\n  [ 0.,  0.,  0.],\n  [ 0.,  0.,  0.],\n  [ 0.,  0.,  0.]]], dtype=float32)>\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    offset_height: Number of rows of zeros to add on top.\n    offset_width: Number of columns of zeros to add on the left.\n    target_height: Height of output image.\n    target_width: Width of output image.\n\n  Returns:\n    If `image` was 4-D, a 4-D float Tensor of shape\n    `[batch, target_height, target_width, channels]`\n    If `image` was 3-D, a 3-D float Tensor of shape\n    `[target_height, target_width, channels]`\n\n  Raises:\n    ValueError: If the shape of `image` is incompatible with the `offset_*` or\n      `target_*` arguments, or either `offset_height` or `offset_width` is\n      negative.\n  '
    return pad_to_bounding_box_internal(image, offset_height, offset_width, target_height, target_width, check_dims=True)

def pad_to_bounding_box_internal(image, offset_height, offset_width, target_height, target_width, check_dims):
    if False:
        while True:
            i = 10
    'Pad `image` with zeros to the specified `height` and `width`.\n\n  Adds `offset_height` rows of zeros on top, `offset_width` columns of\n  zeros on the left, and then pads the image on the bottom and right\n  with zeros until it has dimensions `target_height`, `target_width`.\n\n  This op does nothing if `offset_*` is zero and the image already has size\n  `target_height` by `target_width`.\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    offset_height: Number of rows of zeros to add on top.Must be 0-D `Tensor` of\n      dtype int32 or int64. Can also a python integer.\n    offset_width: Number of columns of zeros to add on the left.Must be 0-D\n      `Tensor` of dtype int32 or int64. Can also a python integer.\n    target_height: Height of output image.Must be 0-D `Tensor` of dtype int32 or\n      int64. Can also a python integer.\n    target_width: Width of output image.Must be 0-D `Tensor` of dtype int32 or\n      int64. Can also a python integer.\n    check_dims: If True, assert that dimensions are non-negative and in range.\n      In multi-GPU distributed settings, assertions can cause program slowdown.\n      Setting this parameter to `False` avoids this, resulting in faster speed\n      in some situations, with the tradeoff being that some error checking is\n      not happening.\n\n  Returns:\n    If `image` was 4-D, a 4-D float Tensor of shape\n    `[batch, target_height, target_width, channels]`\n    If `image` was 3-D, a 3-D float Tensor of shape\n    `[target_height, target_width, channels]`\n\n  Raises:\n    ValueError: If the shape of `image` is incompatible with the `offset_*` or\n      `target_*` arguments, or either `offset_height` or `offset_width` is\n      negative. Not raised if `check_dims` is `False`.\n  '
    with ops.name_scope(None, 'pad_to_bounding_box', [image]):
        image = ops.convert_to_tensor(image, name='image')
        is_batch = True
        image_shape = image.get_shape()
        if image_shape.ndims == 3:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image_shape.ndims != 4:
            raise ValueError("'image' (shape %s) must have either 3 or 4 dimensions." % image_shape)
        (batch, height, width, depth) = _ImageDimensions(image, rank=4)
        after_padding_width = target_width - offset_width - width
        after_padding_height = target_height - offset_height - height
        if check_dims:
            assert_ops = _CheckAtLeast3DImage(image, require_static=False)
            assert_ops += _assert(offset_height >= 0, ValueError, 'offset_height must be >= 0')
            assert_ops += _assert(offset_width >= 0, ValueError, 'offset_width must be >= 0')
            assert_ops += _assert(after_padding_width >= 0, ValueError, 'width must be <= target - offset')
            assert_ops += _assert(after_padding_height >= 0, ValueError, 'height must be <= target - offset')
            image = control_flow_ops.with_dependencies(assert_ops, image)
        paddings = array_ops.reshape(array_ops_stack.stack([0, 0, offset_height, after_padding_height, offset_width, after_padding_width, 0, 0]), [4, 2])
        padded = array_ops.pad(image, paddings)
        padded_shape = [None if _is_tensor(i) else i for i in [batch, target_height, target_width, depth]]
        padded.set_shape(padded_shape)
        if not is_batch:
            padded = array_ops.squeeze(padded, axis=[0])
        return padded

@tf_export('image.crop_to_bounding_box')
@dispatch.add_dispatch_support
def crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    if False:
        while True:
            i = 10
    'Crops an `image` to a specified bounding box.\n\n  This op cuts a rectangular bounding box out of `image`. The top-left corner\n  of the bounding box is at `offset_height, offset_width` in `image`, and the\n  lower-right corner is at\n  `offset_height + target_height, offset_width + target_width`.\n\n  Example Usage:\n\n  >>> image = tf.constant(np.arange(1, 28, dtype=np.float32), shape=[3, 3, 3])\n  >>> image[:,:,0] # print the first channel of the 3-D tensor\n  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=\n  array([[ 1.,  4.,  7.],\n         [10., 13., 16.],\n         [19., 22., 25.]], dtype=float32)>\n  >>> cropped_image = tf.image.crop_to_bounding_box(image, 0, 0, 2, 2)\n  >>> cropped_image[:,:,0] # print the first channel of the cropped 3-D tensor\n  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=\n  array([[ 1.,  4.],\n         [10., 13.]], dtype=float32)>\n\n  Args:\n    image: 4-D `Tensor` of shape `[batch, height, width, channels]` or 3-D\n      `Tensor` of shape `[height, width, channels]`.\n    offset_height: Vertical coordinate of the top-left corner of the bounding\n      box in `image`. Must be 0-D int32 `Tensor` or python integer.\n    offset_width: Horizontal coordinate of the top-left corner of the bounding\n      box in `image`. Must be 0-D int32 `Tensor` or python integer.\n    target_height: Height of the bounding box. Must be 0-D int32 `Tensor` or\n      python integer.\n    target_width: Width of the bounding box. Must be 0-D int32 `Tensor` or\n      python integer.\n\n  Returns:\n    If `image` was 4-D, a 4-D `Tensor` of shape\n    `[batch, target_height, target_width, channels]`.\n    If `image` was 3-D, a 3-D `Tensor` of shape\n    `[target_height, target_width, channels]`.\n    It has the same dtype with `image`.\n\n  Raises:\n    ValueError: `image` is not a 3-D or 4-D `Tensor`.\n    ValueError: `offset_width < 0` or `offset_height < 0`.\n    ValueError: `target_width <= 0` or `target_height <= 0`.\n    ValueError: `width < offset_width + target_width` or\n      `height < offset_height + target_height`.\n  '
    with ops.name_scope(None, 'crop_to_bounding_box', [image]):
        image = ops.convert_to_tensor(image, name='image')
        is_batch = True
        image_shape = image.get_shape()
        if image_shape.ndims == 3:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image_shape.ndims != 4:
            raise ValueError("'image' (shape %s) must have either 3 or 4 dimensions." % image_shape)
        assert_ops = _CheckAtLeast3DImage(image, require_static=False)
        (batch, height, width, depth) = _ImageDimensions(image, rank=4)
        assert_ops += _assert(offset_width >= 0, ValueError, 'offset_width must be >= 0.')
        assert_ops += _assert(offset_height >= 0, ValueError, 'offset_height must be >= 0.')
        assert_ops += _assert(target_width > 0, ValueError, 'target_width must be > 0.')
        assert_ops += _assert(target_height > 0, ValueError, 'target_height must be > 0.')
        assert_ops += _assert(width >= target_width + offset_width, ValueError, 'width must be >= target + offset.')
        assert_ops += _assert(height >= target_height + offset_height, ValueError, 'height must be >= target + offset.')
        image = control_flow_ops.with_dependencies(assert_ops, image)
        cropped = array_ops.slice(image, array_ops_stack.stack([0, offset_height, offset_width, 0]), array_ops_stack.stack([array_ops.shape(image)[0], target_height, target_width, array_ops.shape(image)[3]]))
        cropped_shape = [None if _is_tensor(i) else i for i in [batch, target_height, target_width, depth]]
        cropped.set_shape(cropped_shape)
        if not is_batch:
            cropped = array_ops.squeeze(cropped, axis=[0])
        return cropped

@tf_export('image.resize_with_crop_or_pad', v1=['image.resize_with_crop_or_pad', 'image.resize_image_with_crop_or_pad'])
@dispatch.add_dispatch_support
def resize_image_with_crop_or_pad(image, target_height, target_width):
    if False:
        i = 10
        return i + 15
    'Crops and/or pads an image to a target width and height.\n\n  Resizes an image to a target width and height by either centrally\n  cropping the image or padding it evenly with zeros.\n\n  If `width` or `height` is greater than the specified `target_width` or\n  `target_height` respectively, this op centrally crops along that dimension.\n\n  For example:\n\n  >>> image = np.arange(75).reshape(5, 5, 3)  # create 3-D image input\n  >>> image[:,:,0]  # print first channel just for demo purposes\n  array([[ 0,  3,  6,  9, 12],\n         [15, 18, 21, 24, 27],\n         [30, 33, 36, 39, 42],\n         [45, 48, 51, 54, 57],\n         [60, 63, 66, 69, 72]])\n  >>> image = tf.image.resize_with_crop_or_pad(image, 3, 3)  # crop\n  >>> # print first channel for demo purposes; centrally cropped output\n  >>> image[:,:,0]\n  <tf.Tensor: shape=(3, 3), dtype=int64, numpy=\n  array([[18, 21, 24],\n         [33, 36, 39],\n         [48, 51, 54]])>\n\n  If `width` or `height` is smaller than the specified `target_width` or\n  `target_height` respectively, this op centrally pads with 0 along that\n  dimension.\n\n  For example:\n\n  >>> image = np.arange(1, 28).reshape(3, 3, 3)  # create 3-D image input\n  >>> image[:,:,0]  # print first channel just for demo purposes\n  array([[ 1,  4,  7],\n         [10, 13, 16],\n         [19, 22, 25]])\n  >>> image = tf.image.resize_with_crop_or_pad(image, 5, 5)  # pad\n  >>> # print first channel for demo purposes; we should see 0 paddings\n  >>> image[:,:,0]\n  <tf.Tensor: shape=(5, 5), dtype=int64, numpy=\n  array([[ 0,  0,  0,  0,  0],\n         [ 0,  1,  4,  7,  0],\n         [ 0, 10, 13, 16,  0],\n         [ 0, 19, 22, 25,  0],\n         [ 0,  0,  0,  0,  0]])>\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    target_height: Target height.\n    target_width: Target width.\n\n  Raises:\n    ValueError: if `target_height` or `target_width` are zero or negative.\n\n  Returns:\n    Cropped and/or padded image.\n    If `images` was 4-D, a 4-D float Tensor of shape\n    `[batch, new_height, new_width, channels]`.\n    If `images` was 3-D, a 3-D float Tensor of shape\n    `[new_height, new_width, channels]`.\n  '
    with ops.name_scope(None, 'resize_image_with_crop_or_pad', [image]):
        image = ops.convert_to_tensor(image, name='image')
        image_shape = image.get_shape()
        is_batch = True
        if image_shape.ndims == 3:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image_shape.ndims != 4:
            raise ValueError("'image' (shape %s) must have either 3 or 4 dimensions." % image_shape)
        assert_ops = _CheckAtLeast3DImage(image, require_static=False)
        assert_ops += _assert(target_width > 0, ValueError, 'target_width must be > 0.')
        assert_ops += _assert(target_height > 0, ValueError, 'target_height must be > 0.')
        image = control_flow_ops.with_dependencies(assert_ops, image)
        if _is_tensor(target_height):
            target_height = control_flow_ops.with_dependencies(assert_ops, target_height)
        if _is_tensor(target_width):
            target_width = control_flow_ops.with_dependencies(assert_ops, target_width)

        def max_(x, y):
            if False:
                for i in range(10):
                    print('nop')
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.maximum(x, y)
            else:
                return max(x, y)

        def min_(x, y):
            if False:
                print('Hello World!')
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.minimum(x, y)
            else:
                return min(x, y)

        def equal_(x, y):
            if False:
                return 10
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.equal(x, y)
            else:
                return x == y
        (_, height, width, _) = _ImageDimensions(image, rank=4)
        width_diff = target_width - width
        offset_crop_width = max_(-width_diff // 2, 0)
        offset_pad_width = max_(width_diff // 2, 0)
        height_diff = target_height - height
        offset_crop_height = max_(-height_diff // 2, 0)
        offset_pad_height = max_(height_diff // 2, 0)
        cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width, min_(target_height, height), min_(target_width, width))
        resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width, target_height, target_width)
        if resized.get_shape().ndims is None:
            raise ValueError('resized contains no shape.')
        (_, resized_height, resized_width, _) = _ImageDimensions(resized, rank=4)
        assert_ops = []
        assert_ops += _assert(equal_(resized_height, target_height), ValueError, 'resized height is not correct.')
        assert_ops += _assert(equal_(resized_width, target_width), ValueError, 'resized width is not correct.')
        resized = control_flow_ops.with_dependencies(assert_ops, resized)
        if not is_batch:
            resized = array_ops.squeeze(resized, axis=[0])
        return resized

@tf_export(v1=['image.ResizeMethod'])
class ResizeMethodV1:
    """See `v1.image.resize` for details."""
    BILINEAR = 0
    NEAREST_NEIGHBOR = 1
    BICUBIC = 2
    AREA = 3

@tf_export('image.ResizeMethod', v1=[])
class ResizeMethod:
    """See `tf.image.resize` for details."""
    BILINEAR = 'bilinear'
    NEAREST_NEIGHBOR = 'nearest'
    BICUBIC = 'bicubic'
    AREA = 'area'
    LANCZOS3 = 'lanczos3'
    LANCZOS5 = 'lanczos5'
    GAUSSIAN = 'gaussian'
    MITCHELLCUBIC = 'mitchellcubic'

def _resize_images_common(images, resizer_fn, size, preserve_aspect_ratio, name, skip_resize_if_same):
    if False:
        for i in range(10):
            print('nop')
    'Core functionality for v1 and v2 resize functions.'
    with ops.name_scope(name, 'resize', [images, size]):
        images = ops.convert_to_tensor(images, name='images')
        if images.get_shape().ndims is None:
            raise ValueError("'images' contains no shape.")
        is_batch = True
        if images.get_shape().ndims == 3:
            is_batch = False
            images = array_ops.expand_dims(images, 0)
        elif images.get_shape().ndims != 4:
            raise ValueError("'images' must have either 3 or 4 dimensions.")
        (_, height, width, _) = images.get_shape().as_list()
        try:
            size = ops.convert_to_tensor(size, dtypes.int32, name='size')
        except (TypeError, ValueError):
            raise ValueError("'size' must be a 1-D int32 Tensor")
        if not size.get_shape().is_compatible_with([2]):
            raise ValueError("'size' must be a 1-D Tensor of 2 elements: new_height, new_width")
        if preserve_aspect_ratio:
            (_, current_height, current_width, _) = _ImageDimensions(images, rank=4)
            scale_factor_height = math_ops.cast(size[0], dtypes.float32) / math_ops.cast(current_height, dtypes.float32)
            scale_factor_width = math_ops.cast(size[1], dtypes.float32) / math_ops.cast(current_width, dtypes.float32)
            scale_factor = math_ops.minimum(scale_factor_height, scale_factor_width)
            scaled_height_const = math_ops.cast(math_ops.round(scale_factor * math_ops.cast(current_height, dtypes.float32)), dtypes.int32)
            scaled_width_const = math_ops.cast(math_ops.round(scale_factor * math_ops.cast(current_width, dtypes.float32)), dtypes.int32)
            size = ops.convert_to_tensor([scaled_height_const, scaled_width_const], dtypes.int32, name='size')
        size_const_as_shape = tensor_util.constant_value_as_shape(size)
        new_height_const = tensor_shape.dimension_at_index(size_const_as_shape, 0).value
        new_width_const = tensor_shape.dimension_at_index(size_const_as_shape, 1).value
        if skip_resize_if_same and all((x is not None for x in [new_width_const, width, new_height_const, height])) and (width == new_width_const and height == new_height_const):
            if not is_batch:
                images = array_ops.squeeze(images, axis=[0])
            return images
        images = resizer_fn(images, size)
        images.set_shape([None, new_height_const, new_width_const, None])
        if not is_batch:
            images = array_ops.squeeze(images, axis=[0])
        return images

@tf_export(v1=['image.resize_images', 'image.resize'])
@dispatch.add_dispatch_support
def resize_images(images, size, method=ResizeMethodV1.BILINEAR, align_corners=False, preserve_aspect_ratio=False, name=None):
    if False:
        i = 10
        return i + 15
    'Resize `images` to `size` using the specified `method`.\n\n  Resized images will be distorted if their original aspect ratio is not\n  the same as `size`.  To avoid distortions see\n  `tf.image.resize_with_pad` or `tf.image.resize_with_crop_or_pad`.\n\n  The `method` can be one of:\n\n  *   <b>`tf.image.ResizeMethod.BILINEAR`</b>: [Bilinear interpolation.](\n    https://en.wikipedia.org/wiki/Bilinear_interpolation)\n  *   <b>`tf.image.ResizeMethod.NEAREST_NEIGHBOR`</b>: [\n    Nearest neighbor interpolation.](\n    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)\n  *   <b>`tf.image.ResizeMethod.BICUBIC`</b>: [Bicubic interpolation.](\n    https://en.wikipedia.org/wiki/Bicubic_interpolation)\n  *   <b>`tf.image.ResizeMethod.AREA`</b>: Area interpolation.\n\n  The return value has the same type as `images` if `method` is\n  `tf.image.ResizeMethod.NEAREST_NEIGHBOR`. It will also have the same type\n  as `images` if the size of `images` can be statically determined to be the\n  same as `size`, because `images` is returned in this case. Otherwise, the\n  return value has type `float32`.\n\n  Args:\n    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new\n      size for the images.\n    method: ResizeMethod.  Defaults to `tf.image.ResizeMethod.BILINEAR`.\n    align_corners: bool.  If True, the centers of the 4 corner pixels of the\n      input and output tensors are aligned, preserving the values at the corner\n      pixels. Defaults to `False`.\n    preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,\n      then `images` will be resized to a size that fits in `size` while\n      preserving the aspect ratio of the original image. Scales up the image if\n      `size` is bigger than the current size of the `image`. Defaults to False.\n    name: A name for this operation (optional).\n\n  Raises:\n    ValueError: if the shape of `images` is incompatible with the\n      shape arguments to this function\n    ValueError: if `size` has invalid shape or type.\n    ValueError: if an unsupported resize method is specified.\n\n  Returns:\n    If `images` was 4-D, a 4-D float Tensor of shape\n    `[batch, new_height, new_width, channels]`.\n    If `images` was 3-D, a 3-D float Tensor of shape\n    `[new_height, new_width, channels]`.\n  '

    def resize_fn(images_t, new_size):
        if False:
            while True:
                i = 10
        'Legacy resize core function, passed to _resize_images_common.'
        if method == ResizeMethodV1.BILINEAR or method == ResizeMethod.BILINEAR:
            return gen_image_ops.resize_bilinear(images_t, new_size, align_corners=align_corners)
        elif method == ResizeMethodV1.NEAREST_NEIGHBOR or method == ResizeMethod.NEAREST_NEIGHBOR:
            return gen_image_ops.resize_nearest_neighbor(images_t, new_size, align_corners=align_corners)
        elif method == ResizeMethodV1.BICUBIC or method == ResizeMethod.BICUBIC:
            return gen_image_ops.resize_bicubic(images_t, new_size, align_corners=align_corners)
        elif method == ResizeMethodV1.AREA or method == ResizeMethod.AREA:
            return gen_image_ops.resize_area(images_t, new_size, align_corners=align_corners)
        else:
            raise ValueError('Resize method is not implemented: {}'.format(method))
    return _resize_images_common(images, resize_fn, size, preserve_aspect_ratio=preserve_aspect_ratio, name=name, skip_resize_if_same=True)

@tf_export('image.resize', v1=[])
@dispatch.add_dispatch_support
def resize_images_v2(images, size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Resize `images` to `size` using the specified `method`.\n\n  Resized images will be distorted if their original aspect ratio is not\n  the same as `size`.  To avoid distortions see\n  `tf.image.resize_with_pad`.\n\n  >>> image = tf.constant([\n  ...  [1,0,0,0,0],\n  ...  [0,1,0,0,0],\n  ...  [0,0,1,0,0],\n  ...  [0,0,0,1,0],\n  ...  [0,0,0,0,1],\n  ... ])\n  >>> # Add "batch" and "channels" dimensions\n  >>> image = image[tf.newaxis, ..., tf.newaxis]\n  >>> image.shape.as_list()  # [batch, height, width, channels]\n  [1, 5, 5, 1]\n  >>> tf.image.resize(image, [3,5])[0,...,0].numpy()\n  array([[0.6666667, 0.3333333, 0.       , 0.       , 0.       ],\n         [0.       , 0.       , 1.       , 0.       , 0.       ],\n         [0.       , 0.       , 0.       , 0.3333335, 0.6666665]],\n        dtype=float32)\n\n  It works equally well with a single image instead of a batch of images:\n\n  >>> tf.image.resize(image[0], [3,5]).shape.as_list()\n  [3, 5, 1]\n\n  When `antialias` is true, the sampling filter will anti-alias the input image\n  as well as interpolate.  When downsampling an image with [anti-aliasing](\n  https://en.wikipedia.org/wiki/Spatial_anti-aliasing) the sampling filter\n  kernel is scaled in order to properly anti-alias the input image signal.\n  `antialias` has no effect when upsampling an image:\n\n  >>> a = tf.image.resize(image, [5,10])\n  >>> b = tf.image.resize(image, [5,10], antialias=True)\n  >>> tf.reduce_max(abs(a - b)).numpy()\n  0.0\n\n  The `method` argument expects an item from the `image.ResizeMethod` enum, or\n  the string equivalent. The options are:\n\n  *   <b>`bilinear`</b>: [Bilinear interpolation.](\n    https://en.wikipedia.org/wiki/Bilinear_interpolation) If `antialias` is\n    true, becomes a hat/tent filter function with radius 1 when downsampling.\n  *   <b>`lanczos3`</b>:  [Lanczos kernel](\n    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 3.\n    High-quality practical filter but may have some ringing, especially on\n    synthetic images.\n  *   <b>`lanczos5`</b>: [Lanczos kernel] (\n    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 5.\n    Very-high-quality filter but may have stronger ringing.\n  *   <b>`bicubic`</b>: [Cubic interpolant](\n    https://en.wikipedia.org/wiki/Bicubic_interpolation) of Keys. Equivalent to\n    Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel,\n    particularly when upsampling.\n  *   <b>`gaussian`</b>: [Gaussian kernel](\n    https://en.wikipedia.org/wiki/Gaussian_filter) with radius 3,\n    sigma = 1.5 / 3.0.\n  *   <b>`nearest`</b>: [Nearest neighbor interpolation.](\n    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)\n    `antialias` has no effect when used with nearest neighbor interpolation.\n  *   <b>`area`</b>: Anti-aliased resampling with area interpolation.\n    `antialias` has no effect when used with area interpolation; it\n    always anti-aliases.\n  *   <b>`mitchellcubic`</b>: Mitchell-Netravali Cubic non-interpolating filter.\n    For synthetic images (especially those lacking proper prefiltering), less\n    ringing than Keys cubic kernel but less sharp.\n\n  Note: Near image edges the filtering kernel may be partially outside the\n  image boundaries. For these pixels, only input pixels inside the image will be\n  included in the filter sum, and the output value will be appropriately\n  normalized.\n\n  The return value has type `float32`, unless the `method` is\n  `ResizeMethod.NEAREST_NEIGHBOR`, then the return dtype is the dtype\n  of `images`:\n\n  >>> nn = tf.image.resize(image, [5,7], method=\'nearest\')\n  >>> nn[0,...,0].numpy()\n  array([[1, 0, 0, 0, 0, 0, 0],\n         [0, 1, 1, 0, 0, 0, 0],\n         [0, 0, 0, 1, 0, 0, 0],\n         [0, 0, 0, 0, 1, 1, 0],\n         [0, 0, 0, 0, 0, 0, 1]], dtype=int32)\n\n  With `preserve_aspect_ratio=True`, the aspect ratio is preserved, so `size`\n  is the maximum for each dimension:\n\n  >>> max_10_20 = tf.image.resize(image, [10,20], preserve_aspect_ratio=True)\n  >>> max_10_20.shape.as_list()\n  [1, 10, 10, 1]\n\n  Args:\n    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new\n      size for the images.\n    method: An `image.ResizeMethod`, or string equivalent.  Defaults to\n      `bilinear`.\n    preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,\n      then `images` will be resized to a size that fits in `size` while\n      preserving the aspect ratio of the original image. Scales up the image if\n      `size` is bigger than the current size of the `image`. Defaults to False.\n    antialias: Whether to use an anti-aliasing filter when downsampling an\n      image.\n    name: A name for this operation (optional).\n\n  Raises:\n    ValueError: if the shape of `images` is incompatible with the\n      shape arguments to this function\n    ValueError: if `size` has an invalid shape or type.\n    ValueError: if an unsupported resize method is specified.\n\n  Returns:\n    If `images` was 4-D, a 4-D float Tensor of shape\n    `[batch, new_height, new_width, channels]`.\n    If `images` was 3-D, a 3-D float Tensor of shape\n    `[new_height, new_width, channels]`.\n  '

    def resize_fn(images_t, new_size):
        if False:
            return 10
        'Resize core function, passed to _resize_images_common.'
        scale_and_translate_methods = [ResizeMethod.LANCZOS3, ResizeMethod.LANCZOS5, ResizeMethod.GAUSSIAN, ResizeMethod.MITCHELLCUBIC]

        def resize_with_scale_and_translate(method):
            if False:
                return 10
            scale = math_ops.cast(new_size, dtype=dtypes.float32) / math_ops.cast(array_ops.shape(images_t)[1:3], dtype=dtypes.float32)
            return gen_image_ops.scale_and_translate(images_t, new_size, scale, array_ops.zeros([2]), kernel_type=method, antialias=antialias)
        if method == ResizeMethod.BILINEAR:
            if antialias:
                return resize_with_scale_and_translate('triangle')
            else:
                return gen_image_ops.resize_bilinear(images_t, new_size, half_pixel_centers=True)
        elif method == ResizeMethod.NEAREST_NEIGHBOR:
            return gen_image_ops.resize_nearest_neighbor(images_t, new_size, half_pixel_centers=True)
        elif method == ResizeMethod.BICUBIC:
            if antialias:
                return resize_with_scale_and_translate('keyscubic')
            else:
                return gen_image_ops.resize_bicubic(images_t, new_size, half_pixel_centers=True)
        elif method == ResizeMethod.AREA:
            return gen_image_ops.resize_area(images_t, new_size)
        elif method in scale_and_translate_methods:
            return resize_with_scale_and_translate(method)
        else:
            raise ValueError('Resize method is not implemented: {}'.format(method))
    return _resize_images_common(images, resize_fn, size, preserve_aspect_ratio=preserve_aspect_ratio, name=name, skip_resize_if_same=False)

def _resize_image_with_pad_common(image, target_height, target_width, resize_fn):
    if False:
        return 10
    'Core functionality for v1 and v2 resize_image_with_pad functions.'
    with ops.name_scope(None, 'resize_image_with_pad', [image]):
        image = ops.convert_to_tensor(image, name='image')
        image_shape = image.get_shape()
        is_batch = True
        if image_shape.ndims == 3:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image_shape.ndims != 4:
            raise ValueError("'image' (shape %s) must have either 3 or 4 dimensions." % image_shape)
        assert_ops = _CheckAtLeast3DImage(image, require_static=False)
        assert_ops += _assert(target_width > 0, ValueError, 'target_width must be > 0.')
        assert_ops += _assert(target_height > 0, ValueError, 'target_height must be > 0.')
        image = control_flow_ops.with_dependencies(assert_ops, image)

        def max_(x, y):
            if False:
                print('Hello World!')
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.maximum(x, y)
            else:
                return max(x, y)
        (_, height, width, _) = _ImageDimensions(image, rank=4)
        f_height = math_ops.cast(height, dtype=dtypes.float32)
        f_width = math_ops.cast(width, dtype=dtypes.float32)
        f_target_height = math_ops.cast(target_height, dtype=dtypes.float32)
        f_target_width = math_ops.cast(target_width, dtype=dtypes.float32)
        ratio = max_(f_width / f_target_width, f_height / f_target_height)
        resized_height_float = f_height / ratio
        resized_width_float = f_width / ratio
        resized_height = math_ops.cast(math_ops.floor(resized_height_float), dtype=dtypes.int32)
        resized_width = math_ops.cast(math_ops.floor(resized_width_float), dtype=dtypes.int32)
        padding_height = (f_target_height - resized_height_float) / 2
        padding_width = (f_target_width - resized_width_float) / 2
        f_padding_height = math_ops.floor(padding_height)
        f_padding_width = math_ops.floor(padding_width)
        p_height = max_(0, math_ops.cast(f_padding_height, dtype=dtypes.int32))
        p_width = max_(0, math_ops.cast(f_padding_width, dtype=dtypes.int32))
        resized = resize_fn(image, [resized_height, resized_width])
        padded = pad_to_bounding_box(resized, p_height, p_width, target_height, target_width)
        if padded.get_shape().ndims is None:
            raise ValueError('padded contains no shape.')
        _ImageDimensions(padded, rank=4)
        if not is_batch:
            padded = array_ops.squeeze(padded, axis=[0])
        return padded

@tf_export(v1=['image.resize_image_with_pad'])
@dispatch.add_dispatch_support
def resize_image_with_pad_v1(image, target_height, target_width, method=ResizeMethodV1.BILINEAR, align_corners=False):
    if False:
        while True:
            i = 10
    "Resizes and pads an image to a target width and height.\n\n  Resizes an image to a target width and height by keeping\n  the aspect ratio the same without distortion. If the target\n  dimensions don't match the image dimensions, the image\n  is resized and then padded with zeroes to match requested\n  dimensions.\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    target_height: Target height.\n    target_width: Target width.\n    method: Method to use for resizing image. See `resize_images()`\n    align_corners: bool.  If True, the centers of the 4 corner pixels of the\n      input and output tensors are aligned, preserving the values at the corner\n      pixels. Defaults to `False`.\n\n  Raises:\n    ValueError: if `target_height` or `target_width` are zero or negative.\n\n  Returns:\n    Resized and padded image.\n    If `images` was 4-D, a 4-D float Tensor of shape\n    `[batch, new_height, new_width, channels]`.\n    If `images` was 3-D, a 3-D float Tensor of shape\n    `[new_height, new_width, channels]`.\n  "

    def _resize_fn(im, new_size):
        if False:
            for i in range(10):
                print('nop')
        return resize_images(im, new_size, method, align_corners=align_corners)
    return _resize_image_with_pad_common(image, target_height, target_width, _resize_fn)

@tf_export('image.resize_with_pad', v1=[])
@dispatch.add_dispatch_support
def resize_image_with_pad_v2(image, target_height, target_width, method=ResizeMethod.BILINEAR, antialias=False):
    if False:
        i = 10
        return i + 15
    "Resizes and pads an image to a target width and height.\n\n  Resizes an image to a target width and height by keeping\n  the aspect ratio the same without distortion. If the target\n  dimensions don't match the image dimensions, the image\n  is resized and then padded with zeroes to match requested\n  dimensions.\n\n  Args:\n    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    target_height: Target height.\n    target_width: Target width.\n    method: Method to use for resizing image. See `image.resize()`\n    antialias: Whether to use anti-aliasing when resizing. See 'image.resize()'.\n\n  Raises:\n    ValueError: if `target_height` or `target_width` are zero or negative.\n\n  Returns:\n    Resized and padded image.\n    If `images` was 4-D, a 4-D float Tensor of shape\n    `[batch, new_height, new_width, channels]`.\n    If `images` was 3-D, a 3-D float Tensor of shape\n    `[new_height, new_width, channels]`.\n  "

    def _resize_fn(im, new_size):
        if False:
            print('Hello World!')
        return resize_images_v2(im, new_size, method, antialias=antialias)
    return _resize_image_with_pad_common(image, target_height, target_width, _resize_fn)

@tf_export('image.per_image_standardization')
@dispatch.add_dispatch_support
def per_image_standardization(image):
    if False:
        return 10
    'Linearly scales each image in `image` to have mean 0 and variance 1.\n\n  For each 3-D image `x` in `image`, computes `(x - mean) / adjusted_stddev`,\n  where\n\n  - `mean` is the average of all values in `x`\n  - `adjusted_stddev = max(stddev, 1.0/sqrt(N))` is capped away from 0 to\n    protect against division by 0 when handling uniform images\n    - `N` is the number of elements in `x`\n    - `stddev` is the standard deviation of all values in `x`\n\n  Example Usage:\n\n  >>> image = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])\n  >>> image # 3-D tensor\n  <tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=\n  array([[[ 1,  2,  3],\n          [ 4,  5,  6]],\n         [[ 7,  8,  9],\n          [10, 11, 12]]], dtype=int32)>\n  >>> new_image = tf.image.per_image_standardization(image)\n  >>> new_image # 3-D tensor with mean ~= 0 and variance ~= 1\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[-1.593255  , -1.3035723 , -1.0138896 ],\n          [-0.7242068 , -0.4345241 , -0.14484136]],\n         [[ 0.14484136,  0.4345241 ,  0.7242068 ],\n          [ 1.0138896 ,  1.3035723 ,  1.593255  ]]], dtype=float32)>\n\n  Args:\n    image: An n-D `Tensor` with at least 3 dimensions, the last 3 of which are\n      the dimensions of each image.\n\n  Returns:\n    A `Tensor` with the same shape as `image` and its dtype is `float32`.\n\n  Raises:\n    ValueError: The shape of `image` has fewer than 3 dimensions.\n  '
    with ops.name_scope(None, 'per_image_standardization', [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        image = _AssertAtLeast3DImage(image)
        image = math_ops.cast(image, dtype=dtypes.float32)
        num_pixels = math_ops.reduce_prod(array_ops.shape(image)[-3:])
        image_mean = math_ops.reduce_mean(image, axis=[-1, -2, -3], keepdims=True)
        stddev = math_ops.reduce_std(image, axis=[-1, -2, -3], keepdims=True)
        min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
        adjusted_stddev = math_ops.maximum(stddev, min_stddev)
        image -= image_mean
        image = math_ops.divide(image, adjusted_stddev, name=scope)
        return image

@tf_export('image.random_brightness')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def random_brightness(image, max_delta, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Adjust the brightness of images by a random factor.\n\n  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the\n  interval `[-max_delta, max_delta)`.\n\n  For producing deterministic results given a `seed` value, use\n  `tf.image.stateless_random_brightness`. Unlike using the `seed` param\n  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the\n  same results given the same seed independent of how many times the function is\n  called, and independent of global seed settings (e.g. tf.random.set_seed).\n\n  Args:\n    image: An image or images to adjust.\n    max_delta: float, must be non-negative.\n    seed: A Python integer. Used to create a random seed. See\n      `tf.compat.v1.set_random_seed` for behavior.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...      [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.random_brightness(x, 0.2)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>\n\n  Returns:\n    The brightness-adjusted image(s).\n\n  Raises:\n    ValueError: if `max_delta` is negative.\n  '
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')
    delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
    return adjust_brightness(image, delta)

@tf_export('image.stateless_random_brightness', v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def stateless_random_brightness(image, max_delta, seed):
    if False:
        while True:
            i = 10
    'Adjust the brightness of images by a random factor deterministically.\n\n  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the\n  interval `[-max_delta, max_delta)`.\n\n  Guarantees the same results given the same `seed` independent of how many\n  times the function is called, and independent of global seed settings (e.g.\n  `tf.random.set_seed`).\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...      [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> seed = (1, 2)\n  >>> tf.image.stateless_random_brightness(x, 0.2, seed)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 1.1376241,  2.1376243,  3.1376243],\n          [ 4.1376243,  5.1376243,  6.1376243]],\n         [[ 7.1376243,  8.137624 ,  9.137624 ],\n          [10.137624 , 11.137624 , 12.137624 ]]], dtype=float32)>\n\n  Args:\n    image: An image or images to adjust.\n    max_delta: float, must be non-negative.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n\n  Returns:\n    The brightness-adjusted image(s).\n\n  Raises:\n    ValueError: if `max_delta` is negative.\n  '
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')
    delta = stateless_random_ops.stateless_random_uniform(shape=[], minval=-max_delta, maxval=max_delta, seed=seed)
    return adjust_brightness(image, delta)

@tf_export('image.random_contrast')
@dispatch.add_dispatch_support
def random_contrast(image, lower, upper, seed=None):
    if False:
        print('Hello World!')
    'Adjust the contrast of an image or images by a random factor.\n\n  Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly\n  picked in the interval `[lower, upper)`.\n\n  For producing deterministic results given a `seed` value, use\n  `tf.image.stateless_random_contrast`. Unlike using the `seed` param\n  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the\n  same results given the same seed independent of how many times the function is\n  called, and independent of global seed settings (e.g. tf.random.set_seed).\n\n  Args:\n    image: An image tensor with 3 or more dimensions.\n    lower: float.  Lower bound for the random contrast factor.\n    upper: float.  Upper bound for the random contrast factor.\n    seed: A Python integer. Used to create a random seed. See\n      `tf.compat.v1.set_random_seed` for behavior.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.random_contrast(x, 0.2, 0.5)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>\n\n  Returns:\n    The contrast-adjusted image(s).\n\n  Raises:\n    ValueError: if `upper <= lower` or if `lower < 0`.\n  '
    if upper <= lower:
        raise ValueError('upper must be > lower.')
    if lower < 0:
        raise ValueError('lower must be non-negative.')
    contrast_factor = random_ops.random_uniform([], lower, upper, seed=seed)
    return adjust_contrast(image, contrast_factor)

@tf_export('image.stateless_random_contrast', v1=[])
@dispatch.add_dispatch_support
def stateless_random_contrast(image, lower, upper, seed):
    if False:
        for i in range(10):
            print('nop')
    'Adjust the contrast of images by a random factor deterministically.\n\n  Guarantees the same results given the same `seed` independent of how many\n  times the function is called, and independent of global seed settings (e.g.\n  `tf.random.set_seed`).\n\n  Args:\n    image: An image tensor with 3 or more dimensions.\n    lower: float.  Lower bound for the random contrast factor.\n    upper: float.  Upper bound for the random contrast factor.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...      [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> seed = (1, 2)\n  >>> tf.image.stateless_random_contrast(x, 0.2, 0.5, seed)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[3.4605184, 4.4605184, 5.4605184],\n          [4.820173 , 5.820173 , 6.820173 ]],\n         [[6.179827 , 7.179827 , 8.179828 ],\n          [7.5394816, 8.539482 , 9.539482 ]]], dtype=float32)>\n\n  Returns:\n    The contrast-adjusted image(s).\n\n  Raises:\n    ValueError: if `upper <= lower` or if `lower < 0`.\n  '
    if upper <= lower:
        raise ValueError('upper must be > lower.')
    if lower < 0:
        raise ValueError('lower must be non-negative.')
    contrast_factor = stateless_random_ops.stateless_random_uniform(shape=[], minval=lower, maxval=upper, seed=seed)
    return adjust_contrast(image, contrast_factor)

@tf_export('image.adjust_brightness')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def adjust_brightness(image, delta):
    if False:
        print('Hello World!')
    'Adjust the brightness of RGB or Grayscale images.\n\n  This is a convenience method that converts RGB images to float\n  representation, adjusts their brightness, and then converts them back to the\n  original data type. If several adjustments are chained, it is advisable to\n  minimize the number of redundant conversions.\n\n  The value `delta` is added to all components of the tensor `image`. `image` is\n  converted to `float` and scaled appropriately if it is in fixed-point\n  representation, and `delta` is converted to the same data type. For regular\n  images, `delta` should be in the range `(-1,1)`, as it is added to the image\n  in floating point representation, where pixel values are in the `[0,1)` range.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.adjust_brightness(x, delta=0.1)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 1.1,  2.1,  3.1],\n          [ 4.1,  5.1,  6.1]],\n         [[ 7.1,  8.1,  9.1],\n          [10.1, 11.1, 12.1]]], dtype=float32)>\n\n  Args:\n    image: RGB image or images to adjust.\n    delta: A scalar. Amount to add to the pixel values.\n\n  Returns:\n    A brightness-adjusted tensor of the same shape and type as `image`.\n  '
    with ops.name_scope(None, 'adjust_brightness', [image, delta]) as name:
        image = ops.convert_to_tensor(image, name='image')
        orig_dtype = image.dtype
        if orig_dtype in [dtypes.float16, dtypes.float32]:
            flt_image = image
        else:
            flt_image = convert_image_dtype(image, dtypes.float32)
        adjusted = math_ops.add(flt_image, math_ops.cast(delta, flt_image.dtype), name=name)
        return convert_image_dtype(adjusted, orig_dtype, saturate=True)

@tf_export('image.adjust_contrast')
@dispatch.add_dispatch_support
def adjust_contrast(images, contrast_factor):
    if False:
        for i in range(10):
            print('nop')
    'Adjust contrast of RGB or grayscale images.\n\n  This is a convenience method that converts RGB images to float\n  representation, adjusts their contrast, and then converts them back to the\n  original data type. If several adjustments are chained, it is advisable to\n  minimize the number of redundant conversions.\n\n  `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are\n  interpreted as `[height, width, channels]`.  The other dimensions only\n  represent a collection of images, such as `[batch, height, width, channels].`\n\n  Contrast is adjusted independently for each channel of each image.\n\n  For each channel, this Op computes the mean of the image pixels in the\n  channel and then adjusts each component `x` of each pixel to\n  `(x - mean) * contrast_factor + mean`.\n\n  `contrast_factor` must be in the interval `(-inf, inf)`.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.adjust_contrast(x, 2.)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[-3.5, -2.5, -1.5],\n          [ 2.5,  3.5,  4.5]],\n         [[ 8.5,  9.5, 10.5],\n          [14.5, 15.5, 16.5]]], dtype=float32)>\n\n  Args:\n    images: Images to adjust.  At least 3-D.\n    contrast_factor: A float multiplier for adjusting contrast.\n\n  Returns:\n    The contrast-adjusted image or images.\n  '
    with ops.name_scope(None, 'adjust_contrast', [images, contrast_factor]) as name:
        images = ops.convert_to_tensor(images, name='images')
        orig_dtype = images.dtype
        if orig_dtype in (dtypes.float16, dtypes.float32):
            flt_images = images
        else:
            flt_images = convert_image_dtype(images, dtypes.float32)
        adjusted = gen_image_ops.adjust_contrastv2(flt_images, contrast_factor=contrast_factor, name=name)
        return convert_image_dtype(adjusted, orig_dtype, saturate=True)

@tf_export('image.adjust_gamma')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def adjust_gamma(image, gamma=1, gain=1):
    if False:
        for i in range(10):
            print('nop')
    'Performs [Gamma Correction](http://en.wikipedia.org/wiki/Gamma_correction).\n\n  on the input image.\n\n  Also known as Power Law Transform. This function converts the\n  input images at first to float representation, then transforms them\n  pixelwise according to the equation `Out = gain * In**gamma`,\n  and then converts the back to the original data type.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.adjust_gamma(x, 0.2)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[1.       , 1.1486983, 1.2457309],\n          [1.319508 , 1.3797297, 1.4309691]],\n         [[1.4757731, 1.5157166, 1.5518456],\n          [1.5848932, 1.6153942, 1.6437519]]], dtype=float32)>\n\n  Args:\n    image : RGB image or images to adjust.\n    gamma : A scalar or tensor. Non-negative real number.\n    gain  : A scalar or tensor. The constant multiplier.\n\n  Returns:\n    A Tensor. A Gamma-adjusted tensor of the same shape and type as `image`.\n\n  Raises:\n    ValueError: If gamma is negative.\n  Notes:\n    For gamma greater than 1, the histogram will shift towards left and\n    the output image will be darker than the input image.\n    For gamma less than 1, the histogram will shift towards right and\n    the output image will be brighter than the input image.\n  References:\n    [Wikipedia](http://en.wikipedia.org/wiki/Gamma_correction)\n  '
    with ops.name_scope(None, 'adjust_gamma', [image, gamma, gain]) as name:
        image = ops.convert_to_tensor(image, name='image')
        orig_dtype = image.dtype
        if orig_dtype in [dtypes.float16, dtypes.float32]:
            flt_image = image
        else:
            flt_image = convert_image_dtype(image, dtypes.float32)
        assert_op = _assert(gamma >= 0, ValueError, 'Gamma should be a non-negative real number.')
        if assert_op:
            gamma = control_flow_ops.with_dependencies(assert_op, gamma)
        adjusted_img = gain * flt_image ** gamma
        return convert_image_dtype(adjusted_img, orig_dtype, saturate=True)

@tf_export('image.convert_image_dtype')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def convert_image_dtype(image, dtype, saturate=False, name=None):
    if False:
        return 10
    "Convert `image` to `dtype`, scaling its values if needed.\n\n  The operation supports data types (for `image` and `dtype`) of\n  `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`,\n  `float16`, `float32`, `float64`, `bfloat16`.\n\n  Images that are represented using floating point values are expected to have\n  values in the range [0,1). Image data stored in integer data types are\n  expected to have values in the range `[0,MAX]`, where `MAX` is the largest\n  positive representable number for the data type.\n\n  This op converts between data types, scaling the values appropriately before\n  casting.\n\n  Usage Example:\n\n  >>> x = [[[1, 2, 3], [4, 5, 6]],\n  ...      [[7, 8, 9], [10, 11, 12]]]\n  >>> x_int8 = tf.convert_to_tensor(x, dtype=tf.int8)\n  >>> tf.image.convert_image_dtype(x_int8, dtype=tf.float16, saturate=False)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float16, numpy=\n  array([[[0.00787, 0.01575, 0.02362],\n          [0.0315 , 0.03937, 0.04724]],\n         [[0.0551 , 0.063  , 0.07086],\n          [0.07874, 0.0866 , 0.0945 ]]], dtype=float16)>\n\n  Converting integer types to floating point types returns normalized floating\n  point values in the range [0, 1); the values are normalized by the `MAX` value\n  of the input dtype. Consider the following two examples:\n\n  >>> a = [[[1], [2]], [[3], [4]]]\n  >>> a_int8 = tf.convert_to_tensor(a, dtype=tf.int8)\n  >>> tf.image.convert_image_dtype(a_int8, dtype=tf.float32)\n  <tf.Tensor: shape=(2, 2, 1), dtype=float32, numpy=\n  array([[[0.00787402],\n          [0.01574803]],\n         [[0.02362205],\n          [0.03149606]]], dtype=float32)>\n\n  >>> a_int32 = tf.convert_to_tensor(a, dtype=tf.int32)\n  >>> tf.image.convert_image_dtype(a_int32, dtype=tf.float32)\n  <tf.Tensor: shape=(2, 2, 1), dtype=float32, numpy=\n  array([[[4.6566129e-10],\n          [9.3132257e-10]],\n         [[1.3969839e-09],\n          [1.8626451e-09]]], dtype=float32)>\n\n  Despite having identical values of `a` and output dtype of `float32`, the\n  outputs differ due to the different input dtypes (`int8` vs. `int32`). This\n  is, again, because the values are normalized by the `MAX` value of the input\n  dtype.\n\n  Note that converting floating point values to integer type may lose precision.\n  In the example below, an image tensor `b` of dtype `float32` is converted to\n  `int8` and back to `float32`. The final output, however, is different from\n  the original input `b` due to precision loss.\n\n  >>> b = [[[0.12], [0.34]], [[0.56], [0.78]]]\n  >>> b_float32 = tf.convert_to_tensor(b, dtype=tf.float32)\n  >>> b_int8 = tf.image.convert_image_dtype(b_float32, dtype=tf.int8)\n  >>> tf.image.convert_image_dtype(b_int8, dtype=tf.float32)\n  <tf.Tensor: shape=(2, 2, 1), dtype=float32, numpy=\n  array([[[0.11811024],\n          [0.33858266]],\n         [[0.5590551 ],\n          [0.77952754]]], dtype=float32)>\n\n  Scaling up from an integer type (input dtype) to another integer type (output\n  dtype) will not map input dtype's `MAX` to output dtype's `MAX` but converting\n  back and forth should result in no change. For example, as shown below, the\n  `MAX` value of int8 (=127) is not mapped to the `MAX` value of int16 (=32,767)\n  but, when scaled back, we get the same, original values of `c`.\n\n  >>> c = [[[1], [2]], [[127], [127]]]\n  >>> c_int8 = tf.convert_to_tensor(c, dtype=tf.int8)\n  >>> c_int16 = tf.image.convert_image_dtype(c_int8, dtype=tf.int16)\n  >>> print(c_int16)\n  tf.Tensor(\n  [[[  256]\n    [  512]]\n   [[32512]\n    [32512]]], shape=(2, 2, 1), dtype=int16)\n  >>> c_int8_back = tf.image.convert_image_dtype(c_int16, dtype=tf.int8)\n  >>> print(c_int8_back)\n  tf.Tensor(\n  [[[  1]\n    [  2]]\n   [[127]\n    [127]]], shape=(2, 2, 1), dtype=int8)\n\n  Scaling down from an integer type to another integer type can be a lossy\n  conversion. Notice in the example below that converting `int16` to `uint8` and\n  back to `int16` has lost precision.\n\n  >>> d = [[[1000], [2000]], [[3000], [4000]]]\n  >>> d_int16 = tf.convert_to_tensor(d, dtype=tf.int16)\n  >>> d_uint8 = tf.image.convert_image_dtype(d_int16, dtype=tf.uint8)\n  >>> d_int16_back = tf.image.convert_image_dtype(d_uint8, dtype=tf.int16)\n  >>> print(d_int16_back)\n  tf.Tensor(\n  [[[ 896]\n    [1920]]\n   [[2944]\n    [3968]]], shape=(2, 2, 1), dtype=int16)\n\n  Note that converting from floating point inputs to integer types may lead to\n  over/underflow problems. Set saturate to `True` to avoid such problem in\n  problematic conversions. If enabled, saturation will clip the output into the\n  allowed range before performing a potentially dangerous cast (and only before\n  performing such a cast, i.e., when casting from a floating point to an integer\n  type, and when casting from a signed to an unsigned type; `saturate` has no\n  effect on casts between floats, or on casts that increase the type's range).\n\n  Args:\n    image: An image.\n    dtype: A `DType` to convert `image` to.\n    saturate: If `True`, clip the input before casting (if necessary).\n    name: A name for this operation (optional).\n\n  Returns:\n    `image`, converted to `dtype`.\n\n  Raises:\n    AttributeError: Raises an attribute error when dtype is neither\n    float nor integer.\n  "
    image = ops.convert_to_tensor(image, name='image')
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_floating and (not dtype.is_integer):
        raise AttributeError('dtype must be either floating point or integer')
    if not image.dtype.is_floating and (not image.dtype.is_integer):
        raise AttributeError('image dtype must be either floating point or integer')
    if dtype == image.dtype:
        return array_ops.identity(image, name=name)
    with ops.name_scope(name, 'convert_image', [image]) as name:
        if image.dtype.is_integer and dtype.is_integer:
            scale_in = image.dtype.max
            scale_out = dtype.max
            if scale_in > scale_out:
                scale = (scale_in + 1) // (scale_out + 1)
                scaled = math_ops.floordiv(image, scale)
                if saturate:
                    return math_ops.saturate_cast(scaled, dtype, name=name)
                else:
                    return math_ops.cast(scaled, dtype, name=name)
            else:
                if saturate:
                    cast = math_ops.saturate_cast(image, dtype)
                else:
                    cast = math_ops.cast(image, dtype)
                scale = (scale_out + 1) // (scale_in + 1)
                return math_ops.multiply(cast, scale, name=name)
        elif image.dtype.is_floating and dtype.is_floating:
            return math_ops.cast(image, dtype, name=name)
        elif image.dtype.is_integer:
            cast = math_ops.cast(image, dtype)
            scale = 1.0 / image.dtype.max
            return math_ops.multiply(cast, scale, name=name)
        else:
            scale = dtype.max + 0.5
            scaled = math_ops.multiply(image, scale)
            if saturate:
                return math_ops.saturate_cast(scaled, dtype, name=name)
            else:
                return math_ops.cast(scaled, dtype, name=name)

@tf_export('image.rgb_to_grayscale')
@dispatch.add_dispatch_support
def rgb_to_grayscale(images, name=None):
    if False:
        print('Hello World!')
    'Converts one or more images from RGB to Grayscale.\n\n  Outputs a tensor of the same `DType` and rank as `images`.  The size of the\n  last dimension of the output is 1, containing the Grayscale value of the\n  pixels.\n\n  >>> original = tf.constant([[[1.0, 2.0, 3.0]]])\n  >>> converted = tf.image.rgb_to_grayscale(original)\n  >>> print(converted.numpy())\n  [[[1.81...]]]\n\n  Args:\n    images: The RGB tensor to convert. The last dimension must have size 3 and\n      should contain RGB values.\n    name: A name for the operation (optional).\n\n  Returns:\n    The converted grayscale image(s).\n  '
    with ops.name_scope(name, 'rgb_to_grayscale', [images]) as name:
        images = ops.convert_to_tensor(images, name='images')
        orig_dtype = images.dtype
        flt_image = convert_image_dtype(images, dtypes.float32)
        rgb_weights = [0.2989, 0.587, 0.114]
        gray_float = math_ops.tensordot(flt_image, rgb_weights, [-1, -1])
        gray_float = array_ops.expand_dims(gray_float, -1)
        return convert_image_dtype(gray_float, orig_dtype, name=name)

@tf_export('image.grayscale_to_rgb')
@dispatch.add_dispatch_support
def grayscale_to_rgb(images, name=None):
    if False:
        i = 10
        return i + 15
    "Converts one or more images from Grayscale to RGB.\n\n  Outputs a tensor of the same `DType` and rank as `images`.  The size of the\n  last dimension of the output is 3, containing the RGB value of the pixels.\n  The input images' last dimension must be size 1.\n\n  >>> original = tf.constant([[[1.0], [2.0], [3.0]]])\n  >>> converted = tf.image.grayscale_to_rgb(original)\n  >>> print(converted.numpy())\n  [[[1. 1. 1.]\n    [2. 2. 2.]\n    [3. 3. 3.]]]\n\n  Args:\n    images: The Grayscale tensor to convert. The last dimension must be size 1.\n    name: A name for the operation (optional).\n\n  Returns:\n    The converted grayscale image(s).\n  "
    with ops.name_scope(name, 'grayscale_to_rgb', [images]) as name:
        images = _AssertGrayscaleImage(images)
        images = ops.convert_to_tensor(images, name='images')
        rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0)
        shape_list = [array_ops.ones(rank_1, dtype=dtypes.int32)] + [array_ops.expand_dims(3, 0)]
        multiples = array_ops.concat(shape_list, 0)
        rgb = array_ops.tile(images, multiples, name=name)
        rgb.set_shape(images.get_shape()[:-1].concatenate([3]))
        return rgb

@tf_export('image.random_hue')
@dispatch.add_dispatch_support
def random_hue(image, max_delta, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Adjust the hue of RGB images by a random factor.\n\n  Equivalent to `adjust_hue()` but uses a `delta` randomly\n  picked in the interval `[-max_delta, max_delta)`.\n\n  `max_delta` must be in the interval `[0, 0.5]`.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.random_hue(x, 0.2)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>\n\n  For producing deterministic results given a `seed` value, use\n  `tf.image.stateless_random_hue`. Unlike using the `seed` param with\n  `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the same\n  results given the same seed independent of how many times the function is\n  called, and independent of global seed settings (e.g. tf.random.set_seed).\n\n  Args:\n    image: RGB image or images. The size of the last dimension must be 3.\n    max_delta: float. The maximum value for the random delta.\n    seed: An operation-specific seed. It will be used in conjunction with the\n      graph-level seed to determine the real seeds that will be used in this\n      operation. Please see the documentation of set_random_seed for its\n      interaction with the graph-level random seed.\n\n  Returns:\n    Adjusted image(s), same shape and DType as `image`.\n\n  Raises:\n    ValueError: if `max_delta` is invalid.\n  '
    if max_delta > 0.5:
        raise ValueError('max_delta must be <= 0.5.')
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')
    delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
    return adjust_hue(image, delta)

@tf_export('image.stateless_random_hue', v1=[])
@dispatch.add_dispatch_support
def stateless_random_hue(image, max_delta, seed):
    if False:
        print('Hello World!')
    'Adjust the hue of RGB images by a random factor deterministically.\n\n  Equivalent to `adjust_hue()` but uses a `delta` randomly picked in the\n  interval `[-max_delta, max_delta)`.\n\n  Guarantees the same results given the same `seed` independent of how many\n  times the function is called, and independent of global seed settings (e.g.\n  `tf.random.set_seed`).\n\n  `max_delta` must be in the interval `[0, 0.5]`.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...      [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> seed = (1, 2)\n  >>> tf.image.stateless_random_hue(x, 0.2, seed)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 1.6514902,  1.       ,  3.       ],\n          [ 4.65149  ,  4.       ,  6.       ]],\n         [[ 7.65149  ,  7.       ,  9.       ],\n          [10.65149  , 10.       , 12.       ]]], dtype=float32)>\n\n  Args:\n    image: RGB image or images. The size of the last dimension must be 3.\n    max_delta: float. The maximum value for the random delta.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`.\n\n  Returns:\n    Adjusted image(s), same shape and DType as `image`.\n\n  Raises:\n    ValueError: if `max_delta` is invalid.\n  '
    if max_delta > 0.5:
        raise ValueError('max_delta must be <= 0.5.')
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')
    delta = stateless_random_ops.stateless_random_uniform(shape=[], minval=-max_delta, maxval=max_delta, seed=seed)
    return adjust_hue(image, delta)

@tf_export('image.adjust_hue')
@dispatch.add_dispatch_support
def adjust_hue(image, delta, name=None):
    if False:
        i = 10
        return i + 15
    'Adjust hue of RGB images.\n\n  This is a convenience method that converts an RGB image to float\n  representation, converts it to HSV, adds an offset to the\n  hue channel, converts back to RGB and then back to the original\n  data type. If several adjustments are chained it is advisable to minimize\n  the number of redundant conversions.\n\n  `image` is an RGB image.  The image hue is adjusted by converting the\n  image(s) to HSV and rotating the hue channel (H) by\n  `delta`.  The image is then converted back to RGB.\n\n  `delta` must be in the interval `[-1, 1]`.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.adjust_hue(x, 0.2)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 2.3999996,  1.       ,  3.       ],\n          [ 5.3999996,  4.       ,  6.       ]],\n        [[ 8.4      ,  7.       ,  9.       ],\n          [11.4      , 10.       , 12.       ]]], dtype=float32)>\n\n  Args:\n    image: RGB image or images. The size of the last dimension must be 3.\n    delta: float.  How much to add to the hue channel.\n    name: A name for this operation (optional).\n\n  Returns:\n    Adjusted image(s), same shape and DType as `image`.\n\n  Raises:\n    InvalidArgumentError: image must have at least 3 dimensions.\n    InvalidArgumentError: The size of the last dimension must be 3.\n    ValueError: if `delta` is not in the interval of `[-1, 1]`.\n\n  Usage Example:\n\n  >>> image = [[[1, 2, 3], [4, 5, 6]],\n  ...          [[7, 8, 9], [10, 11, 12]],\n  ...          [[13, 14, 15], [16, 17, 18]]]\n  >>> image = tf.constant(image)\n  >>> tf.image.adjust_hue(image, 0.2)\n  <tf.Tensor: shape=(3, 2, 3), dtype=int32, numpy=\n  array([[[ 2,  1,  3],\n        [ 5,  4,  6]],\n       [[ 8,  7,  9],\n        [11, 10, 12]],\n       [[14, 13, 15],\n        [17, 16, 18]]], dtype=int32)>\n  '
    with ops.name_scope(name, 'adjust_hue', [image]) as name:
        if context.executing_eagerly():
            if delta < -1 or delta > 1:
                raise ValueError('delta must be in the interval [-1, 1]')
        image = ops.convert_to_tensor(image, name='image')
        orig_dtype = image.dtype
        if orig_dtype in (dtypes.float16, dtypes.float32):
            flt_image = image
        else:
            flt_image = convert_image_dtype(image, dtypes.float32)
        rgb_altered = gen_image_ops.adjust_hue(flt_image, delta)
        return convert_image_dtype(rgb_altered, orig_dtype)

@tf_export('image.random_jpeg_quality')
@dispatch.add_dispatch_support
def random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Randomly changes jpeg encoding quality for inducing jpeg noise.\n\n  `min_jpeg_quality` must be in the interval `[0, 100]` and less than\n  `max_jpeg_quality`.\n  `max_jpeg_quality` must be in the interval `[0, 100]`.\n\n  Usage Example:\n\n  >>> x = tf.constant([[[1, 2, 3],\n  ...                   [4, 5, 6]],\n  ...                  [[7, 8, 9],\n  ...                   [10, 11, 12]]], dtype=tf.uint8)\n  >>> tf.image.random_jpeg_quality(x, 75, 95)\n  <tf.Tensor: shape=(2, 2, 3), dtype=uint8, numpy=...>\n\n  For producing deterministic results given a `seed` value, use\n  `tf.image.stateless_random_jpeg_quality`. Unlike using the `seed` param\n  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the\n  same results given the same seed independent of how many times the function is\n  called, and independent of global seed settings (e.g. tf.random.set_seed).\n\n  Args:\n    image: 3D image. Size of the last dimension must be 1 or 3.\n    min_jpeg_quality: Minimum jpeg encoding quality to use.\n    max_jpeg_quality: Maximum jpeg encoding quality to use.\n    seed: An operation-specific seed. It will be used in conjunction with the\n      graph-level seed to determine the real seeds that will be used in this\n      operation. Please see the documentation of set_random_seed for its\n      interaction with the graph-level random seed.\n\n  Returns:\n    Adjusted image(s), same shape and DType as `image`.\n\n  Raises:\n    ValueError: if `min_jpeg_quality` or `max_jpeg_quality` is invalid.\n  '
    if min_jpeg_quality < 0 or max_jpeg_quality < 0 or min_jpeg_quality > 100 or (max_jpeg_quality > 100):
        raise ValueError('jpeg encoding range must be between 0 and 100.')
    if min_jpeg_quality >= max_jpeg_quality:
        raise ValueError('`min_jpeg_quality` must be less than `max_jpeg_quality`.')
    jpeg_quality = random_ops.random_uniform([], min_jpeg_quality, max_jpeg_quality, seed=seed, dtype=dtypes.int32)
    return adjust_jpeg_quality(image, jpeg_quality)

@tf_export('image.stateless_random_jpeg_quality', v1=[])
@dispatch.add_dispatch_support
def stateless_random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed):
    if False:
        return 10
    'Deterministically radomize jpeg encoding quality for inducing jpeg noise.\n\n  Guarantees the same results given the same `seed` independent of how many\n  times the function is called, and independent of global seed settings (e.g.\n  `tf.random.set_seed`).\n\n  `min_jpeg_quality` must be in the interval `[0, 100]` and less than\n  `max_jpeg_quality`.\n  `max_jpeg_quality` must be in the interval `[0, 100]`.\n\n  Usage Example:\n\n  >>> x = tf.constant([[[1, 2, 3],\n  ...                   [4, 5, 6]],\n  ...                  [[7, 8, 9],\n  ...                   [10, 11, 12]]], dtype=tf.uint8)\n  >>> seed = (1, 2)\n  >>> tf.image.stateless_random_jpeg_quality(x, 75, 95, seed)\n  <tf.Tensor: shape=(2, 2, 3), dtype=uint8, numpy=\n  array([[[ 0,  4,  5],\n          [ 1,  5,  6]],\n         [[ 5,  9, 10],\n          [ 5,  9, 10]]], dtype=uint8)>\n\n  Args:\n    image: 3D image. Size of the last dimension must be 1 or 3.\n    min_jpeg_quality: Minimum jpeg encoding quality to use.\n    max_jpeg_quality: Maximum jpeg encoding quality to use.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n\n  Returns:\n    Adjusted image(s), same shape and DType as `image`.\n\n  Raises:\n    ValueError: if `min_jpeg_quality` or `max_jpeg_quality` is invalid.\n  '
    if min_jpeg_quality < 0 or max_jpeg_quality < 0 or min_jpeg_quality > 100 or (max_jpeg_quality > 100):
        raise ValueError('jpeg encoding range must be between 0 and 100.')
    if min_jpeg_quality >= max_jpeg_quality:
        raise ValueError('`min_jpeg_quality` must be less than `max_jpeg_quality`.')
    jpeg_quality = stateless_random_ops.stateless_random_uniform(shape=[], minval=min_jpeg_quality, maxval=max_jpeg_quality, seed=seed, dtype=dtypes.int32)
    return adjust_jpeg_quality(image, jpeg_quality)

@tf_export('image.adjust_jpeg_quality')
@dispatch.add_dispatch_support
def adjust_jpeg_quality(image, jpeg_quality, dct_method='', name=None):
    if False:
        for i in range(10):
            print('nop')
    'Adjust jpeg encoding quality of an image.\n\n  This is a convenience method that converts an image to uint8 representation,\n  encodes it to jpeg with `jpeg_quality`, decodes it, and then converts back\n  to the original data type.\n\n  `jpeg_quality` must be in the interval `[0, 100]`.\n\n  Usage Examples:\n\n  >>> x = [[[0.01, 0.02, 0.03],\n  ...       [0.04, 0.05, 0.06]],\n  ...      [[0.07, 0.08, 0.09],\n  ...       [0.10, 0.11, 0.12]]]\n  >>> x_jpeg = tf.image.adjust_jpeg_quality(x, 75)\n  >>> x_jpeg.numpy()\n  array([[[0.00392157, 0.01960784, 0.03137255],\n          [0.02745098, 0.04313726, 0.05490196]],\n         [[0.05882353, 0.07450981, 0.08627451],\n          [0.08235294, 0.09803922, 0.10980393]]], dtype=float32)\n\n  Note that floating point values are expected to have values in the range\n  [0,1) and values outside this range are clipped.\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.adjust_jpeg_quality(x, 75)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[1., 1., 1.],\n          [1., 1., 1.]],\n         [[1., 1., 1.],\n          [1., 1., 1.]]], dtype=float32)>\n\n  Note that `jpeg_quality` 100 is still lossy compression.\n\n  >>> x = tf.constant([[[1, 2, 3],\n  ...                   [4, 5, 6]],\n  ...                  [[7, 8, 9],\n  ...                   [10, 11, 12]]], dtype=tf.uint8)\n  >>> tf.image.adjust_jpeg_quality(x, 100)\n  <tf.Tensor: shape(2, 2, 3), dtype=uint8, numpy=\n  array([[[ 0,  1,  3],\n          [ 3,  4,  6]],\n         [[ 6,  7,  9],\n          [ 9, 10, 12]]], dtype=uint8)>\n\n  Args:\n    image: 3D image. The size of the last dimension must be None, 1 or 3.\n    jpeg_quality: Python int or Tensor of type int32. jpeg encoding quality.\n    dct_method: An optional string. Specifies the DCT method to use for JPEG\n      decompression. Currently available options are ["INTEGER_FAST",\n      "INTEGER_ACCURATE"]. Defaults to "" which maps to "INTEGER_FAST",\n      sacrificing image quality for speed.\n    name: A name for this operation (optional).\n\n  Returns:\n    Adjusted image, same shape and DType as `image`.\n\n  Raises:\n    InvalidArgumentError: quality must be in [0,100]\n    InvalidArgumentError: image must have 1 or 3 channels\n  '
    with ops.name_scope(name, 'adjust_jpeg_quality', [image]):
        image = ops.convert_to_tensor(image, name='image')
        channels = image.shape.as_list()[-1]
        orig_dtype = image.dtype
        image = convert_image_dtype(image, dtypes.uint8, saturate=True)
        if not _is_tensor(jpeg_quality):
            jpeg_quality = ops.convert_to_tensor(jpeg_quality, dtype=dtypes.int32)
        image = gen_image_ops.encode_jpeg_variable_quality(image, jpeg_quality)
        image = gen_image_ops.decode_jpeg(image, channels=channels, dct_method=dct_method)
        return convert_image_dtype(image, orig_dtype, saturate=True)

@tf_export('image.random_saturation')
@dispatch.add_dispatch_support
def random_saturation(image, lower, upper, seed=None):
    if False:
        return 10
    'Adjust the saturation of RGB images by a random factor.\n\n  Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly\n  picked in the interval `[lower, upper)`.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.random_saturation(x, 5, 10)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 0. ,  1.5,  3. ],\n          [ 0. ,  3. ,  6. ]],\n         [[ 0. ,  4.5,  9. ],\n          [ 0. ,  6. , 12. ]]], dtype=float32)>\n\n  For producing deterministic results given a `seed` value, use\n  `tf.image.stateless_random_saturation`. Unlike using the `seed` param\n  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the\n  same results given the same seed independent of how many times the function is\n  called, and independent of global seed settings (e.g. tf.random.set_seed).\n\n  Args:\n    image: RGB image or images. The size of the last dimension must be 3.\n    lower: float.  Lower bound for the random saturation factor.\n    upper: float.  Upper bound for the random saturation factor.\n    seed: An operation-specific seed. It will be used in conjunction with the\n      graph-level seed to determine the real seeds that will be used in this\n      operation. Please see the documentation of set_random_seed for its\n      interaction with the graph-level random seed.\n\n  Returns:\n    Adjusted image(s), same shape and DType as `image`.\n\n  Raises:\n    ValueError: if `upper <= lower` or if `lower < 0`.\n  '
    if upper <= lower:
        raise ValueError('upper must be > lower.')
    if lower < 0:
        raise ValueError('lower must be non-negative.')
    saturation_factor = random_ops.random_uniform([], lower, upper, seed=seed)
    return adjust_saturation(image, saturation_factor)

@tf_export('image.stateless_random_saturation', v1=[])
@dispatch.add_dispatch_support
def stateless_random_saturation(image, lower, upper, seed=None):
    if False:
        i = 10
        return i + 15
    'Adjust the saturation of RGB images by a random factor deterministically.\n\n  Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly\n  picked in the interval `[lower, upper)`.\n\n  Guarantees the same results given the same `seed` independent of how many\n  times the function is called, and independent of global seed settings (e.g.\n  `tf.random.set_seed`).\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...      [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> seed = (1, 2)\n  >>> tf.image.stateless_random_saturation(x, 0.5, 1.0, seed)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 1.1559395,  2.0779698,  3.       ],\n          [ 4.1559396,  5.07797  ,  6.       ]],\n         [[ 7.1559396,  8.07797  ,  9.       ],\n          [10.155939 , 11.07797  , 12.       ]]], dtype=float32)>\n\n  Args:\n    image: RGB image or images. The size of the last dimension must be 3.\n    lower: float.  Lower bound for the random saturation factor.\n    upper: float.  Upper bound for the random saturation factor.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n\n  Returns:\n    Adjusted image(s), same shape and DType as `image`.\n\n  Raises:\n    ValueError: if `upper <= lower` or if `lower < 0`.\n  '
    if upper <= lower:
        raise ValueError('upper must be > lower.')
    if lower < 0:
        raise ValueError('lower must be non-negative.')
    saturation_factor = stateless_random_ops.stateless_random_uniform(shape=[], minval=lower, maxval=upper, seed=seed)
    return adjust_saturation(image, saturation_factor)

@tf_export('image.adjust_saturation')
@dispatch.add_dispatch_support
def adjust_saturation(image, saturation_factor, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Adjust saturation of RGB images.\n\n  This is a convenience method that converts RGB images to float\n  representation, converts them to HSV, adds an offset to the\n  saturation channel, converts back to RGB and then back to the original\n  data type. If several adjustments are chained it is advisable to minimize\n  the number of redundant conversions.\n\n  `image` is an RGB image or images.  The image saturation is adjusted by\n  converting the images to HSV and multiplying the saturation (S) channel by\n  `saturation_factor` and clipping. The images are then converted back to RGB.\n\n  `saturation_factor` must be in the interval `[0, inf)`.\n\n  Usage Example:\n\n  >>> x = [[[1.0, 2.0, 3.0],\n  ...       [4.0, 5.0, 6.0]],\n  ...     [[7.0, 8.0, 9.0],\n  ...       [10.0, 11.0, 12.0]]]\n  >>> tf.image.adjust_saturation(x, 0.5)\n  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=\n  array([[[ 2. ,  2.5,  3. ],\n          [ 5. ,  5.5,  6. ]],\n         [[ 8. ,  8.5,  9. ],\n          [11. , 11.5, 12. ]]], dtype=float32)>\n\n  Args:\n    image: RGB image or images. The size of the last dimension must be 3.\n    saturation_factor: float. Factor to multiply the saturation by.\n    name: A name for this operation (optional).\n\n  Returns:\n    Adjusted image(s), same shape and DType as `image`.\n\n  Raises:\n    InvalidArgumentError: input must have 3 channels\n  '
    with ops.name_scope(name, 'adjust_saturation', [image]) as name:
        image = ops.convert_to_tensor(image, name='image')
        orig_dtype = image.dtype
        if orig_dtype in (dtypes.float16, dtypes.float32):
            flt_image = image
        else:
            flt_image = convert_image_dtype(image, dtypes.float32)
        adjusted = gen_image_ops.adjust_saturation(flt_image, saturation_factor)
        return convert_image_dtype(adjusted, orig_dtype)

@tf_export('io.is_jpeg', 'image.is_jpeg', v1=['io.is_jpeg', 'image.is_jpeg'])
def is_jpeg(contents, name=None):
    if False:
        i = 10
        return i + 15
    "Convenience function to check if the 'contents' encodes a JPEG image.\n\n  Args:\n    contents: 0-D `string`. The encoded image bytes.\n    name: A name for the operation (optional)\n\n  Returns:\n     A scalar boolean tensor indicating if 'contents' may be a JPEG image.\n     is_jpeg is susceptible to false positives.\n  "
    with ops.name_scope(name, 'is_jpeg'):
        substr = string_ops.substr(contents, 0, 3)
        return math_ops.equal(substr, b'\xff\xd8\xff', name=name)

def _is_png(contents, name=None):
    if False:
        return 10
    "Convenience function to check if the 'contents' encodes a PNG image.\n\n  Args:\n    contents: 0-D `string`. The encoded image bytes.\n    name: A name for the operation (optional)\n\n  Returns:\n     A scalar boolean tensor indicating if 'contents' may be a PNG image.\n     is_png is susceptible to false positives.\n  "
    with ops.name_scope(name, 'is_png'):
        substr = string_ops.substr(contents, 0, 3)
        return math_ops.equal(substr, b'\x89PN', name=name)
decode_and_crop_jpeg = tf_export('io.decode_and_crop_jpeg', 'image.decode_and_crop_jpeg', v1=['io.decode_and_crop_jpeg', 'image.decode_and_crop_jpeg'])(dispatch.add_dispatch_support(gen_image_ops.decode_and_crop_jpeg))
decode_bmp = tf_export('io.decode_bmp', 'image.decode_bmp', v1=['io.decode_bmp', 'image.decode_bmp'])(dispatch.add_dispatch_support(gen_image_ops.decode_bmp))
decode_gif = tf_export('io.decode_gif', 'image.decode_gif', v1=['io.decode_gif', 'image.decode_gif'])(dispatch.add_dispatch_support(gen_image_ops.decode_gif))
decode_jpeg = tf_export('io.decode_jpeg', 'image.decode_jpeg', v1=['io.decode_jpeg', 'image.decode_jpeg'])(dispatch.add_dispatch_support(gen_image_ops.decode_jpeg))
decode_png = tf_export('io.decode_png', 'image.decode_png', v1=['io.decode_png', 'image.decode_png'])(dispatch.add_dispatch_support(gen_image_ops.decode_png))
encode_jpeg = tf_export('io.encode_jpeg', 'image.encode_jpeg', v1=['io.encode_jpeg', 'image.encode_jpeg'])(dispatch.add_dispatch_support(gen_image_ops.encode_jpeg))
extract_jpeg_shape = tf_export('io.extract_jpeg_shape', 'image.extract_jpeg_shape', v1=['io.extract_jpeg_shape', 'image.extract_jpeg_shape'])(dispatch.add_dispatch_support(gen_image_ops.extract_jpeg_shape))

@tf_export('io.encode_png', 'image.encode_png')
@dispatch.add_dispatch_support
def encode_png(image, compression=-1, name=None):
    if False:
        for i in range(10):
            print('nop')
    'PNG-encode an image.\n\n  `image` is a rank-N Tensor of type uint8 or uint16 with shape `batch_dims +\n  [height, width, channels]`, where `channels` is:\n\n  *   1: for grayscale.\n  *   2: for grayscale + alpha.\n  *   3: for RGB.\n  *   4: for RGBA.\n\n  The ZLIB compression level, `compression`, can be -1 for the PNG-encoder\n  default or a value from 0 to 9.  9 is the highest compression level,\n  generating the smallest output, but is slower.\n\n  Args:\n    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`.\n      Rank N >= 3 with shape `batch_dims + [height, width, channels]`.\n    compression: An optional `int`. Defaults to `-1`. Compression level.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` of type `string`.\n  '
    return gen_image_ops.encode_png(ops.convert_to_tensor(image), compression, name)

@tf_export('io.decode_image', 'image.decode_image', v1=['io.decode_image', 'image.decode_image'])
@dispatch.add_dispatch_support
def decode_image(contents, channels=None, dtype=dtypes.uint8, name=None, expand_animations=True):
    if False:
        print('Hello World!')
    "Function for `decode_bmp`, `decode_gif`, `decode_jpeg`, and `decode_png`.\n\n  Detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the\n  appropriate operation to convert the input bytes `string` into a `Tensor`\n  of type `dtype`.\n\n  Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as\n  opposed to `decode_bmp`, `decode_jpeg` and `decode_png`, which return 3-D\n  arrays `[height, width, num_channels]`. Make sure to take this into account\n  when constructing your graph if you are intermixing GIF files with BMP, JPEG,\n  and/or PNG files. Alternately, set the `expand_animations` argument of this\n  function to `False`, in which case the op will return 3-dimensional tensors\n  and will truncate animated GIF files to the first frame.\n\n  NOTE: If the first frame of an animated GIF does not occupy the entire\n  canvas (maximum frame width x maximum frame height), then it fills the\n  unoccupied areas (in the first frame) with zeros (black). For frames after the\n  first frame that does not occupy the entire canvas, it uses the previous\n  frame to fill the unoccupied areas.\n\n  Args:\n    contents: A `Tensor` of type `string`. 0-D. The encoded image bytes.\n    channels: An optional `int`. Defaults to `0`. Number of color channels for\n      the decoded image.\n    dtype: The desired DType of the returned `Tensor`.\n    name: A name for the operation (optional)\n    expand_animations: An optional `bool`. Defaults to `True`. Controls the\n      shape of the returned op's output. If `True`, the returned op will produce\n      a 3-D tensor for PNG, JPEG, and BMP files; and a 4-D tensor for all GIFs,\n      whether animated or not. If, `False`, the returned op will produce a 3-D\n      tensor for all file types and will truncate animated GIFs to the first\n      frame.\n\n  Returns:\n    `Tensor` with type `dtype` and a 3- or 4-dimensional shape, depending on\n    the file type and the value of the `expand_animations` parameter.\n\n  Raises:\n    ValueError: On incorrect number of channels.\n  "
    with ops.name_scope(name, 'decode_image'):
        channels = 0 if channels is None else channels
        if dtype not in [dtypes.float32, dtypes.uint8, dtypes.uint16]:
            dest_dtype = dtype
            dtype = dtypes.uint16
            return convert_image_dtype(gen_image_ops.decode_image(contents=contents, channels=channels, expand_animations=expand_animations, dtype=dtype), dest_dtype)
        else:
            return gen_image_ops.decode_image(contents=contents, channels=channels, expand_animations=expand_animations, dtype=dtype)

@tf_export('image.total_variation')
@dispatch.add_dispatch_support
def total_variation(images, name=None):
    if False:
        print('Hello World!')
    'Calculate and return the total variation for one or more images.\n\n  The total variation is the sum of the absolute differences for neighboring\n  pixel-values in the input images. This measures how much noise is in the\n  images.\n\n  This can be used as a loss-function during optimization so as to suppress\n  noise in images. If you have a batch of images, then you should calculate\n  the scalar loss-value as the sum:\n  `loss = tf.reduce_sum(tf.image.total_variation(images))`\n\n  This implements the anisotropic 2-D version of the formula described here:\n\n  https://en.wikipedia.org/wiki/Total_variation_denoising\n\n  Args:\n    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor\n      of shape `[height, width, channels]`.\n    name: A name for the operation (optional).\n\n  Raises:\n    ValueError: if images.shape is not a 3-D or 4-D vector.\n\n  Returns:\n    The total variation of `images`.\n\n    If `images` was 4-D, return a 1-D float Tensor of shape `[batch]` with the\n    total variation for each image in the batch.\n    If `images` was 3-D, return a scalar float with the total variation for\n    that image.\n  '
    with ops.name_scope(name, 'total_variation'):
        ndims = images.get_shape().ndims
        if ndims == 3:
            pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
            pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
            sum_axis = None
        elif ndims == 4:
            pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
            pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
            sum_axis = [1, 2, 3]
        else:
            raise ValueError("'images' must be either 3 or 4-dimensional.")
        tot_var = math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) + math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis)
    return tot_var

@tf_export('image.sample_distorted_bounding_box', v1=[])
@dispatch.add_dispatch_support
def sample_distorted_bounding_box_v2(image_size, bounding_boxes, seed=0, min_object_covered=0.1, aspect_ratio_range=None, area_range=None, max_attempts=None, use_image_if_no_bounding_boxes=None, name=None):
    if False:
        return 10
    "Generate a single randomly distorted bounding box for an image.\n\n  Bounding box annotations are often supplied in addition to ground-truth labels\n  in image recognition or object localization tasks. A common technique for\n  training such a system is to randomly distort an image while preserving\n  its content, i.e. *data augmentation*. This Op outputs a randomly distorted\n  localization of an object, i.e. bounding box, given an `image_size`,\n  `bounding_boxes` and a series of constraints.\n\n  The output of this Op is a single bounding box that may be used to crop the\n  original image. The output is returned as 3 tensors: `begin`, `size` and\n  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the\n  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to\n  visualize what the bounding box looks like.\n\n  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.\n  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width\n  and the height of the underlying image.\n\n  For example,\n\n  ```python\n      # Generate a single distorted bounding box.\n      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(\n          tf.shape(image),\n          bounding_boxes=bounding_boxes,\n          min_object_covered=0.1)\n\n      # Draw the bounding box in an image summary.\n      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),\n                                                    bbox_for_draw)\n      tf.compat.v1.summary.image('images_with_box', image_with_box)\n\n      # Employ the bounding box to distort the image.\n      distorted_image = tf.slice(image, begin, size)\n  ```\n\n  Note that if no bounding box information is available, setting\n  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit\n  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is\n  false and no bounding boxes are supplied, an error is raised.\n\n  For producing deterministic results given a `seed` value, use\n  `tf.image.stateless_sample_distorted_bounding_box`. Unlike using the `seed`\n  param with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops\n  guarantee the same results given the same seed independent of how many times\n  the function is called, and independent of global seed settings\n  (e.g. tf.random.set_seed).\n\n  Args:\n    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`,\n      `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.\n    bounding_boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`\n      describing the N bounding boxes associated with the image.\n    seed: An optional `int`. Defaults to `0`. If `seed` is set to non-zero, the\n      random number generator is seeded by the given `seed`.  Otherwise, it is\n      seeded by a random seed.\n    min_object_covered: A Tensor of type `float32`. Defaults to `0.1`. The\n      cropped area of the image must contain at least this fraction of any\n      bounding box supplied. The value of this parameter should be non-negative.\n      In the case of 0, the cropped area does not need to overlap any of the\n      bounding boxes supplied.\n    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75,\n      1.33]`. The cropped area of the image must have an aspect `ratio = width /\n      height` within this range.\n    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The\n      cropped area of the image must contain a fraction of the supplied image\n      within this range.\n    max_attempts: An optional `int`. Defaults to `100`. Number of attempts at\n      generating a cropped region of the image of the specified constraints.\n      After `max_attempts` failures, return the entire image.\n    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.\n      Controls behavior if no bounding boxes supplied. If true, assume an\n      implicit bounding box covering the whole input. If false, raise an error.\n    name: A name for the operation (optional).\n\n  Returns:\n    A tuple of `Tensor` objects (begin, size, bboxes).\n\n    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing\n    `[offset_height, offset_width, 0]`. Provide as input to\n      `tf.slice`.\n    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing\n    `[target_height, target_width, -1]`. Provide as input to\n      `tf.slice`.\n    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing\n    the distorted bounding box.\n    Provide as input to `tf.image.draw_bounding_boxes`.\n\n  Raises:\n    ValueError: If no seed is specified and op determinism is enabled.\n  "
    if seed:
        (seed1, seed2) = random_seed.get_seed(seed)
    else:
        if config.is_op_determinism_enabled():
            raise ValueError(f'tf.image.sample_distorted_bounding_box requires a non-zero seed to be passed in when determinism is enabled, but got seed={seed}. Please pass in a non-zero seed, e.g. by passing "seed=1".')
        (seed1, seed2) = (0, 0)
    with ops.name_scope(name, 'sample_distorted_bounding_box'):
        return gen_image_ops.sample_distorted_bounding_box_v2(image_size, bounding_boxes, seed=seed1, seed2=seed2, min_object_covered=min_object_covered, aspect_ratio_range=aspect_ratio_range, area_range=area_range, max_attempts=max_attempts, use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes, name=name)

@tf_export('image.stateless_sample_distorted_bounding_box', v1=[])
@dispatch.add_dispatch_support
def stateless_sample_distorted_bounding_box(image_size, bounding_boxes, seed, min_object_covered=0.1, aspect_ratio_range=None, area_range=None, max_attempts=None, use_image_if_no_bounding_boxes=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Generate a randomly distorted bounding box for an image deterministically.\n\n  Bounding box annotations are often supplied in addition to ground-truth labels\n  in image recognition or object localization tasks. A common technique for\n  training such a system is to randomly distort an image while preserving\n  its content, i.e. *data augmentation*. This Op, given the same `seed`,\n  deterministically outputs a randomly distorted localization of an object, i.e.\n  bounding box, given an `image_size`, `bounding_boxes` and a series of\n  constraints.\n\n  The output of this Op is a single bounding box that may be used to crop the\n  original image. The output is returned as 3 tensors: `begin`, `size` and\n  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the\n  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to\n  visualize what the bounding box looks like.\n\n  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.\n  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width\n  and the height of the underlying image.\n\n  The output of this Op is guaranteed to be the same given the same `seed` and\n  is independent of how many times the function is called, and independent of\n  global seed settings (e.g. `tf.random.set_seed`).\n\n  Example usage:\n\n  >>> image = np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])\n  >>> bbox = tf.constant(\n  ...   [0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])\n  >>> seed = (1, 2)\n  >>> # Generate a single distorted bounding box.\n  >>> bbox_begin, bbox_size, bbox_draw = (\n  ...   tf.image.stateless_sample_distorted_bounding_box(\n  ...     tf.shape(image), bounding_boxes=bbox, seed=seed))\n  >>> # Employ the bounding box to distort the image.\n  >>> tf.slice(image, bbox_begin, bbox_size)\n  <tf.Tensor: shape=(2, 2, 1), dtype=int64, numpy=\n  array([[[1],\n          [2]],\n         [[4],\n          [5]]])>\n  >>> # Draw the bounding box in an image summary.\n  >>> colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])\n  >>> tf.image.draw_bounding_boxes(\n  ...   tf.expand_dims(tf.cast(image, tf.float32),0), bbox_draw, colors)\n  <tf.Tensor: shape=(1, 3, 3, 1), dtype=float32, numpy=\n  array([[[[1.],\n           [1.],\n           [3.]],\n          [[1.],\n           [1.],\n           [6.]],\n          [[7.],\n           [8.],\n           [9.]]]], dtype=float32)>\n\n  Note that if no bounding box information is available, setting\n  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit\n  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is\n  false and no bounding boxes are supplied, an error is raised.\n\n  Args:\n    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`,\n      `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.\n    bounding_boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`\n      describing the N bounding boxes associated with the image.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    min_object_covered: A Tensor of type `float32`. Defaults to `0.1`. The\n      cropped area of the image must contain at least this fraction of any\n      bounding box supplied. The value of this parameter should be non-negative.\n      In the case of 0, the cropped area does not need to overlap any of the\n      bounding boxes supplied.\n    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75,\n      1.33]`. The cropped area of the image must have an aspect `ratio = width /\n      height` within this range.\n    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The\n      cropped area of the image must contain a fraction of the supplied image\n      within this range.\n    max_attempts: An optional `int`. Defaults to `100`. Number of attempts at\n      generating a cropped region of the image of the specified constraints.\n      After `max_attempts` failures, return the entire image.\n    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.\n      Controls behavior if no bounding boxes supplied. If true, assume an\n      implicit bounding box covering the whole input. If false, raise an error.\n    name: A name for the operation (optional).\n\n  Returns:\n    A tuple of `Tensor` objects (begin, size, bboxes).\n\n    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing\n    `[offset_height, offset_width, 0]`. Provide as input to\n      `tf.slice`.\n    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing\n    `[target_height, target_width, -1]`. Provide as input to\n      `tf.slice`.\n    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing\n    the distorted bounding box.\n    Provide as input to `tf.image.draw_bounding_boxes`.\n  '
    with ops.name_scope(name, 'stateless_sample_distorted_bounding_box'):
        return gen_image_ops.stateless_sample_distorted_bounding_box(image_size=image_size, bounding_boxes=bounding_boxes, seed=seed, min_object_covered=min_object_covered, aspect_ratio_range=aspect_ratio_range, area_range=area_range, max_attempts=max_attempts, use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes, name=name)

@tf_export(v1=['image.sample_distorted_bounding_box'])
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions='`seed2` arg is deprecated.Use sample_distorted_bounding_box_v2 instead.')
def sample_distorted_bounding_box(image_size, bounding_boxes, seed=None, seed2=None, min_object_covered=0.1, aspect_ratio_range=None, area_range=None, max_attempts=None, use_image_if_no_bounding_boxes=None, name=None):
    if False:
        i = 10
        return i + 15
    "Generate a single randomly distorted bounding box for an image.\n\n  Bounding box annotations are often supplied in addition to ground-truth labels\n  in image recognition or object localization tasks. A common technique for\n  training such a system is to randomly distort an image while preserving\n  its content, i.e. *data augmentation*. This Op outputs a randomly distorted\n  localization of an object, i.e. bounding box, given an `image_size`,\n  `bounding_boxes` and a series of constraints.\n\n  The output of this Op is a single bounding box that may be used to crop the\n  original image. The output is returned as 3 tensors: `begin`, `size` and\n  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the\n  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to\n  visualize what the bounding box looks like.\n\n  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.\n  The\n  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and\n  height of the underlying image.\n\n  For example,\n\n  ```python\n      # Generate a single distorted bounding box.\n      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(\n          tf.shape(image),\n          bounding_boxes=bounding_boxes,\n          min_object_covered=0.1)\n\n      # Draw the bounding box in an image summary.\n      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),\n                                                    bbox_for_draw)\n      tf.compat.v1.summary.image('images_with_box', image_with_box)\n\n      # Employ the bounding box to distort the image.\n      distorted_image = tf.slice(image, begin, size)\n  ```\n\n  Note that if no bounding box information is available, setting\n  `use_image_if_no_bounding_boxes = True` will assume there is a single implicit\n  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is\n  false and no bounding boxes are supplied, an error is raised.\n\n  Args:\n    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`,\n      `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.\n    bounding_boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`\n      describing the N bounding boxes associated with the image.\n    seed: An optional `int`. Defaults to `0`. If either `seed` or `seed2` are\n      set to non-zero, the random number generator is seeded by the given\n      `seed`.  Otherwise, it is seeded by a random seed.\n    seed2: An optional `int`. Defaults to `0`. A second seed to avoid seed\n      collision.\n    min_object_covered: A Tensor of type `float32`. Defaults to `0.1`. The\n      cropped area of the image must contain at least this fraction of any\n      bounding box supplied. The value of this parameter should be non-negative.\n      In the case of 0, the cropped area does not need to overlap any of the\n      bounding boxes supplied.\n    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75,\n      1.33]`. The cropped area of the image must have an aspect ratio = width /\n      height within this range.\n    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The\n      cropped area of the image must contain a fraction of the supplied image\n      within this range.\n    max_attempts: An optional `int`. Defaults to `100`. Number of attempts at\n      generating a cropped region of the image of the specified constraints.\n      After `max_attempts` failures, return the entire image.\n    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.\n      Controls behavior if no bounding boxes supplied. If true, assume an\n      implicit bounding box covering the whole input. If false, raise an error.\n    name: A name for the operation (optional).\n\n  Returns:\n    A tuple of `Tensor` objects (begin, size, bboxes).\n\n    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing\n    `[offset_height, offset_width, 0]`. Provide as input to\n      `tf.slice`.\n    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing\n    `[target_height, target_width, -1]`. Provide as input to\n      `tf.slice`.\n    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing\n    the distorted bounding box.\n      Provide as input to `tf.image.draw_bounding_boxes`.\n\n  Raises:\n    ValueError: If no seed is specified and op determinism is enabled.\n  "
    if not seed and (not seed2) and config.is_op_determinism_enabled():
        raise ValueError(f'tf.compat.v1.image.sample_distorted_bounding_box requires "seed" or "seed2" to be non-zero when determinism is enabled. Please pass in a non-zero seed, e.g. by passing "seed=1". Got seed={seed} and seed2={seed2}')
    with ops.name_scope(name, 'sample_distorted_bounding_box'):
        return gen_image_ops.sample_distorted_bounding_box_v2(image_size, bounding_boxes, seed=seed, seed2=seed2, min_object_covered=min_object_covered, aspect_ratio_range=aspect_ratio_range, area_range=area_range, max_attempts=max_attempts, use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes, name=name)

@tf_export('image.non_max_suppression')
@dispatch.add_dispatch_support
def non_max_suppression(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), name=None):
    if False:
        return 10
    'Greedily selects a subset of bounding boxes in descending order of score.\n\n  Prunes away boxes that have high intersection-over-union (IOU) overlap\n  with previously selected boxes.  Bounding boxes are supplied as\n  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any\n  diagonal pair of box corners and the coordinates can be provided as normalized\n  (i.e., lying in the interval `[0, 1]`) or absolute.  Note that this algorithm\n  is agnostic to where the origin is in the coordinate system.  Note that this\n  algorithm is invariant to orthogonal transformations and translations\n  of the coordinate system; thus translating or reflections of the coordinate\n  system result in the same boxes being selected by the algorithm.\n  The output of this operation is a set of integers indexing into the input\n  collection of bounding boxes representing the selected boxes.  The bounding\n  box coordinates corresponding to the selected indices can then be obtained\n  using the `tf.gather` operation.  For example:\n    ```python\n    selected_indices = tf.image.non_max_suppression(\n        boxes, scores, max_output_size, iou_threshold)\n    selected_boxes = tf.gather(boxes, selected_indices)\n    ```\n\n  Args:\n    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.\n    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single\n      score corresponding to each box (each row of boxes).\n    max_output_size: A scalar integer `Tensor` representing the maximum number\n      of boxes to be selected by non-max suppression.\n    iou_threshold: A 0-D float tensor representing the threshold for deciding\n      whether boxes overlap too much with respect to IOU.\n    score_threshold: A 0-D float tensor representing the threshold for deciding\n      when to remove boxes based on score.\n    name: A name for the operation (optional).\n\n  Returns:\n    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the\n      selected indices from the boxes tensor, where `M <= max_output_size`.\n  '
    with ops.name_scope(name, 'non_max_suppression'):
        iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
        score_threshold = ops.convert_to_tensor(score_threshold, name='score_threshold')
        return gen_image_ops.non_max_suppression_v3(boxes, scores, max_output_size, iou_threshold, score_threshold)

@tf_export('image.non_max_suppression_with_scores')
@dispatch.add_dispatch_support
def non_max_suppression_with_scores(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), soft_nms_sigma=0.0, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Greedily selects a subset of bounding boxes in descending order of score.\n\n  Prunes away boxes that have high intersection-over-union (IOU) overlap\n  with previously selected boxes.  Bounding boxes are supplied as\n  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any\n  diagonal pair of box corners and the coordinates can be provided as normalized\n  (i.e., lying in the interval `[0, 1]`) or absolute.  Note that this algorithm\n  is agnostic to where the origin is in the coordinate system.  Note that this\n  algorithm is invariant to orthogonal transformations and translations\n  of the coordinate system; thus translating or reflections of the coordinate\n  system result in the same boxes being selected by the algorithm.\n  The output of this operation is a set of integers indexing into the input\n  collection of bounding boxes representing the selected boxes.  The bounding\n  box coordinates corresponding to the selected indices can then be obtained\n  using the `tf.gather` operation.  For example:\n    ```python\n    selected_indices, selected_scores = tf.image.non_max_suppression_padded(\n        boxes, scores, max_output_size, iou_threshold=1.0, score_threshold=0.1,\n        soft_nms_sigma=0.5)\n    selected_boxes = tf.gather(boxes, selected_indices)\n    ```\n\n  This function generalizes the `tf.image.non_max_suppression` op by also\n  supporting a Soft-NMS (with Gaussian weighting) mode (c.f.\n  Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score\n  of other overlapping boxes instead of directly causing them to be pruned.\n  Consequently, in contrast to `tf.image.non_max_suppression`,\n  `tf.image.non_max_suppression_with_scores` returns the new scores of each\n  input box in the second output, `selected_scores`.\n\n  To enable this Soft-NMS mode, set the `soft_nms_sigma` parameter to be\n  larger than 0.  When `soft_nms_sigma` equals 0, the behavior of\n  `tf.image.non_max_suppression_with_scores` is identical to that of\n  `tf.image.non_max_suppression` (except for the extra output) both in function\n  and in running time.\n\n  Note that when `soft_nms_sigma` > 0, Soft-NMS is performed and `iou_threshold`\n  is ignored. `iou_threshold` is only used for standard NMS.\n\n  Args:\n    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.\n    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single\n      score corresponding to each box (each row of boxes).\n    max_output_size: A scalar integer `Tensor` representing the maximum number\n      of boxes to be selected by non-max suppression.\n    iou_threshold: A 0-D float tensor representing the threshold for deciding\n      whether boxes overlap too much with respect to IOU.\n    score_threshold: A 0-D float tensor representing the threshold for deciding\n      when to remove boxes based on score.\n    soft_nms_sigma: A 0-D float tensor representing the sigma parameter for Soft\n      NMS; see Bodla et al (c.f. https://arxiv.org/abs/1704.04503).  When\n      `soft_nms_sigma=0.0` (which is default), we fall back to standard (hard)\n      NMS.\n    name: A name for the operation (optional).\n\n  Returns:\n    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the\n      selected indices from the boxes tensor, where `M <= max_output_size`.\n    selected_scores: A 1-D float tensor of shape `[M]` representing the\n      corresponding scores for each selected box, where `M <= max_output_size`.\n      Scores only differ from corresponding input scores when using Soft NMS\n      (i.e. when `soft_nms_sigma>0`)\n  '
    with ops.name_scope(name, 'non_max_suppression_with_scores'):
        iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
        score_threshold = ops.convert_to_tensor(score_threshold, name='score_threshold')
        soft_nms_sigma = ops.convert_to_tensor(soft_nms_sigma, name='soft_nms_sigma')
        (selected_indices, selected_scores, _) = gen_image_ops.non_max_suppression_v5(boxes, scores, max_output_size, iou_threshold, score_threshold, soft_nms_sigma, pad_to_max_output_size=False)
        return (selected_indices, selected_scores)

@tf_export('image.non_max_suppression_overlaps')
@dispatch.add_dispatch_support
def non_max_suppression_with_overlaps(overlaps, scores, max_output_size, overlap_threshold=0.5, score_threshold=float('-inf'), name=None):
    if False:
        return 10
    'Greedily selects a subset of bounding boxes in descending order of score.\n\n  Prunes away boxes that have high overlap with previously selected boxes.\n  N-by-n overlap values are supplied as square matrix.\n  The output of this operation is a set of integers indexing into the input\n  collection of bounding boxes representing the selected boxes.  The bounding\n  box coordinates corresponding to the selected indices can then be obtained\n  using the `tf.gather` operation.  For example:\n    ```python\n    selected_indices = tf.image.non_max_suppression_overlaps(\n        overlaps, scores, max_output_size, iou_threshold)\n    selected_boxes = tf.gather(boxes, selected_indices)\n    ```\n\n  Args:\n    overlaps: A 2-D float `Tensor` of shape `[num_boxes, num_boxes]`\n      representing the n-by-n box overlap values.\n    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single\n      score corresponding to each box (each row of boxes).\n    max_output_size: A scalar integer `Tensor` representing the maximum number\n      of boxes to be selected by non-max suppression.\n    overlap_threshold: A 0-D float tensor representing the threshold for\n      deciding whether boxes overlap too much with respect to the provided\n      overlap values.\n    score_threshold: A 0-D float tensor representing the threshold for deciding\n      when to remove boxes based on score.\n    name: A name for the operation (optional).\n\n  Returns:\n    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the\n      selected indices from the overlaps tensor, where `M <= max_output_size`.\n  '
    with ops.name_scope(name, 'non_max_suppression_overlaps'):
        overlap_threshold = ops.convert_to_tensor(overlap_threshold, name='overlap_threshold')
        return gen_image_ops.non_max_suppression_with_overlaps(overlaps, scores, max_output_size, overlap_threshold, score_threshold)
_rgb_to_yiq_kernel = [[0.299, 0.59590059, 0.2115], [0.587, -0.27455667, -0.52273617], [0.114, -0.32134392, 0.31119955]]

@tf_export('image.rgb_to_yiq')
@dispatch.add_dispatch_support
def rgb_to_yiq(images):
    if False:
        while True:
            i = 10
    'Converts one or more images from RGB to YIQ.\n\n  Outputs a tensor of the same shape as the `images` tensor, containing the YIQ\n  value of the pixels.\n  The output is only well defined if the value in images are in [0,1].\n\n  Usage Example:\n\n  >>> x = tf.constant([[[1.0, 2.0, 3.0]]])\n  >>> tf.image.rgb_to_yiq(x)\n  <tf.Tensor: shape=(1, 1, 3), dtype=float32,\n  numpy=array([[[ 1.815     , -0.91724455,  0.09962624]]], dtype=float32)>\n\n  Args:\n    images: 2-D or higher rank. Image data to convert. Last dimension must be\n      size 3.\n\n  Returns:\n    images: tensor with the same shape as `images`.\n  '
    images = ops.convert_to_tensor(images, name='images')
    kernel = ops.convert_to_tensor(_rgb_to_yiq_kernel, dtype=images.dtype, name='kernel')
    ndims = images.get_shape().ndims
    return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])
_yiq_to_rgb_kernel = [[1, 1, 1], [0.95598634, -0.27201283, -1.10674021], [0.6208248, -0.64720424, 1.70423049]]

@tf_export('image.yiq_to_rgb')
@dispatch.add_dispatch_support
def yiq_to_rgb(images):
    if False:
        i = 10
        return i + 15
    'Converts one or more images from YIQ to RGB.\n\n  Outputs a tensor of the same shape as the `images` tensor, containing the RGB\n  value of the pixels.\n  The output is only well defined if the Y value in images are in [0,1],\n  I value are in [-0.5957,0.5957] and Q value are in [-0.5226,0.5226].\n\n  Args:\n    images: 2-D or higher rank. Image data to convert. Last dimension must be\n      size 3.\n\n  Returns:\n    images: tensor with the same shape as `images`.\n  '
    images = ops.convert_to_tensor(images, name='images')
    kernel = ops.convert_to_tensor(_yiq_to_rgb_kernel, dtype=images.dtype, name='kernel')
    ndims = images.get_shape().ndims
    return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])
_rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538], [0.587, -0.28886916, -0.51496512], [0.114, 0.43601035, -0.10001026]]

@tf_export('image.rgb_to_yuv')
@dispatch.add_dispatch_support
def rgb_to_yuv(images):
    if False:
        i = 10
        return i + 15
    'Converts one or more images from RGB to YUV.\n\n  Outputs a tensor of the same shape as the `images` tensor, containing the YUV\n  value of the pixels.\n  The output is only well defined if the value in images are in [0, 1].\n  There are two ways of representing an image: [0, 255] pixel values range or\n  [0, 1] (as float) pixel values range. Users need to convert the input image\n  into a float [0, 1] range.\n\n  Args:\n    images: 2-D or higher rank. Image data to convert. Last dimension must be\n      size 3.\n\n  Returns:\n    images: tensor with the same shape as `images`.\n  '
    images = ops.convert_to_tensor(images, name='images')
    kernel = ops.convert_to_tensor(_rgb_to_yuv_kernel, dtype=images.dtype, name='kernel')
    ndims = images.get_shape().ndims
    return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])
_yuv_to_rgb_kernel = [[1, 1, 1], [0, -0.394642334, 2.03206185], [1.13988303, -0.58062185, 0]]

@tf_export('image.yuv_to_rgb')
@dispatch.add_dispatch_support
def yuv_to_rgb(images):
    if False:
        for i in range(10):
            print('nop')
    'Converts one or more images from YUV to RGB.\n\n  Outputs a tensor of the same shape as the `images` tensor, containing the RGB\n  value of the pixels.\n  The output is only well defined if the Y value in images are in [0,1],\n  U and V value are in [-0.5,0.5].\n\n  As per the above description, you need to scale your YUV images if their\n  pixel values are not in the required range. Below given example illustrates\n  preprocessing of each channel of images before feeding them to `yuv_to_rgb`.\n\n  ```python\n  yuv_images = tf.random.uniform(shape=[100, 64, 64, 3], maxval=255)\n  last_dimension_axis = len(yuv_images.shape) - 1\n  yuv_tensor_images = tf.truediv(\n      tf.subtract(\n          yuv_images,\n          tf.reduce_min(yuv_images)\n      ),\n      tf.subtract(\n          tf.reduce_max(yuv_images),\n          tf.reduce_min(yuv_images)\n       )\n  )\n  y, u, v = tf.split(yuv_tensor_images, 3, axis=last_dimension_axis)\n  target_uv_min, target_uv_max = -0.5, 0.5\n  u = u * (target_uv_max - target_uv_min) + target_uv_min\n  v = v * (target_uv_max - target_uv_min) + target_uv_min\n  preprocessed_yuv_images = tf.concat([y, u, v], axis=last_dimension_axis)\n  rgb_tensor_images = tf.image.yuv_to_rgb(preprocessed_yuv_images)\n  ```\n\n  Args:\n    images: 2-D or higher rank. Image data to convert. Last dimension must be\n      size 3.\n\n  Returns:\n    images: tensor with the same shape as `images`.\n  '
    images = ops.convert_to_tensor(images, name='images')
    kernel = ops.convert_to_tensor(_yuv_to_rgb_kernel, dtype=images.dtype, name='kernel')
    ndims = images.get_shape().ndims
    return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])

def _verify_compatible_image_shapes(img1, img2):
    if False:
        while True:
            i = 10
    'Checks if two image tensors are compatible for applying SSIM or PSNR.\n\n  This function checks if two sets of images have ranks at least 3, and if the\n  last three dimensions match.\n\n  Args:\n    img1: Tensor containing the first image batch.\n    img2: Tensor containing the second image batch.\n\n  Returns:\n    A tuple containing: the first tensor shape, the second tensor shape, and a\n    list of control_flow_ops.Assert() ops implementing the checks.\n\n  Raises:\n    ValueError: When static shape check fails.\n  '
    shape1 = img1.get_shape().with_rank_at_least(3)
    shape2 = img2.get_shape().with_rank_at_least(3)
    shape1[-3:].assert_is_compatible_with(shape2[-3:])
    if shape1.ndims is not None and shape2.ndims is not None:
        for (dim1, dim2) in zip(reversed(shape1.dims[:-3]), reversed(shape2.dims[:-3])):
            if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
                raise ValueError('Two images are not compatible: %s and %s' % (shape1, shape2))
    (shape1, shape2) = array_ops.shape_n([img1, img2])
    checks = []
    checks.append(control_flow_assert.Assert(math_ops.greater_equal(array_ops.size(shape1), 3), [shape1, shape2], summarize=10))
    checks.append(control_flow_assert.Assert(math_ops.reduce_all(math_ops.equal(shape1[-3:], shape2[-3:])), [shape1, shape2], summarize=10))
    return (shape1, shape2, checks)

@tf_export('image.psnr')
@dispatch.add_dispatch_support
def psnr(a, b, max_val, name=None):
    if False:
        while True:
            i = 10
    "Returns the Peak Signal-to-Noise Ratio between a and b.\n\n  This is intended to be used on signals (or images). Produces a PSNR value for\n  each image in batch.\n\n  The last three dimensions of input are expected to be [height, width, depth].\n\n  Example:\n\n  ```python\n      # Read images from file.\n      im1 = tf.decode_png('path/to/im1.png')\n      im2 = tf.decode_png('path/to/im2.png')\n      # Compute PSNR over tf.uint8 Tensors.\n      psnr1 = tf.image.psnr(im1, im2, max_val=255)\n\n      # Compute PSNR over tf.float32 Tensors.\n      im1 = tf.image.convert_image_dtype(im1, tf.float32)\n      im2 = tf.image.convert_image_dtype(im2, tf.float32)\n      psnr2 = tf.image.psnr(im1, im2, max_val=1.0)\n      # psnr1 and psnr2 both have type tf.float32 and are almost equal.\n  ```\n\n  Args:\n    a: First set of images.\n    b: Second set of images.\n    max_val: The dynamic range of the images (i.e., the difference between the\n      maximum the and minimum allowed values).\n    name: Namespace to embed the computation in.\n\n  Returns:\n    The scalar PSNR between a and b. The returned tensor has type `tf.float32`\n    and shape [batch_size, 1].\n  "
    with ops.name_scope(name, 'PSNR', [a, b]):
        max_val = math_ops.cast(max_val, a.dtype)
        max_val = convert_image_dtype(max_val, dtypes.float32)
        a = convert_image_dtype(a, dtypes.float32)
        b = convert_image_dtype(b, dtypes.float32)
        mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-3, -2, -1])
        psnr_val = math_ops.subtract(20 * math_ops.log(max_val) / math_ops.log(10.0), np.float32(10 / np.log(10)) * math_ops.log(mse), name='psnr')
        (_, _, checks) = _verify_compatible_image_shapes(a, b)
        with ops.control_dependencies(checks):
            return array_ops.identity(psnr_val)

def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
    if False:
        i = 10
        return i + 15
    "Helper function for computing SSIM.\n\n  SSIM estimates covariances with weighted sums.  The default parameters\n  use a biased estimate of the covariance:\n  Suppose `reducer` is a weighted sum, then the mean estimators are\n    \\mu_x = \\sum_i w_i x_i,\n    \\mu_y = \\sum_i w_i y_i,\n  where w_i's are the weighted-sum weights, and covariance estimator is\n    cov_{xy} = \\sum_i w_i (x_i - \\mu_x) (y_i - \\mu_y)\n  with assumption \\sum_i w_i = 1. This covariance estimator is biased, since\n    E[cov_{xy}] = (1 - \\sum_i w_i ^ 2) Cov(X, Y).\n  For SSIM measure with unbiased covariance estimators, pass as `compensation`\n  argument (1 - \\sum_i w_i ^ 2).\n\n  Args:\n    x: First set of images.\n    y: Second set of images.\n    reducer: Function that computes 'local' averages from the set of images. For\n      non-convolutional version, this is usually tf.reduce_mean(x, [1, 2]), and\n      for convolutional version, this is usually tf.nn.avg_pool2d or\n      tf.nn.conv2d with weighted-sum kernel.\n    max_val: The dynamic range (i.e., the difference between the maximum\n      possible allowed value and the minimum allowed value).\n    compensation: Compensation factor. See above.\n    k1: Default value 0.01\n    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so\n      it would be better if we took the values in the range of 0 < K2 < 0.4).\n\n  Returns:\n    A pair containing the luminance measure, and the contrast-structure measure.\n  "
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = math_ops.square(mean0) + math_ops.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)
    num1 = reducer(x * y) * 2.0
    den1 = reducer(math_ops.square(x) + math_ops.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)
    return (luminance, cs)

def _fspecial_gauss(size, sigma):
    if False:
        print('Hello World!')
    "Function to mimic the 'fspecial' gaussian MATLAB function."
    size = ops.convert_to_tensor(size, dtypes.int32)
    sigma = ops.convert_to_tensor(sigma)
    coords = math_ops.cast(math_ops.range(size), sigma.dtype)
    coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0
    g = math_ops.square(coords)
    g *= -0.5 / math_ops.square(sigma)
    g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
    g = array_ops.reshape(g, shape=[1, -1])
    g = nn_ops.softmax(g)
    return array_ops.reshape(g, shape=[size, size, 1, 1])

def _ssim_per_channel(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_index_map=False):
    if False:
        print('Hello World!')
    'Computes SSIM index between img1 and img2 per color channel.\n\n  This function matches the standard SSIM implementation from:\n  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image\n  quality assessment: from error visibility to structural similarity. IEEE\n  transactions on image processing.\n\n  Details:\n    - 11x11 Gaussian filter of width 1.5 is used.\n    - k1 = 0.01, k2 = 0.03 as in the original paper.\n\n  Args:\n    img1: First image batch.\n    img2: Second image batch.\n    max_val: The dynamic range of the images (i.e., the difference between the\n      maximum the and minimum allowed values).\n    filter_size: Default value 11 (size of gaussian filter).\n    filter_sigma: Default value 1.5 (width of gaussian filter).\n    k1: Default value 0.01\n    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so\n      it would be better if we took the values in the range of 0 < K2 < 0.4).\n    return_index_map: If True returns local SSIM map instead of the global mean.\n\n  Returns:\n    A pair of tensors containing and channel-wise SSIM and contrast-structure\n    values. The shape is [..., channels].\n  '
    filter_size = constant_op.constant(filter_size, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)
    (shape1, shape2) = array_ops.shape_n([img1, img2])
    checks = [control_flow_assert.Assert(math_ops.reduce_all(math_ops.greater_equal(shape1[-3:-1], filter_size)), [shape1, filter_size], summarize=8), control_flow_assert.Assert(math_ops.reduce_all(math_ops.greater_equal(shape2[-3:-1], filter_size)), [shape2, filter_size], summarize=8)]
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])
    compensation = 1.0

    def reducer(x):
        if False:
            for i in range(10):
                print('nop')
        shape = array_ops.shape(x)
        x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
        y = nn_impl.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return array_ops.reshape(y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0))
    (luminance, cs) = _ssim_helper(img1, img2, reducer, max_val, compensation, k1, k2)
    if return_index_map:
        ssim_val = luminance * cs
    else:
        axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
        ssim_val = math_ops.reduce_mean(luminance * cs, axes)
        cs = math_ops.reduce_mean(cs, axes)
    return (ssim_val, cs)

@tf_export('image.ssim')
@dispatch.add_dispatch_support
def ssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_index_map=False):
    if False:
        while True:
            i = 10
    "Computes SSIM index between img1 and img2.\n\n  This function is based on the standard SSIM implementation from:\n  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image\n  quality assessment: from error visibility to structural similarity. IEEE\n  transactions on image processing.\n\n  Note: The true SSIM is only defined on grayscale.  This function does not\n  perform any colorspace transform.  (If the input is already YUV, then it will\n  compute YUV SSIM average.)\n\n  Details:\n    - 11x11 Gaussian filter of width 1.5 is used.\n    - k1 = 0.01, k2 = 0.03 as in the original paper.\n\n  The image sizes must be at least 11x11 because of the filter size.\n\n  Example:\n\n  ```python\n      # Read images (of size 255 x 255) from file.\n      im1 = tf.image.decode_image(tf.io.read_file('path/to/im1.png'))\n      im2 = tf.image.decode_image(tf.io.read_file('path/to/im2.png'))\n      tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`\n      tf.shape(im2)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`\n      # Add an outer batch for each image.\n      im1 = tf.expand_dims(im1, axis=0)\n      im2 = tf.expand_dims(im2, axis=0)\n      # Compute SSIM over tf.uint8 Tensors.\n      ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,\n                            filter_sigma=1.5, k1=0.01, k2=0.03)\n\n      # Compute SSIM over tf.float32 Tensors.\n      im1 = tf.image.convert_image_dtype(im1, tf.float32)\n      im2 = tf.image.convert_image_dtype(im2, tf.float32)\n      ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,\n                            filter_sigma=1.5, k1=0.01, k2=0.03)\n      # ssim1 and ssim2 both have type tf.float32 and are almost equal.\n  ```\n\n  Args:\n    img1: First image batch. 4-D Tensor of shape `[batch, height, width,\n      channels]` with only Positive Pixel Values.\n    img2: Second image batch. 4-D Tensor of shape `[batch, height, width,\n      channels]` with only Positive Pixel Values.\n    max_val: The dynamic range of the images (i.e., the difference between the\n      maximum the and minimum allowed values).\n    filter_size: Default value 11 (size of gaussian filter).\n    filter_sigma: Default value 1.5 (width of gaussian filter).\n    k1: Default value 0.01\n    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so\n      it would be better if we took the values in the range of 0 < K2 < 0.4).\n    return_index_map: If True returns local SSIM map instead of the global mean.\n\n  Returns:\n    A tensor containing an SSIM value for each image in batch or a tensor\n    containing an SSIM value for each pixel for each image in batch if\n    return_index_map is True. Returned SSIM values are in range (-1, 1], when\n    pixel values are non-negative. Returns a tensor with shape:\n    broadcast(img1.shape[:-3], img2.shape[:-3]) or broadcast(img1.shape[:-1],\n    img2.shape[:-1]).\n  "
    with ops.name_scope(None, 'SSIM', [img1, img2]):
        img1 = ops.convert_to_tensor(img1, name='img1')
        img2 = ops.convert_to_tensor(img2, name='img2')
        (_, _, checks) = _verify_compatible_image_shapes(img1, img2)
        with ops.control_dependencies(checks):
            img1 = array_ops.identity(img1)
        max_val = math_ops.cast(max_val, img1.dtype)
        max_val = convert_image_dtype(max_val, dtypes.float32)
        img1 = convert_image_dtype(img1, dtypes.float32)
        img2 = convert_image_dtype(img2, dtypes.float32)
        (ssim_per_channel, _) = _ssim_per_channel(img1, img2, max_val, filter_size, filter_sigma, k1, k2, return_index_map)
        return math_ops.reduce_mean(ssim_per_channel, [-1])
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

@tf_export('image.ssim_multiscale')
@dispatch.add_dispatch_support
def ssim_multiscale(img1, img2, max_val, power_factors=_MSSSIM_WEIGHTS, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    if False:
        while True:
            i = 10
    'Computes the MS-SSIM between img1 and img2.\n\n  This function assumes that `img1` and `img2` are image batches, i.e. the last\n  three dimensions are [height, width, channels].\n\n  Note: The true SSIM is only defined on grayscale.  This function does not\n  perform any colorspace transform.  (If the input is already YUV, then it will\n  compute YUV SSIM average.)\n\n  Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale\n  structural similarity for image quality assessment." Signals, Systems and\n  Computers, 2004.\n\n  Args:\n    img1: First image batch with only Positive Pixel Values.\n    img2: Second image batch with only Positive Pixel Values. Must have the\n    same rank as img1.\n    max_val: The dynamic range of the images (i.e., the difference between the\n      maximum the and minimum allowed values).\n    power_factors: Iterable of weights for each of the scales. The number of\n      scales used is the length of the list. Index 0 is the unscaled\n      resolution\'s weight and each increasing scale corresponds to the image\n      being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,\n      0.1333), which are the values obtained in the original paper.\n    filter_size: Default value 11 (size of gaussian filter).\n    filter_sigma: Default value 1.5 (width of gaussian filter).\n    k1: Default value 0.01\n    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so\n      it would be better if we took the values in the range of 0 < K2 < 0.4).\n\n  Returns:\n    A tensor containing an MS-SSIM value for each image in batch.  The values\n    are in range [0, 1].  Returns a tensor with shape:\n    broadcast(img1.shape[:-3], img2.shape[:-3]).\n  '
    with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
        img1 = ops.convert_to_tensor(img1, name='img1')
        img2 = ops.convert_to_tensor(img2, name='img2')
        (shape1, shape2, checks) = _verify_compatible_image_shapes(img1, img2)
        with ops.control_dependencies(checks):
            img1 = array_ops.identity(img1)
        max_val = math_ops.cast(max_val, img1.dtype)
        max_val = convert_image_dtype(max_val, dtypes.float32)
        img1 = convert_image_dtype(img1, dtypes.float32)
        img2 = convert_image_dtype(img2, dtypes.float32)
        imgs = [img1, img2]
        shapes = [shape1, shape2]
        heads = [s[:-3] for s in shapes]
        tails = [s[-3:] for s in shapes]
        divisor = [1, 2, 2, 1]
        divisor_tensor = constant_op.constant(divisor[1:], dtype=dtypes.int32)

        def do_pad(images, remainder):
            if False:
                return 10
            padding = array_ops.expand_dims(remainder, -1)
            padding = array_ops.pad(padding, [[1, 0], [1, 0]])
            return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]
        mcs = []
        for k in range(len(power_factors)):
            with ops.name_scope(None, 'Scale%d' % k, imgs):
                if k > 0:
                    flat_imgs = [array_ops.reshape(x, array_ops.concat([[-1], t], 0)) for (x, t) in zip(imgs, tails)]
                    remainder = tails[0] % divisor_tensor
                    need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
                    padded = tf_cond.cond(need_padding, lambda : do_pad(flat_imgs, remainder), lambda : flat_imgs)
                    downscaled = [nn_ops.avg_pool(x, ksize=divisor, strides=divisor, padding='VALID') for x in padded]
                    tails = [x[1:] for x in array_ops.shape_n(downscaled)]
                    imgs = [array_ops.reshape(x, array_ops.concat([h, t], 0)) for (x, h, t) in zip(downscaled, heads, tails)]
                (ssim_per_channel, cs) = _ssim_per_channel(*imgs, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
                mcs.append(nn_ops.relu(cs))
        mcs.pop()
        mcs_and_ssim = array_ops_stack.stack(mcs + [nn_ops.relu(ssim_per_channel)], axis=-1)
        ms_ssim = math_ops.reduce_prod(math_ops.pow(mcs_and_ssim, power_factors), [-1])
        return math_ops.reduce_mean(ms_ssim, [-1])

@tf_export('image.image_gradients')
@dispatch.add_dispatch_support
def image_gradients(image):
    if False:
        i = 10
        return i + 15
    'Returns image gradients (dy, dx) for each color channel.\n\n  Both output tensors have the same shape as the input: [batch_size, h, w,\n  d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in\n  location (x, y). That means that dy will always have zeros in the last row,\n  and dx will always have zeros in the last column.\n\n  Usage Example:\n    ```python\n    BATCH_SIZE = 1\n    IMAGE_HEIGHT = 5\n    IMAGE_WIDTH = 5\n    CHANNELS = 1\n    image = tf.reshape(tf.range(IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS,\n      delta=1, dtype=tf.float32),\n      shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))\n    dy, dx = tf.image.image_gradients(image)\n    print(image[0, :,:,0])\n    tf.Tensor(\n      [[ 0.  1.  2.  3.  4.]\n      [ 5.  6.  7.  8.  9.]\n      [10. 11. 12. 13. 14.]\n      [15. 16. 17. 18. 19.]\n      [20. 21. 22. 23. 24.]], shape=(5, 5), dtype=float32)\n    print(dy[0, :,:,0])\n    tf.Tensor(\n      [[5. 5. 5. 5. 5.]\n      [5. 5. 5. 5. 5.]\n      [5. 5. 5. 5. 5.]\n      [5. 5. 5. 5. 5.]\n      [0. 0. 0. 0. 0.]], shape=(5, 5), dtype=float32)\n    print(dx[0, :,:,0])\n    tf.Tensor(\n      [[1. 1. 1. 1. 0.]\n      [1. 1. 1. 1. 0.]\n      [1. 1. 1. 1. 0.]\n      [1. 1. 1. 1. 0.]\n      [1. 1. 1. 1. 0.]], shape=(5, 5), dtype=float32)\n    ```\n\n  Args:\n    image: Tensor with shape [batch_size, h, w, d].\n\n  Returns:\n    Pair of tensors (dy, dx) holding the vertical and horizontal image\n    gradients (1-step finite difference).\n\n  Raises:\n    ValueError: If `image` is not a 4D tensor.\n  '
    if image.get_shape().ndims != 4:
        raise ValueError('image_gradients expects a 4D tensor [batch_size, h, w, d], not {}.'.format(image.get_shape()))
    image_shape = array_ops.shape(image)
    (batch_size, height, width, depth) = array_ops_stack.unstack(image_shape)
    dy = image[:, 1:, :, :] - image[:, :-1, :, :]
    dx = image[:, :, 1:, :] - image[:, :, :-1, :]
    shape = array_ops_stack.stack([batch_size, 1, width, depth])
    dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
    dy = array_ops.reshape(dy, image_shape)
    shape = array_ops_stack.stack([batch_size, height, 1, depth])
    dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
    dx = array_ops.reshape(dx, image_shape)
    return (dy, dx)

@tf_export('image.sobel_edges')
@dispatch.add_dispatch_support
def sobel_edges(image):
    if False:
        print('Hello World!')
    "Returns a tensor holding Sobel edge maps.\n\n  Example usage:\n\n  For general usage, `image` would be loaded from a file as below:\n\n  ```python\n  image_bytes = tf.io.read_file(path_to_image_file)\n  image = tf.image.decode_image(image_bytes)\n  image = tf.cast(image, tf.float32)\n  image = tf.expand_dims(image, 0)\n  ```\n  But for demo purposes, we are using randomly generated values for `image`:\n\n  >>> image = tf.random.uniform(\n  ...   maxval=255, shape=[1, 28, 28, 3], dtype=tf.float32)\n  >>> sobel = tf.image.sobel_edges(image)\n  >>> sobel_y = np.asarray(sobel[0, :, :, :, 0]) # sobel in y-direction\n  >>> sobel_x = np.asarray(sobel[0, :, :, :, 1]) # sobel in x-direction\n\n  For displaying the sobel results, PIL's [Image Module](\n  https://pillow.readthedocs.io/en/stable/reference/Image.html) can be used:\n\n  ```python\n  # Display edge maps for the first channel (at index 0)\n  Image.fromarray(sobel_y[..., 0] / 4 + 0.5).show()\n  Image.fromarray(sobel_x[..., 0] / 4 + 0.5).show()\n  ```\n\n  Args:\n    image: Image tensor with shape [batch_size, h, w, d] and type float32 or\n      float64.  The image(s) must be 2x2 or larger.\n\n  Returns:\n    Tensor holding edge maps for each channel. Returns a tensor with shape\n    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],\n    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.\n  "
    static_image_shape = image.get_shape()
    image_shape = array_ops.shape(image)
    kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    num_kernels = len(kernels)
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2)
    kernels_tf = constant_op.constant(kernels, dtype=image.dtype)
    kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')
    pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
    padded = array_ops.pad(image, pad_sizes, mode='REFLECT')
    strides = [1, 1, 1, 1]
    output = nn_impl.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
    shape = array_ops.concat([image_shape, [num_kernels]], 0)
    output = array_ops.reshape(output, shape=shape)
    output.set_shape(static_image_shape.concatenate([num_kernels]))
    return output

@tf_export(v1=['image.resize_bicubic'])
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions='Use `tf.image.resize(...method=ResizeMethod.BICUBIC...)` instead.')
def resize_bicubic(images, size, align_corners=False, name=None, half_pixel_centers=False):
    if False:
        return 10
    return gen_image_ops.resize_bicubic(images=images, size=size, align_corners=align_corners, half_pixel_centers=half_pixel_centers, name=name)

@tf_export(v1=['image.resize_bilinear'])
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions='Use `tf.image.resize(...method=ResizeMethod.BILINEAR...)` instead.')
def resize_bilinear(images, size, align_corners=False, name=None, half_pixel_centers=False):
    if False:
        return 10
    return gen_image_ops.resize_bilinear(images=images, size=size, align_corners=align_corners, half_pixel_centers=half_pixel_centers, name=name)

@tf_export(v1=['image.resize_nearest_neighbor'])
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions='Use `tf.image.resize(...method=ResizeMethod.NEAREST_NEIGHBOR...)` instead.')
def resize_nearest_neighbor(images, size, align_corners=False, name=None, half_pixel_centers=False):
    if False:
        return 10
    return gen_image_ops.resize_nearest_neighbor(images=images, size=size, align_corners=align_corners, half_pixel_centers=half_pixel_centers, name=name)
resize_area_deprecation = deprecation.deprecated(date=None, instructions='Use `tf.image.resize(...method=ResizeMethod.AREA...)` instead.')
resize_area = tf_export(v1=['image.resize_area'])(resize_area_deprecation(dispatch.add_dispatch_support(gen_image_ops.resize_area)))

@tf_export('image.crop_and_resize', v1=[])
@dispatch.add_dispatch_support
def crop_and_resize_v2(image, boxes, box_indices, crop_size, method='bilinear', extrapolation_value=0.0, name=None):
    if False:
        while True:
            i = 10
    'Extracts crops from the input image tensor and resizes them.\n\n  Extracts crops from the input image tensor and resizes them using bilinear\n  sampling or nearest neighbor sampling (possibly with aspect ratio change) to a\n  common output size specified by `crop_size`. This is more general than the\n  `crop_to_bounding_box` op which extracts a fixed size slice from the input\n  image and does not allow resizing or aspect ratio change. The crops occur\n  first and then the resize.\n\n  Returns a tensor with `crops` from the input `image` at positions defined at\n  the bounding box locations in `boxes`. The cropped boxes are all resized (with\n  bilinear or nearest neighbor interpolation) to a fixed\n  `size = [crop_height, crop_width]`. The result is a 4-D tensor\n  `[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.\n  In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical\n  results to using `tf.compat.v1.image.resize_bilinear()` or\n  `tf.compat.v1.image.resize_nearest_neighbor()`(depends on the `method`\n  argument) with\n  `align_corners=True`.\n\n  Args:\n    image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.\n      Both `image_height` and `image_width` need to be positive.\n    boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor\n      specifies the coordinates of a box in the `box_ind[i]` image and is\n      specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized\n      coordinate value of `y` is mapped to the image coordinate at `y *\n      (image_height - 1)`, so as the `[0, 1]` interval of normalized image\n      height is mapped to `[0, image_height - 1]` in image height coordinates.\n      We do allow `y1` > `y2`, in which case the sampled crop is an up-down\n      flipped version of the original image. The width dimension is treated\n      similarly. Normalized coordinates outside the `[0, 1]` range are allowed,\n      in which case we use `extrapolation_value` to extrapolate the input image\n      values.\n    box_indices: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0,\n      batch)`. The value of `box_ind[i]` specifies the image that the `i`-th box\n      refers to.\n    crop_size: A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`.\n      All cropped image patches are resized to this size. The aspect ratio of\n      the image content is not preserved. Both `crop_height` and `crop_width`\n      need to be positive.\n    method: An optional string specifying the sampling method for resizing. It\n      can be either `"bilinear"` or `"nearest"` and default to `"bilinear"`.\n      Currently two sampling methods are supported: Bilinear and Nearest\n        Neighbor.\n    extrapolation_value: An optional `float`. Defaults to `0.0`. Value used for\n      extrapolation, when applicable.\n    name: A name for the operation (optional).\n\n  Returns:\n    A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.\n\n  Usage example:\n\n  >>> BATCH_SIZE = 1\n  >>> NUM_BOXES = 5\n  >>> IMAGE_HEIGHT = 256\n  >>> IMAGE_WIDTH = 256\n  >>> CHANNELS = 3\n  >>> CROP_SIZE = (24, 24)\n\n  >>> image = tf.random.normal(shape=(\n  ...   BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) )\n  >>> boxes = tf.random.uniform(shape=(NUM_BOXES, 4))\n  >>> box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0,\n  ...   maxval=BATCH_SIZE, dtype=tf.int32)\n  >>> output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)\n  >>> output.shape\n  TensorShape([5, 24, 24, 3])\n\n  Example with linear interpolation:\n\n  >>> image = np.arange(0, 18, 2).astype(\'float32\').reshape(3, 3)\n  >>> result = tf.image.crop_and_resize(\n  ...   image[None, :, :, None],\n  ...   np.asarray([[0.5,0.5,1,1]]), [0], [3, 3], method=\'bilinear\')\n  >>> result[0][:, :, 0]\n  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=\n    array([[ 8.,  9., 10.],\n           [11., 12., 13.],\n           [14., 15., 16.]], dtype=float32)>\n\n  Example with nearest interpolation:\n\n  >>> image = np.arange(0, 18, 2).astype(\'float32\').reshape(3, 3)\n  >>> result = tf.image.crop_and_resize(\n  ...   image[None, :, :, None],\n  ...   np.asarray([[0.5,0.5,1,1]]), [0], [3, 3], method=\'nearest\')\n  >>> result[0][:, :, 0]\n  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=\n    array([[ 8., 10., 10.],\n           [14., 16., 16.],\n           [14., 16., 16.]], dtype=float32)>\n\n\n  '
    return gen_image_ops.crop_and_resize(image, boxes, box_indices, crop_size, method, extrapolation_value, name)

@tf_export(v1=['image.crop_and_resize'])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, 'box_ind is deprecated, use box_indices instead', 'box_ind')
def crop_and_resize_v1(image, boxes, box_ind=None, crop_size=None, method='bilinear', extrapolation_value=0, name=None, box_indices=None):
    if False:
        for i in range(10):
            print('nop')
    box_ind = deprecation.deprecated_argument_lookup('box_indices', box_indices, 'box_ind', box_ind)
    return gen_image_ops.crop_and_resize(image, boxes, box_ind, crop_size, method, extrapolation_value, name)
crop_and_resize_v1.__doc__ = gen_image_ops.crop_and_resize.__doc__

@tf_export(v1=['image.extract_glimpse'])
@dispatch.add_dispatch_support
def extract_glimpse(input, size, offsets, centered=True, normalized=True, uniform_noise=True, name=None):
    if False:
        return 10
    'Extracts a glimpse from the input tensor.\n\n  Returns a set of windows called glimpses extracted at location\n  `offsets` from the input tensor. If the windows only partially\n  overlaps the inputs, the non-overlapping areas will be filled with\n  random noise.\n\n  The result is a 4-D tensor of shape `[batch_size, glimpse_height,\n  glimpse_width, channels]`. The channels and batch dimensions are the\n  same as that of the input tensor. The height and width of the output\n  windows are specified in the `size` parameter.\n\n  The argument `normalized` and `centered` controls how the windows are built:\n\n  * If the coordinates are normalized but not centered, 0.0 and 1.0\n    correspond to the minimum and maximum of each height and width\n    dimension.\n  * If the coordinates are both normalized and centered, they range from\n    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper\n    left corner, the lower right corner is located at (1.0, 1.0) and the\n    center is at (0, 0).\n  * If the coordinates are not normalized they are interpreted as\n    numbers of pixels.\n\n  Usage Example:\n\n  >>> x = [[[[0.0],\n  ...           [1.0],\n  ...           [2.0]],\n  ...          [[3.0],\n  ...           [4.0],\n  ...           [5.0]],\n  ...          [[6.0],\n  ...           [7.0],\n  ...           [8.0]]]]\n  >>> tf.compat.v1.image.extract_glimpse(x, size=(2, 2), offsets=[[1, 1]],\n  ...                                    centered=False, normalized=False)\n  <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=\n  array([[[[0.],\n           [1.]],\n          [[3.],\n           [4.]]]], dtype=float32)>\n\n  Args:\n    input: A `Tensor` of type `float32`. A 4-D float tensor of shape\n      `[batch_size, height, width, channels]`.\n    size: A `Tensor` of type `int32`. A 1-D tensor of 2 elements containing the\n      size of the glimpses to extract.  The glimpse height must be specified\n      first, following by the glimpse width.\n    offsets: A `Tensor` of type `float32`. A 2-D integer tensor of shape\n      `[batch_size, 2]` containing the y, x locations of the center of each\n      window.\n    centered: An optional `bool`. Defaults to `True`. indicates if the offset\n      coordinates are centered relative to the image, in which case the (0, 0)\n      offset is relative to the center of the input images. If false, the (0,0)\n      offset corresponds to the upper left corner of the input images.\n    normalized: An optional `bool`. Defaults to `True`. indicates if the offset\n      coordinates are normalized.\n    uniform_noise: An optional `bool`. Defaults to `True`. indicates if the\n      noise should be generated using a uniform distribution or a Gaussian\n      distribution.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` of type `float32`.\n  '
    return gen_image_ops.extract_glimpse(input=input, size=size, offsets=offsets, centered=centered, normalized=normalized, uniform_noise=uniform_noise, name=name)

@tf_export('image.extract_glimpse', v1=[])
@dispatch.add_dispatch_support
def extract_glimpse_v2(input, size, offsets, centered=True, normalized=True, noise='uniform', name=None):
    if False:
        for i in range(10):
            print('nop')
    'Extracts a glimpse from the input tensor.\n\n  Returns a set of windows called glimpses extracted at location\n  `offsets` from the input tensor. If the windows only partially\n  overlaps the inputs, the non-overlapping areas will be filled with\n  random noise.\n\n  The result is a 4-D tensor of shape `[batch_size, glimpse_height,\n  glimpse_width, channels]`. The channels and batch dimensions are the\n  same as that of the input tensor. The height and width of the output\n  windows are specified in the `size` parameter.\n\n  The argument `normalized` and `centered` controls how the windows are built:\n\n  * If the coordinates are normalized but not centered, 0.0 and 1.0\n    correspond to the minimum and maximum of each height and width\n    dimension.\n  * If the coordinates are both normalized and centered, they range from\n    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper\n    left corner, the lower right corner is located at (1.0, 1.0) and the\n    center is at (0, 0).\n  * If the coordinates are not normalized they are interpreted as\n    numbers of pixels.\n\n  Usage Example:\n\n  >>> x = [[[[0.0],\n  ...           [1.0],\n  ...           [2.0]],\n  ...          [[3.0],\n  ...           [4.0],\n  ...           [5.0]],\n  ...          [[6.0],\n  ...           [7.0],\n  ...           [8.0]]]]\n  >>> tf.image.extract_glimpse(x, size=(2, 2), offsets=[[1, 1]],\n  ...                         centered=False, normalized=False)\n  <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=\n  array([[[[4.],\n           [5.]],\n          [[7.],\n           [8.]]]], dtype=float32)>\n\n  Args:\n    input: A `Tensor` of type `float32`. A 4-D float tensor of shape\n      `[batch_size, height, width, channels]`.\n    size: A `Tensor` of type `int32`. A 1-D tensor of 2 elements containing the\n      size of the glimpses to extract.  The glimpse height must be specified\n      first, following by the glimpse width.\n    offsets: A `Tensor` of type `float32`. A 2-D integer tensor of shape\n      `[batch_size, 2]` containing the y, x locations of the center of each\n      window.\n    centered: An optional `bool`. Defaults to `True`. indicates if the offset\n      coordinates are centered relative to the image, in which case the (0, 0)\n      offset is relative to the center of the input images. If false, the (0,0)\n      offset corresponds to the upper left corner of the input images.\n    normalized: An optional `bool`. Defaults to `True`. indicates if the offset\n      coordinates are normalized.\n    noise: An optional `string`. Defaults to `uniform`. indicates if the noise\n      should be `uniform` (uniform distribution), `gaussian` (gaussian\n      distribution), or `zero` (zero padding).\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` of type `float32`.\n  '
    return gen_image_ops.extract_glimpse_v2(input=input, size=size, offsets=offsets, centered=centered, normalized=normalized, noise=noise, uniform_noise=False, name=name)

@tf_export('image.combined_non_max_suppression')
@dispatch.add_dispatch_support
def combined_non_max_suppression(boxes, scores, max_output_size_per_class, max_total_size, iou_threshold=0.5, score_threshold=float('-inf'), pad_per_class=False, clip_boxes=True, name=None):
    if False:
        print('Hello World!')
    "Greedily selects a subset of bounding boxes in descending order of score.\n\n  This operation performs non_max_suppression on the inputs per batch, across\n  all classes.\n  Prunes away boxes that have high intersection-over-union (IOU) overlap\n  with previously selected boxes.  Bounding boxes are supplied as\n  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any\n  diagonal pair of box corners and the coordinates can be provided as normalized\n  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm\n  is agnostic to where the origin is in the coordinate system. Also note that\n  this algorithm is invariant to orthogonal transformations and translations\n  of the coordinate system; thus translating or reflections of the coordinate\n  system result in the same boxes being selected by the algorithm.\n  The output of this operation is the final boxes, scores and classes tensor\n  returned after performing non_max_suppression.\n\n  Args:\n    boxes: A 4-D float `Tensor` of shape `[batch_size, num_boxes, q, 4]`. If `q`\n      is 1 then same boxes are used for all classes otherwise, if `q` is equal\n      to number of classes, class-specific boxes are used.\n    scores: A 3-D float `Tensor` of shape `[batch_size, num_boxes, num_classes]`\n      representing a single score corresponding to each box (each row of boxes).\n    max_output_size_per_class: A scalar integer `Tensor` representing the\n      maximum number of boxes to be selected by non-max suppression per class\n    max_total_size: A int32 scalar representing maximum number of boxes retained\n      over all classes. Note that setting this value to a large number may\n      result in OOM error depending on the system workload.\n    iou_threshold: A float representing the threshold for deciding whether boxes\n      overlap too much with respect to IOU.\n    score_threshold: A float representing the threshold for deciding when to\n      remove boxes based on score.\n    pad_per_class: If false, the output nmsed boxes, scores and classes are\n      padded/clipped to `max_total_size`. If true, the output nmsed boxes,\n      scores and classes are padded to be of length\n      `max_size_per_class`*`num_classes`, unless it exceeds `max_total_size` in\n      which case it is clipped to `max_total_size`. Defaults to false.\n    clip_boxes: If true, the coordinates of output nmsed boxes will be clipped\n      to [0, 1]. If false, output the box coordinates as it is. Defaults to\n      true.\n    name: A name for the operation (optional).\n\n  Returns:\n    'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor\n      containing the non-max suppressed boxes.\n    'nmsed_scores': A [batch_size, max_detections] float32 tensor containing\n      the scores for the boxes.\n    'nmsed_classes': A [batch_size, max_detections] float32 tensor\n      containing the class for boxes.\n    'valid_detections': A [batch_size] int32 tensor indicating the number of\n      valid detections per batch item. Only the top valid_detections[i] entries\n      in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the\n      entries are zero paddings.\n  "
    with ops.name_scope(name, 'combined_non_max_suppression'):
        iou_threshold = ops.convert_to_tensor(iou_threshold, dtype=dtypes.float32, name='iou_threshold')
        score_threshold = ops.convert_to_tensor(score_threshold, dtype=dtypes.float32, name='score_threshold')
        max_total_size = ops.convert_to_tensor(max_total_size)
        return gen_image_ops.combined_non_max_suppression(boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold, pad_per_class, clip_boxes)

def _bbox_overlap(boxes_a, boxes_b):
    if False:
        while True:
            i = 10
    'Calculates the overlap (iou - intersection over union) between boxes_a and boxes_b.\n\n  Args:\n    boxes_a: a tensor with a shape of [batch_size, N, 4]. N is the number of\n      boxes per image. The last dimension is the pixel coordinates in\n      [ymin, xmin, ymax, xmax] form.\n    boxes_b: a tensor with a shape of [batch_size, M, 4]. M is the number of\n      boxes. The last dimension is the pixel coordinates in\n      [ymin, xmin, ymax, xmax] form.\n  Returns:\n    intersection_over_union: a tensor with as a shape of [batch_size, N, M],\n    representing the ratio of intersection area over union area (IoU) between\n    two boxes\n  '
    with ops.name_scope('bbox_overlap'):
        (a_y_min, a_x_min, a_y_max, a_x_max) = array_ops.split(value=boxes_a, num_or_size_splits=4, axis=2)
        (b_y_min, b_x_min, b_y_max, b_x_max) = array_ops.split(value=boxes_b, num_or_size_splits=4, axis=2)
        i_xmin = math_ops.maximum(a_x_min, array_ops.transpose(b_x_min, [0, 2, 1]))
        i_xmax = math_ops.minimum(a_x_max, array_ops.transpose(b_x_max, [0, 2, 1]))
        i_ymin = math_ops.maximum(a_y_min, array_ops.transpose(b_y_min, [0, 2, 1]))
        i_ymax = math_ops.minimum(a_y_max, array_ops.transpose(b_y_max, [0, 2, 1]))
        i_area = math_ops.maximum(i_xmax - i_xmin, 0) * math_ops.maximum(i_ymax - i_ymin, 0)
        a_area = (a_y_max - a_y_min) * (a_x_max - a_x_min)
        b_area = (b_y_max - b_y_min) * (b_x_max - b_x_min)
        EPSILON = 1e-08
        u_area = a_area + array_ops.transpose(b_area, [0, 2, 1]) - i_area + EPSILON
        intersection_over_union = i_area / u_area
        return intersection_over_union

def _self_suppression(iou, _, iou_sum, iou_threshold):
    if False:
        i = 10
        return i + 15
    'Suppress boxes in the same tile.\n\n     Compute boxes that cannot be suppressed by others (i.e.,\n     can_suppress_others), and then use them to suppress boxes in the same tile.\n\n  Args:\n    iou: a tensor of shape [batch_size, num_boxes_with_padding] representing\n    intersection over union.\n    iou_sum: a scalar tensor.\n    iou_threshold: a scalar tensor.\n\n  Returns:\n    iou_suppressed: a tensor of shape [batch_size, num_boxes_with_padding].\n    iou_diff: a scalar tensor representing whether any box is supressed in\n      this step.\n    iou_sum_new: a scalar tensor of shape [batch_size] that represents\n      the iou sum after suppression.\n    iou_threshold: a scalar tensor.\n  '
    batch_size = array_ops.shape(iou)[0]
    can_suppress_others = math_ops.cast(array_ops.reshape(math_ops.reduce_max(iou, 1) < iou_threshold, [batch_size, -1, 1]), iou.dtype)
    iou_after_suppression = array_ops.reshape(math_ops.cast(math_ops.reduce_max(can_suppress_others * iou, 1) < iou_threshold, iou.dtype), [batch_size, -1, 1]) * iou
    iou_sum_new = math_ops.reduce_sum(iou_after_suppression, [1, 2])
    return [iou_after_suppression, math_ops.reduce_any(iou_sum - iou_sum_new > iou_threshold), iou_sum_new, iou_threshold]

def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx, tile_size):
    if False:
        i = 10
        return i + 15
    'Suppress boxes between different tiles.\n\n  Args:\n    boxes: a tensor of shape [batch_size, num_boxes_with_padding, 4]\n    box_slice: a tensor of shape [batch_size, tile_size, 4]\n    iou_threshold: a scalar tensor\n    inner_idx: a scalar tensor representing the tile index of the tile\n      that is used to supress box_slice\n    tile_size: an integer representing the number of boxes in a tile\n\n  Returns:\n    boxes: unchanged boxes as input\n    box_slice_after_suppression: box_slice after suppression\n    iou_threshold: unchanged\n  '
    batch_size = array_ops.shape(boxes)[0]
    new_slice = array_ops.slice(boxes, [0, inner_idx * tile_size, 0], [batch_size, tile_size, 4])
    iou = _bbox_overlap(new_slice, box_slice)
    box_slice_after_suppression = array_ops.expand_dims(math_ops.cast(math_ops.reduce_all(iou < iou_threshold, [1]), box_slice.dtype), 2) * box_slice
    return (boxes, box_slice_after_suppression, iou_threshold, inner_idx + 1)

def _suppression_loop_body(boxes, iou_threshold, output_size, idx, tile_size):
    if False:
        while True:
            i = 10
    'Process boxes in the range [idx*tile_size, (idx+1)*tile_size).\n\n  Args:\n    boxes: a tensor with a shape of [batch_size, anchors, 4].\n    iou_threshold: a float representing the threshold for deciding whether boxes\n      overlap too much with respect to IOU.\n    output_size: an int32 tensor of size [batch_size]. Representing the number\n      of selected boxes for each batch.\n    idx: an integer scalar representing induction variable.\n    tile_size: an integer representing the number of boxes in a tile\n\n  Returns:\n    boxes: updated boxes.\n    iou_threshold: pass down iou_threshold to the next iteration.\n    output_size: the updated output_size.\n    idx: the updated induction variable.\n  '
    with ops.name_scope('suppression_loop_body'):
        num_tiles = array_ops.shape(boxes)[1] // tile_size
        batch_size = array_ops.shape(boxes)[0]

        def cross_suppression_func(boxes, box_slice, iou_threshold, inner_idx):
            if False:
                return 10
            return _cross_suppression(boxes, box_slice, iou_threshold, inner_idx, tile_size)
        box_slice = array_ops.slice(boxes, [0, idx * tile_size, 0], [batch_size, tile_size, 4])
        (_, box_slice, _, _) = while_loop.while_loop(lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx, cross_suppression_func, [boxes, box_slice, iou_threshold, constant_op.constant(0)])
        iou = _bbox_overlap(box_slice, box_slice)
        mask = array_ops.expand_dims(array_ops.reshape(math_ops.range(tile_size), [1, -1]) > array_ops.reshape(math_ops.range(tile_size), [-1, 1]), 0)
        iou *= math_ops.cast(math_ops.logical_and(mask, iou >= iou_threshold), iou.dtype)
        (suppressed_iou, _, _, _) = while_loop.while_loop(lambda _iou, loop_condition, _iou_sum, _: loop_condition, _self_suppression, [iou, constant_op.constant(True), math_ops.reduce_sum(iou, [1, 2]), iou_threshold])
        suppressed_box = math_ops.reduce_sum(suppressed_iou, 1) > 0
        box_slice *= array_ops.expand_dims(1.0 - math_ops.cast(suppressed_box, box_slice.dtype), 2)
        mask = array_ops.reshape(math_ops.cast(math_ops.equal(math_ops.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
        boxes = array_ops.tile(array_ops.expand_dims(box_slice, [1]), [1, num_tiles, 1, 1]) * mask + array_ops.reshape(boxes, [batch_size, num_tiles, tile_size, 4]) * (1 - mask)
        boxes = array_ops.reshape(boxes, [batch_size, -1, 4])
        output_size += math_ops.reduce_sum(math_ops.cast(math_ops.reduce_any(box_slice > 0, [2]), dtypes.int32), [1])
    return (boxes, iou_threshold, output_size, idx + 1)

@tf_export('image.non_max_suppression_padded')
@dispatch.add_dispatch_support
def non_max_suppression_padded(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), pad_to_max_output_size=False, name=None, sorted_input=False, canonicalized_coordinates=False, tile_size=512):
    if False:
        while True:
            i = 10
    'Greedily selects a subset of bounding boxes in descending order of score.\n\n  Performs algorithmically equivalent operation to tf.image.non_max_suppression,\n  with the addition of an optional parameter which zero-pads the output to\n  be of size `max_output_size`.\n  The output of this operation is a tuple containing the set of integers\n  indexing into the input collection of bounding boxes representing the selected\n  boxes and the number of valid indices in the index set.  The bounding box\n  coordinates corresponding to the selected indices can then be obtained using\n  the `tf.slice` and `tf.gather` operations.  For example:\n    ```python\n    selected_indices_padded, num_valid = tf.image.non_max_suppression_padded(\n        boxes, scores, max_output_size, iou_threshold,\n        score_threshold, pad_to_max_output_size=True)\n    selected_indices = tf.slice(\n        selected_indices_padded, tf.constant([0]), num_valid)\n    selected_boxes = tf.gather(boxes, selected_indices)\n    ```\n\n  Args:\n    boxes: a tensor of rank 2 or higher with a shape of [..., num_boxes, 4].\n      Dimensions except the last two are batch dimensions.\n    scores: a tensor of rank 1 or higher with a shape of [..., num_boxes].\n    max_output_size: a scalar integer `Tensor` representing the maximum number\n      of boxes to be selected by non max suppression. Note that setting this\n      value to a large number may result in OOM error depending on the system\n      workload.\n    iou_threshold: a float representing the threshold for deciding whether boxes\n      overlap too much with respect to IoU (intersection over union).\n    score_threshold: a float representing the threshold for box scores. Boxes\n      with a score that is not larger than this threshold will be suppressed.\n    pad_to_max_output_size: whether to pad the output idx to max_output_size.\n      Must be set to True when the input is a batch of images.\n    name: name of operation.\n    sorted_input: a boolean indicating whether the input boxes and scores\n      are sorted in descending order by the score.\n    canonicalized_coordinates: if box coordinates are given as\n    `[y_min, x_min, y_max, x_max]`, setting to True eliminate redundant\n     computation to canonicalize box coordinates.\n    tile_size: an integer representing the number of boxes in a tile, i.e.,\n      the maximum number of boxes per image that can be used to suppress other\n      boxes in parallel; larger tile_size means larger parallelism and\n      potentially more redundant work.\n  Returns:\n    idx: a tensor with a shape of [..., num_boxes] representing the\n      indices selected by non-max suppression. The leading dimensions\n      are the batch dimensions of the input boxes. All numbers are within\n      [0, num_boxes). For each image (i.e., idx[i]), only the first num_valid[i]\n      indices (i.e., idx[i][:num_valid[i]]) are valid.\n    num_valid: a tensor of rank 0 or higher with a shape of [...]\n      representing the number of valid indices in idx. Its dimensions are the\n      batch dimensions of the input boxes.\n   Raises:\n    ValueError: When set pad_to_max_output_size to False for batched input.\n  '
    with ops.name_scope(name, 'non_max_suppression_padded'):
        if not pad_to_max_output_size:
            if boxes.get_shape().rank is not None and boxes.get_shape().rank > 2:
                raise ValueError("'pad_to_max_output_size' (value {}) must be True for batched input".format(pad_to_max_output_size))
        if name is None:
            name = ''
        (idx, num_valid) = non_max_suppression_padded_v2(boxes, scores, max_output_size, iou_threshold, score_threshold, sorted_input, canonicalized_coordinates, tile_size)
        if not pad_to_max_output_size:
            idx = idx[0, :num_valid]
        else:
            batch_dims = array_ops.concat([array_ops.shape(boxes)[:-2], array_ops.expand_dims(max_output_size, 0)], 0)
            idx = array_ops.reshape(idx, batch_dims)
        return (idx, num_valid)

@def_function.function(experimental_implements='non_max_suppression_padded_v2')
def non_max_suppression_padded_v2(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), sorted_input=False, canonicalized_coordinates=False, tile_size=512):
    if False:
        return 10
    "Non-maximum suppression.\n\n  Prunes away boxes that have high intersection-over-union (IOU) overlap\n  with previously selected boxes. Bounding boxes are supplied as\n  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any\n  diagonal pair of box corners and the coordinates can be provided as normalized\n  (i.e., lying in the interval `[0, 1]`) or absolute. The bounding box\n  coordinates are cannonicalized to `[y_min, x_min, y_max, x_max]`,\n  where `(y_min, x_min)` and `(y_max, x_mas)` are the coordinates of the lower\n  left and upper right corner. User may indiciate the input box coordinates are\n  already canonicalized to eliminate redundant work by setting\n  canonicalized_coordinates to `True`. Note that this algorithm is agnostic to\n  where the origin is in the coordinate system. Note that this algorithm is\n  invariant to orthogonal transformations and translations of the coordinate\n  system; thus translating or reflections of the coordinate system result in the\n  same boxes being selected by the algorithm.\n\n  Similar to tf.image.non_max_suppression, non_max_suppression_padded\n  implements hard NMS but can operate on a batch of images and improves\n  performance by titling the bounding boxes. Non_max_suppression_padded should\n  be preferred over tf.image_non_max_suppression when running on devices with\n  abundant parallelsim for higher computation speed. For soft NMS, refer to\n  tf.image.non_max_suppression_with_scores.\n\n  While a serial NMS algorithm iteratively uses the highest-scored unprocessed\n  box to suppress boxes, this algorithm uses many boxes to suppress other boxes\n  in parallel. The key idea is to partition boxes into tiles based on their\n  score and suppresses boxes tile by tile, thus achieving parallelism within a\n  tile. The tile size determines the degree of parallelism.\n\n  In cross suppression (using boxes of tile A to suppress boxes of tile B),\n  all boxes in A can independently suppress boxes in B.\n\n  Self suppression (suppressing boxes of the same tile) needs to be iteratively\n  applied until there's no more suppression. In each iteration, boxes that\n  cannot be suppressed are used to suppress boxes in the same tile.\n\n  boxes = boxes.pad_to_multiply_of(tile_size)\n  num_tiles = len(boxes) // tile_size\n  output_boxes = []\n  for i in range(num_tiles):\n    box_tile = boxes[i*tile_size : (i+1)*tile_size]\n    for j in range(i - 1):\n      # in parallel suppress boxes in box_tile using boxes from suppressing_tile\n      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]\n      iou = _bbox_overlap(box_tile, suppressing_tile)\n      # if the box is suppressed in iou, clear it to a dot\n      box_tile *= _update_boxes(iou)\n    # Iteratively handle the diagnal tile.\n    iou = _box_overlap(box_tile, box_tile)\n    iou_changed = True\n    while iou_changed:\n      # boxes that are not suppressed by anything else\n      suppressing_boxes = _get_suppressing_boxes(iou)\n      # boxes that are suppressed by suppressing_boxes\n      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)\n      # clear iou to 0 for boxes that are suppressed, as they cannot be used\n      # to suppress other boxes any more\n      new_iou = _clear_iou(iou, suppressed_boxes)\n      iou_changed = (new_iou != iou)\n      iou = new_iou\n    # remaining boxes that can still suppress others, are selected boxes.\n    output_boxes.append(_get_suppressing_boxes(iou))\n    if len(output_boxes) >= max_output_size:\n      break\n\n  Args:\n    boxes: a tensor of rank 2 or higher with a shape of [..., num_boxes, 4].\n      Dimensions except the last two are batch dimensions. The last dimension\n      represents box coordinates, given as [y_1, x_1, y_2, x_2]. The coordinates\n      on each dimension can be given in any order\n      (see also `canonicalized_coordinates`) but must describe a box with\n      a positive area.\n    scores: a tensor of rank 1 or higher with a shape of [..., num_boxes].\n    max_output_size: a scalar integer `Tensor` representing the maximum number\n      of boxes to be selected by non max suppression.\n    iou_threshold: a float representing the threshold for deciding whether boxes\n      overlap too much with respect to IoU (intersection over union).\n    score_threshold: a float representing the threshold for box scores. Boxes\n      with a score that is not larger than this threshold will be suppressed.\n    sorted_input: a boolean indicating whether the input boxes and scores\n      are sorted in descending order by the score.\n    canonicalized_coordinates: if box coordinates are given as\n    `[y_min, x_min, y_max, x_max]`, setting to True eliminate redundant\n     computation to canonicalize box coordinates.\n    tile_size: an integer representing the number of boxes in a tile, i.e.,\n      the maximum number of boxes per image that can be used to suppress other\n      boxes in parallel; larger tile_size means larger parallelism and\n      potentially more redundant work.\n  Returns:\n    idx: a tensor with a shape of [..., num_boxes] representing the\n      indices selected by non-max suppression. The leading dimensions\n      are the batch dimensions of the input boxes. All numbers are within\n      [0, num_boxes). For each image (i.e., idx[i]), only the first num_valid[i]\n      indices (i.e., idx[i][:num_valid[i]]) are valid.\n    num_valid: a tensor of rank 0 or higher with a shape of [...]\n      representing the number of valid indices in idx. Its dimensions are the\n      batch dimensions of the input boxes.\n   Raises:\n    ValueError: When set pad_to_max_output_size to False for batched input.\n  "

    def _sort_scores_and_boxes(scores, boxes):
        if False:
            while True:
                i = 10
        'Sort boxes based their score from highest to lowest.\n\n    Args:\n      scores: a tensor with a shape of [batch_size, num_boxes] representing\n        the scores of boxes.\n      boxes: a tensor with a shape of [batch_size, num_boxes, 4] representing\n        the boxes.\n    Returns:\n      sorted_scores: a tensor with a shape of [batch_size, num_boxes]\n        representing the sorted scores.\n      sorted_boxes: a tensor representing the sorted boxes.\n      sorted_scores_indices: a tensor with a shape of [batch_size, num_boxes]\n        representing the index of the scores in a sorted descending order.\n    '
        with ops.name_scope('sort_scores_and_boxes'):
            sorted_scores_indices = sort_ops.argsort(scores, axis=1, direction='DESCENDING')
            sorted_scores = array_ops.gather(scores, sorted_scores_indices, axis=1, batch_dims=1)
            sorted_boxes = array_ops.gather(boxes, sorted_scores_indices, axis=1, batch_dims=1)
        return (sorted_scores, sorted_boxes, sorted_scores_indices)
    batch_dims = array_ops.shape(boxes)[:-2]
    num_boxes = array_ops.shape(boxes)[-2]
    boxes = array_ops.reshape(boxes, [-1, num_boxes, 4])
    scores = array_ops.reshape(scores, [-1, num_boxes])
    batch_size = array_ops.shape(boxes)[0]
    if score_threshold != float('-inf'):
        with ops.name_scope('filter_by_score'):
            score_mask = math_ops.cast(scores > score_threshold, scores.dtype)
            scores *= score_mask
            box_mask = array_ops.expand_dims(math_ops.cast(score_mask, boxes.dtype), 2)
            boxes *= box_mask
    if not canonicalized_coordinates:
        with ops.name_scope('canonicalize_coordinates'):
            (y_1, x_1, y_2, x_2) = array_ops.split(value=boxes, num_or_size_splits=4, axis=2)
            y_1_is_min = math_ops.reduce_all(math_ops.less_equal(y_1[0, 0, 0], y_2[0, 0, 0]))
            (y_min, y_max) = tf_cond.cond(y_1_is_min, lambda : (y_1, y_2), lambda : (y_2, y_1))
            x_1_is_min = math_ops.reduce_all(math_ops.less_equal(x_1[0, 0, 0], x_2[0, 0, 0]))
            (x_min, x_max) = tf_cond.cond(x_1_is_min, lambda : (x_1, x_2), lambda : (x_2, x_1))
            boxes = array_ops.concat([y_min, x_min, y_max, x_max], axis=2)
    if not sorted_input:
        (scores, boxes, sorted_indices) = _sort_scores_and_boxes(scores, boxes)
    else:
        sorted_indices = array_ops.zeros_like(scores, dtype=dtypes.int32)
    pad = math_ops.cast(math_ops.ceil(math_ops.cast(math_ops.maximum(num_boxes, max_output_size), dtypes.float32) / math_ops.cast(tile_size, dtypes.float32)), dtypes.int32) * tile_size - num_boxes
    boxes = array_ops.pad(math_ops.cast(boxes, dtypes.float32), [[0, 0], [0, pad], [0, 0]])
    scores = array_ops.pad(math_ops.cast(scores, dtypes.float32), [[0, 0], [0, pad]])
    num_boxes_after_padding = num_boxes + pad
    num_iterations = num_boxes_after_padding // tile_size

    def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
        if False:
            for i in range(10):
                print('nop')
        return math_ops.logical_and(math_ops.reduce_min(output_size) < max_output_size, idx < num_iterations)

    def suppression_loop_body(boxes, iou_threshold, output_size, idx):
        if False:
            for i in range(10):
                print('nop')
        return _suppression_loop_body(boxes, iou_threshold, output_size, idx, tile_size)
    (selected_boxes, _, output_size, _) = while_loop.while_loop(_loop_cond, suppression_loop_body, [boxes, iou_threshold, array_ops.zeros([batch_size], dtypes.int32), constant_op.constant(0)], shape_invariants=[tensor_shape.TensorShape([None, None, 4]), tensor_shape.TensorShape([]), tensor_shape.TensorShape([None]), tensor_shape.TensorShape([])])
    num_valid = math_ops.minimum(output_size, max_output_size)
    idx = num_boxes_after_padding - math_ops.cast(nn_ops.top_k(math_ops.cast(math_ops.reduce_any(selected_boxes > 0, [2]), dtypes.int32) * array_ops.expand_dims(math_ops.range(num_boxes_after_padding, 0, -1), 0), max_output_size)[0], dtypes.int32)
    idx = math_ops.minimum(idx, num_boxes - 1)
    if not sorted_input:
        index_offsets = math_ops.range(batch_size) * num_boxes
        gather_idx = array_ops.reshape(idx + array_ops.expand_dims(index_offsets, 1), [-1])
        idx = array_ops.reshape(array_ops.gather(array_ops.reshape(sorted_indices, [-1]), gather_idx), [batch_size, -1])
    invalid_index = array_ops.zeros([batch_size, max_output_size], dtype=dtypes.int32)
    idx_index = array_ops.expand_dims(math_ops.range(max_output_size), 0)
    num_valid_expanded = array_ops.expand_dims(num_valid, 1)
    idx = array_ops.where(idx_index < num_valid_expanded, idx, invalid_index)
    num_valid = array_ops.reshape(num_valid, batch_dims)
    return (idx, num_valid)

def non_max_suppression_padded_v1(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), pad_to_max_output_size=False, name=None):
    if False:
        print('Hello World!')
    'Greedily selects a subset of bounding boxes in descending order of score.\n\n  Performs algorithmically equivalent operation to tf.image.non_max_suppression,\n  with the addition of an optional parameter which zero-pads the output to\n  be of size `max_output_size`.\n  The output of this operation is a tuple containing the set of integers\n  indexing into the input collection of bounding boxes representing the selected\n  boxes and the number of valid indices in the index set.  The bounding box\n  coordinates corresponding to the selected indices can then be obtained using\n  the `tf.slice` and `tf.gather` operations.  For example:\n    ```python\n    selected_indices_padded, num_valid = tf.image.non_max_suppression_padded(\n        boxes, scores, max_output_size, iou_threshold,\n        score_threshold, pad_to_max_output_size=True)\n    selected_indices = tf.slice(\n        selected_indices_padded, tf.constant([0]), num_valid)\n    selected_boxes = tf.gather(boxes, selected_indices)\n    ```\n\n  Args:\n    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.\n    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single\n      score corresponding to each box (each row of boxes).\n    max_output_size: A scalar integer `Tensor` representing the maximum number\n      of boxes to be selected by non-max suppression.\n    iou_threshold: A float representing the threshold for deciding whether boxes\n      overlap too much with respect to IOU.\n    score_threshold: A float representing the threshold for deciding when to\n      remove boxes based on score.\n    pad_to_max_output_size: bool.  If True, size of `selected_indices` output is\n      padded to `max_output_size`.\n    name: A name for the operation (optional).\n\n  Returns:\n    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the\n      selected indices from the boxes tensor, where `M <= max_output_size`.\n    valid_outputs: A scalar integer `Tensor` denoting how many elements in\n    `selected_indices` are valid.  Valid elements occur first, then padding.\n  '
    with ops.name_scope(name, 'non_max_suppression_padded'):
        iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
        score_threshold = ops.convert_to_tensor(score_threshold, name='score_threshold')
        return gen_image_ops.non_max_suppression_v4(boxes, scores, max_output_size, iou_threshold, score_threshold, pad_to_max_output_size)

@tf_export('image.draw_bounding_boxes', v1=[])
@dispatch.add_dispatch_support
def draw_bounding_boxes_v2(images, boxes, colors, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Draw bounding boxes on a batch of images.\n\n  Outputs a copy of `images` but draws on top of the pixels zero or more\n  bounding boxes specified by the locations in `boxes`. The coordinates of the\n  each bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`.\n  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width\n  and the height of the underlying image.\n\n  For example, if an image is 100 x 200 pixels (height x width) and the bounding\n  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of\n  the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).\n\n  Parts of the bounding box may fall outside the image.\n\n  Args:\n    images: A `Tensor`. Must be one of the following types: `float32`, `half`.\n      4-D with shape `[batch, height, width, depth]`. A batch of images.\n    boxes: A `Tensor` of type `float32`. 3-D with shape `[batch,\n      num_bounding_boxes, 4]` containing bounding boxes.\n    colors: A `Tensor` of type `float32`. 2-D. A list of RGBA colors to cycle\n      through for the boxes.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor`. Has the same type as `images`.\n\n  Usage Example:\n\n  >>> # create an empty image\n  >>> img = tf.zeros([1, 3, 3, 3])\n  >>> # draw a box around the image\n  >>> box = np.array([0, 0, 1, 1])\n  >>> boxes = box.reshape([1, 1, 4])\n  >>> # alternate between red and blue\n  >>> colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])\n  >>> tf.image.draw_bounding_boxes(img, boxes, colors)\n  <tf.Tensor: shape=(1, 3, 3, 3), dtype=float32, numpy=\n  array([[[[1., 0., 0.],\n          [1., 0., 0.],\n          [1., 0., 0.]],\n          [[1., 0., 0.],\n          [0., 0., 0.],\n          [1., 0., 0.]],\n          [[1., 0., 0.],\n          [1., 0., 0.],\n          [1., 0., 0.]]]], dtype=float32)>\n  '
    if colors is None:
        return gen_image_ops.draw_bounding_boxes(images, boxes, name)
    return gen_image_ops.draw_bounding_boxes_v2(images, boxes, colors, name)

@tf_export(v1=['image.draw_bounding_boxes'])
@dispatch.add_dispatch_support
def draw_bounding_boxes(images, boxes, name=None, colors=None):
    if False:
        while True:
            i = 10
    'Draw bounding boxes on a batch of images.\n\n  Outputs a copy of `images` but draws on top of the pixels zero or more\n  bounding boxes specified by the locations in `boxes`. The coordinates of the\n  each bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`.\n  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width\n  and the height of the underlying image.\n\n  For example, if an image is 100 x 200 pixels (height x width) and the bounding\n  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of\n  the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).\n\n  Parts of the bounding box may fall outside the image.\n\n  Args:\n    images: A `Tensor`. Must be one of the following types: `float32`, `half`.\n      4-D with shape `[batch, height, width, depth]`. A batch of images.\n    boxes: A `Tensor` of type `float32`. 3-D with shape `[batch,\n      num_bounding_boxes, 4]` containing bounding boxes.\n    name: A name for the operation (optional).\n    colors: A `Tensor` of type `float32`. 2-D. A list of RGBA colors to cycle\n      through for the boxes.\n\n  Returns:\n    A `Tensor`. Has the same type as `images`.\n\n  Usage Example:\n\n  >>> # create an empty image\n  >>> img = tf.zeros([1, 3, 3, 3])\n  >>> # draw a box around the image\n  >>> box = np.array([0, 0, 1, 1])\n  >>> boxes = box.reshape([1, 1, 4])\n  >>> # alternate between red and blue\n  >>> colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])\n  >>> tf.image.draw_bounding_boxes(img, boxes, colors)\n  <tf.Tensor: shape=(1, 3, 3, 3), dtype=float32, numpy=\n  array([[[[1., 0., 0.],\n          [1., 0., 0.],\n          [1., 0., 0.]],\n          [[1., 0., 0.],\n          [0., 0., 0.],\n          [1., 0., 0.]],\n          [[1., 0., 0.],\n          [1., 0., 0.],\n          [1., 0., 0.]]]], dtype=float32)>\n  '
    return draw_bounding_boxes_v2(images, boxes, colors, name)

@tf_export('image.generate_bounding_box_proposals')
@dispatch.add_dispatch_support
def generate_bounding_box_proposals(scores, bbox_deltas, image_info, anchors, nms_threshold=0.7, pre_nms_topn=6000, min_size=16, post_nms_topn=300, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Generate bounding box proposals from encoded bounding boxes.\n\n  Args:\n    scores: A 4-D float `Tensor` of shape\n     `[num_images, height, width, num_achors]` containing scores of\n      the boxes for given anchors, can be unsorted.\n    bbox_deltas: A 4-D float `Tensor` of shape\n     `[num_images, height, width, 4 x num_anchors]` encoding boxes\n      with respect to each anchor. Coordinates are given\n      in the form `[dy, dx, dh, dw]`.\n    image_info: A 2-D float `Tensor` of shape `[num_images, 5]`\n      containing image information Height, Width, Scale.\n    anchors: A 2-D float `Tensor` of shape `[num_anchors, 4]`\n      describing the anchor boxes.\n      Boxes are formatted in the form `[y1, x1, y2, x2]`.\n    nms_threshold: A scalar float `Tensor` for non-maximal-suppression\n      threshold. Defaults to 0.7.\n    pre_nms_topn: A scalar int `Tensor` for the number of\n      top scoring boxes to be used as input. Defaults to 6000.\n    min_size: A scalar float `Tensor`. Any box that has a smaller size\n      than min_size will be discarded. Defaults to 16.\n    post_nms_topn: An integer. Maximum number of rois in the output.\n    name: A name for this operation (optional).\n\n  Returns:\n    rois: Region of interest boxes sorted by their scores.\n    roi_probabilities: scores of the ROI boxes in the ROIs' `Tensor`.\n  "
    return gen_image_ops.generate_bounding_box_proposals(scores=scores, bbox_deltas=bbox_deltas, image_info=image_info, anchors=anchors, nms_threshold=nms_threshold, pre_nms_topn=pre_nms_topn, min_size=min_size, post_nms_topn=post_nms_topn, name=name)