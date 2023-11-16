"""
Tutorial of fast affine transformation.
To run this tutorial, install opencv-python using pip.

Comprehensive explanation of this tutorial can be found https://tensorlayer.readthedocs.io/en/stable/modules/prepro.html
"""
import multiprocessing
import time
import numpy as np
import cv2
import tensorflow as tf
import tensorlayer as tl
image = tl.vis.read_image('data/tiger.jpeg')
(h, w, _) = image.shape

def create_transformation_matrix():
    if False:
        while True:
            i = 10
    M_rotate = tl.prepro.affine_rotation_matrix(angle=(-20, 20))
    M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=0.5)
    M_shift = tl.prepro.affine_shift_matrix(wrg=(-0.1, 0.1), hrg=(-0.1, 0.1), h=h, w=w)
    M_shear = tl.prepro.affine_shear_matrix(x_shear=(-0.2, 0.2), y_shear=(-0.2, 0.2))
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.8, 1.2))
    M_combined = M_shift.dot(M_zoom).dot(M_shear).dot(M_flip).dot(M_rotate)
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
    return transform_matrix

def example1():
    if False:
        while True:
            i = 10
    ' Example 1: Applying transformation one-by-one is very SLOW ! '
    st = time.time()
    for _ in range(100):
        xx = tl.prepro.rotation(image, rg=-20, is_random=False)
        xx = tl.prepro.flip_axis(xx, axis=1, is_random=False)
        xx = tl.prepro.shear2(xx, shear=(0.0, -0.2), is_random=False)
        xx = tl.prepro.zoom(xx, zoom_range=1 / 0.8)
        xx = tl.prepro.shift(xx, wrg=-0.1, hrg=0, is_random=False)
    print('apply transforms one-by-one took %fs for each image' % ((time.time() - st) / 100))
    tl.vis.save_image(xx, '_result_slow.png')

def example2():
    if False:
        print('Hello World!')
    ' Example 2: Applying all transforms in one is very FAST ! '
    st = time.time()
    for _ in range(100):
        transform_matrix = create_transformation_matrix()
        result = tl.prepro.affine_transform_cv2(image, transform_matrix, border_mode='replicate')
        tl.vis.save_image(result, '_result_fast_{}.png'.format(_))
    print('apply all transforms once took %fs for each image' % ((time.time() - st) / 100))
    tl.vis.save_image(result, '_result_fast.png')

def example3():
    if False:
        while True:
            i = 10
    ' Example 3: Using TF dataset API to load and process image for training '
    n_data = 100
    imgs_file_list = ['data/tiger.jpeg'] * n_data
    train_targets = [np.ones(1)] * n_data

    def generator():
        if False:
            return 10
        if len(imgs_file_list) != len(train_targets):
            raise RuntimeError('len(imgs_file_list) != len(train_targets)')
        for (_input, _target) in zip(imgs_file_list, train_targets):
            yield (_input, _target)

    def _data_aug_fn(image):
        if False:
            while True:
                i = 10
        transform_matrix = create_transformation_matrix()
        result = tl.prepro.affine_transform_cv2(image, transform_matrix)
        return result

    def _map_fn(image_path, target):
        if False:
            return 10
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.numpy_function(_data_aug_fn, [image], [tf.float32])[0]
        target = tf.reshape(target, ())
        return (image, target)
    n_epoch = 10
    batch_size = 5
    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.int64))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.repeat(n_epoch)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.prefetch(1)
    n_step = 0
    st = time.time()
    for (img, target) in dataset:
        n_step += 1
        pass
    assert n_step == n_epoch * n_data / batch_size
    print('dataset APIs took %fs for each image' % ((time.time() - st) / batch_size / n_step))

def example4():
    if False:
        return 10
    ' Example 4: Transforming coordinates using affine matrix. '
    transform_matrix = create_transformation_matrix()
    result = tl.prepro.affine_transform_cv2(image, transform_matrix)
    coords = [[(50, 100), (100, 100), (100, 50), (200, 200)], [(250, 50), (200, 50), (200, 100)]]
    coords_result = tl.prepro.affine_transform_keypoints(coords, transform_matrix)

    def imwrite(image, coords_list, name):
        if False:
            return 10
        coords_list_ = []
        for coords in coords_list:
            coords = np.array(coords, np.int32)
            coords = coords.reshape((-1, 1, 2))
            coords_list_.append(coords)
        image = cv2.polylines(image, coords_list_, True, (0, 255, 255), 3)
        cv2.imwrite(name, image[..., ::-1])
    imwrite(image, coords, '_with_keypoints_origin.png')
    imwrite(result, coords_result, '_with_keypoints_result.png')
if __name__ == '__main__':
    example1()
    example2()
    example3()
    example4()