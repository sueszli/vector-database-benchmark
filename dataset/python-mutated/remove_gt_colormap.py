"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import glob
import os.path
import numpy as np
from PIL import Image
import tensorflow as tf
FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('original_gt_folder', './VOCdevkit/VOC2012/SegmentationClass', 'Original ground truth annotations.')
tf.compat.v1.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')
tf.compat.v1.flags.DEFINE_string('output_dir', './VOCdevkit/VOC2012/SegmentationClassRaw', 'folder to save modified ground truth annotations.')

def _remove_colormap(filename):
    if False:
        print('Hello World!')
    'Removes the color map from the annotation.\n\n  Args:\n    filename: Ground truth annotation filename.\n\n  Returns:\n    Annotation without color map.\n  '
    return np.array(Image.open(filename))

def _save_annotation(annotation, filename):
    if False:
        for i in range(10):
            print('nop')
    'Saves the annotation as png file.\n\n  Args:\n    annotation: Segmentation annotation.\n    filename: Output filename.\n  '
    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
    with tf.io.gfile.GFile(filename, mode='w') as f:
        pil_image.save(f, 'PNG')

def main(unused_argv):
    if False:
        for i in range(10):
            print('nop')
    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)
    annotations = glob.glob(os.path.join(FLAGS.original_gt_folder, '*.' + FLAGS.segmentation_format))
    for annotation in annotations:
        raw_annotation = _remove_colormap(annotation)
        filename = os.path.basename(annotation)[:-4]
        _save_annotation(raw_annotation, os.path.join(FLAGS.output_dir, filename + '.' + FLAGS.segmentation_format))
if __name__ == '__main__':
    tf.compat.v1.app.run()