"""Extracts bounding boxes from a list of images, saving them to files.

The images must be in JPG format. The program checks if boxes already
exist, and skips computation for those.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from delf import box_io
from delf import detector
cmd_args = None
_BOX_EXT = '.boxes'
_VIZ_SUFFIX = '_viz.jpg'
_BOX_EDGE_COLORS = ['r', 'y', 'b', 'm', 'k', 'g', 'c', 'w']
_STATUS_CHECK_ITERATIONS = 100

def _ReadImageList(list_path):
    if False:
        i = 10
        return i + 15
    'Helper function to read image paths.\n\n  Args:\n    list_path: Path to list of images, one image path per line.\n\n  Returns:\n    image_paths: List of image paths.\n  '
    with tf.gfile.GFile(list_path, 'r') as f:
        image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths

def _FilterBoxesByScore(boxes, scores, class_indices, score_threshold):
    if False:
        return 10
    'Filter boxes based on detection scores.\n\n  Boxes with detection score >= score_threshold are returned.\n\n  Args:\n    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,\n      left, bottom, right].\n    scores: [N] float array with detection scores.\n    class_indices: [N] int array with class indices.\n    score_threshold: Float detection score threshold to use.\n\n  Returns:\n    selected_boxes: selected `boxes`.\n    selected_scores: selected `scores`.\n    selected_class_indices: selected `class_indices`.\n  '
    selected_boxes = []
    selected_scores = []
    selected_class_indices = []
    for (i, box) in enumerate(boxes):
        if scores[i] >= score_threshold:
            selected_boxes.append(box)
            selected_scores.append(scores[i])
            selected_class_indices.append(class_indices[i])
    return (np.array(selected_boxes), np.array(selected_scores), np.array(selected_class_indices))

def _PlotBoxesAndSaveImage(image, boxes, output_path):
    if False:
        print('Hello World!')
    'Plot boxes on image and save to output path.\n\n  Args:\n    image: Numpy array containing image.\n    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,\n      left, bottom, right].\n    output_path: String containing output path.\n  '
    height = image.shape[0]
    width = image.shape[1]
    (fig, ax) = plt.subplots(1)
    ax.imshow(image)
    for (i, box) in enumerate(boxes):
        scaled_box = [box[0] * height, box[1] * width, box[2] * height, box[3] * width]
        rect = patches.Rectangle([scaled_box[1], scaled_box[0]], scaled_box[3] - scaled_box[1], scaled_box[2] - scaled_box[0], linewidth=3, edgecolor=_BOX_EDGE_COLORS[i % len(_BOX_EDGE_COLORS)], facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def main(argv):
    if False:
        i = 10
        return i + 15
    if len(argv) > 1:
        raise RuntimeError('Too many command-line arguments.')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Reading list of images...')
    image_paths = _ReadImageList(cmd_args.list_images_path)
    num_images = len(image_paths)
    tf.logging.info('done! Found %d images', num_images)
    if not tf.gfile.Exists(cmd_args.output_dir):
        tf.gfile.MakeDirs(cmd_args.output_dir)
    if cmd_args.output_viz_dir and (not tf.gfile.Exists(cmd_args.output_viz_dir)):
        tf.gfile.MakeDirs(cmd_args.output_viz_dir)
    with tf.Graph().as_default():
        filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
        reader = tf.WholeFileReader()
        (_, value) = reader.read(filename_queue)
        image_tf = tf.image.decode_jpeg(value, channels=3)
        image_tf = tf.expand_dims(image_tf, 0)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            detector_fn = detector.MakeDetector(sess, cmd_args.detector_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            start = time.clock()
            for (i, image_path) in enumerate(image_paths):
                if i == 0:
                    tf.logging.info('Starting to detect objects in images...')
                elif i % _STATUS_CHECK_ITERATIONS == 0:
                    elapsed = time.clock() - start
                    tf.logging.info('Processing image %d out of %d, last %d images took %f seconds', i, num_images, _STATUS_CHECK_ITERATIONS, elapsed)
                    start = time.clock()
                im = sess.run(image_tf)
                (base_boxes_filename, _) = os.path.splitext(os.path.basename(image_path))
                out_boxes_filename = base_boxes_filename + _BOX_EXT
                out_boxes_fullpath = os.path.join(cmd_args.output_dir, out_boxes_filename)
                if tf.gfile.Exists(out_boxes_fullpath):
                    tf.logging.info('Skipping %s', image_path)
                    continue
                (boxes_out, scores_out, class_indices_out) = detector_fn(im)
                (selected_boxes, selected_scores, selected_class_indices) = _FilterBoxesByScore(boxes_out[0], scores_out[0], class_indices_out[0], cmd_args.detector_thresh)
                box_io.WriteToFile(out_boxes_fullpath, selected_boxes, selected_scores, selected_class_indices)
                if cmd_args.output_viz_dir:
                    out_viz_filename = base_boxes_filename + _VIZ_SUFFIX
                    out_viz_fullpath = os.path.join(cmd_args.output_viz_dir, out_viz_filename)
                    _PlotBoxesAndSaveImage(im[0], selected_boxes, out_viz_fullpath)
            coord.request_stop()
            coord.join(threads)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--detector_path', type=str, default='/tmp/d2r_frcnn_20190411/', help='\n      Path to exported detector model.\n      ')
    parser.add_argument('--detector_thresh', type=float, default=0.0, help='\n      Detector threshold. Any box with confidence score lower than this is not\n      returned.\n      ')
    parser.add_argument('--list_images_path', type=str, default='list_images.txt', help='\n      Path to list of images to undergo object detection.\n      ')
    parser.add_argument('--output_dir', type=str, default='test_boxes', help="\n      Directory where bounding boxes will be written to. Each image's boxes\n      will be written to a file with same name, and extension replaced by\n      .boxes.\n      ")
    parser.add_argument('--output_viz_dir', type=str, default='', help='\n      Optional. If set, a visualization of the detected boxes overlaid on the\n      image is produced, and saved to this directory. Each image is saved with\n      _viz.jpg suffix.\n      ')
    (cmd_args, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)