from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import argparse
import numpy as np
import tensorflow as tf
import os

def load_graph(model_file):
    if False:
        print('Hello World!')
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def read_tensor_from_image_file(frame, input_height=299, input_width=299, input_mean=0, input_std=255):
    if False:
        for i in range(10):
            print('nop')
    output_name = 'normalized'
    image_reader = tf.convert_to_tensor(frame)
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result

def load_labels(label_file):
    if False:
        print('Hello World!')
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label
if __name__ == '__main__':
    model_file = '/tmpz/output_graph.pb'
    label_file = '/tmpz/output_label.txt'
    input_height = 224
    input_width = 224
    input_mean = 0
    input_std = 255
    input_layer = 'Placeholder'
    output_layer = 'final_result'
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='image to be processed')
    parser.add_argument('--graph', help='graph/model to be executed')
    parser.add_argument('--labels', help='name of file containing labels')
    parser.add_argument('--input_height', type=int, help='input height')
    parser.add_argument('--input_width', type=int, help='input width')
    parser.add_argument('--input_mean', type=int, help='input mean')
    parser.add_argument('--input_std', type=int, help='input std')
    parser.add_argument('--input_layer', help='name of input layer')
    parser.add_argument('--output_layer', help='name of output layer')
    args = parser.parse_args()
    if args.graph:
        model_file = args.graph
    if args.image:
        files = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer
    graph = load_graph(model_file)
    input_name = 'import/' + input_layer
    output_name = 'import/' + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        cap = cv2.VideoCapture('test_vid1.mp4')
        while cap.isOpened():
            (ret, image) = cap.read()
            image = cv2.flip(image, 1)
            resize = cv2.resize(image, (224, 224))
            cv2.imshow('frame', resize)
            t = read_tensor_from_image_file(resize, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)
            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
            results = np.squeeze(results)
            print(results)
            top_k = results.argsort()[-2:][::-1]
            print(top_k)
            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()