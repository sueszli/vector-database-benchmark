"""Runs a trained audio graph against WAVE files and reports the results.

The model, labels and .wav files specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav_dir.py \\
--graph=/tmp/my_frozen_graph.pb \\
--labels=/tmp/speech_commands_train/conv_labels.txt \\
--wav_dir=/tmp/speech_dataset/left

"""
import argparse
import glob
import sys
import tensorflow as tf
FLAGS = None

def load_graph(filename):
    if False:
        for i in range(10):
            print('nop')
    'Unpersists graph from file as default graph.'
    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def load_labels(filename):
    if False:
        i = 10
        return i + 15
    'Read in labels, one label per line.'
    return [line.rstrip() for line in tf.io.gfile.GFile(filename)]

def run_graph(wav_dir, labels, input_layer_name, output_layer_name, num_top_predictions):
    if False:
        for i in range(10):
            print('nop')
    'Runs the audio data through the graph and prints predictions.'
    with tf.compat.v1.Session() as sess:
        for wav_path in glob.glob(wav_dir + '/*.wav'):
            if not wav_path or not tf.io.gfile.exists(wav_path):
                raise ValueError('Audio file does not exist at {0}'.format(wav_path))
            with open(wav_path, 'rb') as wav_file:
                wav_data = wav_file.read()
            softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
            (predictions,) = sess.run(softmax_tensor, {input_layer_name: wav_data})
            print('\n%s' % wav_path.split('/')[-1])
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
        return 0

def label_wav(wav_dir, labels, graph, input_name, output_name, how_many_labels):
    if False:
        return 10
    'Loads the model and labels, and runs the inference to print predictions.'
    if not labels or not tf.io.gfile.exists(labels):
        raise ValueError('Labels file does not exist at {0}'.format(labels))
    if not graph or not tf.io.gfile.exists(graph):
        raise ValueError('Graph file does not exist at {0}'.format(graph))
    labels_list = load_labels(labels)
    load_graph(graph)
    run_graph(wav_dir, labels_list, input_name, output_name, how_many_labels)

def main(_):
    if False:
        print('Hello World!')
    'Entry point for script, converts flags to arguments.'
    label_wav(FLAGS.wav_dir, FLAGS.labels, FLAGS.graph, FLAGS.input_name, FLAGS.output_name, FLAGS.how_many_labels)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, default='', help='Audio file to be identified.')
    parser.add_argument('--graph', type=str, default='', help='Model to use for identification.')
    parser.add_argument('--labels', type=str, default='', help='Path to file containing labels.')
    parser.add_argument('--input_name', type=str, default='wav_data:0', help='Name of WAVE data input node in model.')
    parser.add_argument('--output_name', type=str, default='labels_softmax:0', help='Name of node outputting a prediction in the model.')
    parser.add_argument('--how_many_labels', type=int, default=3, help='Number of results to show.')
    (FLAGS, unparsed) = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)