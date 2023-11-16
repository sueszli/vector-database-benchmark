"""Generate captions for images using default beam search parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import tensorflow as tf
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint_path', '', 'Model checkpoint file or directory containing a model checkpoint file.')
tf.flags.DEFINE_string('vocab_file', '', 'Text file containing the vocabulary.')
tf.flags.DEFINE_string('input_files', '', 'File pattern or comma-separated list of file patterns of image files.')
tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    if False:
        print('Hello World!')
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(), FLAGS.checkpoint_path)
    g.finalize()
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    filenames = []
    for file_pattern in FLAGS.input_files.split(','):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info('Running caption generation on %d files matching %s', len(filenames), FLAGS.input_files)
    with tf.Session(graph=g) as sess:
        restore_fn(sess)
        generator = caption_generator.CaptionGenerator(model, vocab)
        for filename in filenames:
            with tf.gfile.GFile(filename, 'rb') as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            print('Captions for image %s:' % os.path.basename(filename))
            for (i, caption) in enumerate(captions):
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = ' '.join(sentence)
                print('  %d) %s (p=%f)' % (i, sentence, math.exp(caption.logprob)))
if __name__ == '__main__':
    tf.app.run()