"""Construct the spec for the CONLL2017 Parser baseline."""
from absl import flags
import tensorflow as tf
from tensorflow.python.platform import gfile
from dragnn.protos import spec_pb2
from dragnn.python import spec_builder
FLAGS = flags.FLAGS
flags.DEFINE_string('spec_file', 'parser_spec.textproto', 'Filename to save the spec to.')

def main(unused_argv):
    if False:
        while True:
            i = 10
    char2word = spec_builder.ComponentSpecBuilder('char_lstm')
    char2word.set_network_unit(name='wrapped_units.LayerNormBasicLSTMNetwork', hidden_layer_sizes='256')
    char2word.set_transition_system(name='char-shift-only', left_to_right='true')
    char2word.add_fixed_feature(name='chars', fml='char-input.text-char', embedding_dim=16)
    lookahead = spec_builder.ComponentSpecBuilder('lookahead')
    lookahead.set_network_unit(name='wrapped_units.LayerNormBasicLSTMNetwork', hidden_layer_sizes='256')
    lookahead.set_transition_system(name='shift-only', left_to_right='false')
    lookahead.add_link(source=char2word, fml='input.last-char-focus', embedding_dim=64)
    tagger = spec_builder.ComponentSpecBuilder('tagger')
    tagger.set_network_unit(name='wrapped_units.LayerNormBasicLSTMNetwork', hidden_layer_sizes='256')
    tagger.set_transition_system(name='tagger')
    tagger.add_token_link(source=lookahead, fml='input.focus', embedding_dim=64)
    parser = spec_builder.ComponentSpecBuilder('parser')
    parser.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes='256', layer_norm_hidden='true')
    parser.set_transition_system(name='arc-standard')
    parser.add_token_link(source=lookahead, fml='input.focus', embedding_dim=64)
    parser.add_token_link(source=tagger, fml='input.focus stack.focus stack(1).focus', embedding_dim=64)
    parser.add_fixed_feature(name='labels', embedding_dim=16, fml=' '.join(['stack.child(1).label', 'stack.child(1).sibling(-1).label', 'stack.child(-1).label', 'stack.child(-1).sibling(1).label', 'stack(1).child(1).label', 'stack(1).child(1).sibling(-1).label', 'stack(1).child(-1).label', 'stack(1).child(-1).sibling(1).label', 'stack.child(2).label', 'stack.child(-2).label', 'stack(1).child(2).label', 'stack(1).child(-2).label']))
    parser.add_link(source=parser, name='rnn-stack', fml='stack.focus stack(1).focus', source_translator='shift-reduce-step', embedding_dim=64)
    master_spec = spec_pb2.MasterSpec()
    master_spec.component.extend([char2word.spec, lookahead.spec, tagger.spec, parser.spec])
    with gfile.FastGFile(FLAGS.spec_file, 'w') as f:
        f.write(str(master_spec).encode('utf-8'))
if __name__ == '__main__':
    tf.app.run()