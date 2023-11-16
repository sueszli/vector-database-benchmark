"""Tests for render_spec_with_graphviz."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.platform import googletest
from dragnn.protos import spec_pb2
from dragnn.python import render_spec_with_graphviz
from dragnn.python import spec_builder

def _make_basic_master_spec():
    if False:
        i = 10
        return i + 15
    'Constructs a simple spec.\n\n  Modified version of dragnn/tools/parser_trainer.py\n\n  Returns:\n    spec_pb2.MasterSpec instance.\n  '
    lookahead = spec_builder.ComponentSpecBuilder('lookahead')
    lookahead.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes='256')
    lookahead.set_transition_system(name='shift-only', left_to_right='true')
    lookahead.add_fixed_feature(name='words', fml='input.word', embedding_dim=64)
    lookahead.add_rnn_link(embedding_dim=-1)
    parser = spec_builder.ComponentSpecBuilder('parser')
    parser.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes='256')
    parser.set_transition_system(name='arc-standard')
    parser.add_token_link(source=lookahead, fml='input.focus', embedding_dim=32)
    master_spec = spec_pb2.MasterSpec()
    master_spec.component.extend([lookahead.spec, parser.spec])
    return master_spec

class RenderSpecWithGraphvizTest(googletest.TestCase):

    def test_constructs_simple_graph(self):
        if False:
            i = 10
            return i + 15
        master_spec = _make_basic_master_spec()
        contents = render_spec_with_graphviz.master_spec_graph(master_spec)
        self.assertIn('lookahead', contents)
        self.assertIn('<polygon', contents)
        self.assertIn('roboto, helvetica, arial', contents)
        self.assertIn('FeedForwardNetwork', contents)
        self.assertTrue('arc-standard' in contents or 'arc&#45;standard' in contents)
        self.assertIn('input.focus', contents)
        self.assertTrue('input.word' not in contents, "We don't yet show fixed features")
if __name__ == '__main__':
    googletest.main()