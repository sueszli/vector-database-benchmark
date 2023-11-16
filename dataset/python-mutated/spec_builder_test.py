"""Tests for the DRAGNN spec builder."""
import os.path
import tempfile
import tensorflow as tf
from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import spec_builder
from syntaxnet import parser_trainer

class SpecBuilderTest(tf.test.TestCase):

    def assertSpecEqual(self, expected_spec_text, spec):
        if False:
            while True:
                i = 10
        expected_spec = spec_pb2.ComponentSpec()
        text_format.Parse(expected_spec_text, expected_spec)
        self.assertProtoEquals(expected_spec, spec)

    def testComponentSpecBuilderEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        builder = spec_builder.ComponentSpecBuilder('test')
        self.assertSpecEqual('\nname: "test"\nbackend { registered_name: "SyntaxNetComponent" }\ncomponent_builder { registered_name: "DynamicComponentBuilder" }\n        ', builder.spec)

    def testComponentSpecBuilderFixedFeature(self):
        if False:
            while True:
                i = 10
        builder = spec_builder.ComponentSpecBuilder('test')
        builder.set_network_unit('FeedForwardNetwork', hidden_layer_sizes='64,64')
        builder.set_transition_system('shift-only')
        builder.add_fixed_feature(name='words', fml='input.word', embedding_dim=16)
        self.assertSpecEqual('\nname: "test"\nfixed_feature { name: "words" fml: "input.word" embedding_dim: 16 }\nbackend { registered_name: "SyntaxNetComponent" }\ncomponent_builder { registered_name: "DynamicComponentBuilder" }\nnetwork_unit { registered_name: "FeedForwardNetwork"\n               parameters { key: "hidden_layer_sizes" value: "64,64" } }\ntransition_system { registered_name: "shift-only" }\n        ', builder.spec)

    def testComponentSpecBuilderLinkedFeature(self):
        if False:
            while True:
                i = 10
        builder1 = spec_builder.ComponentSpecBuilder('test1')
        builder1.set_transition_system('shift-only')
        builder1.add_fixed_feature(name='words', fml='input.word', embedding_dim=16)
        builder2 = spec_builder.ComponentSpecBuilder('test2')
        builder2.set_network_unit('IdentityNetwork')
        builder2.set_transition_system('tagger')
        builder2.add_token_link(source=builder1, source_layer='words', fml='input.focus', embedding_dim=-1)
        self.assertSpecEqual('\nname: "test2"\nlinked_feature { name: "test1" source_component: "test1" source_layer: "words"\n                 source_translator: "identity" fml: "input.focus"\n                 embedding_dim: -1 }\nbackend { registered_name: "SyntaxNetComponent" }\ncomponent_builder { registered_name: "DynamicComponentBuilder" }\nnetwork_unit { registered_name: "IdentityNetwork" }\ntransition_system { registered_name: "tagger" }\n        ', builder2.spec)

    def testFillsTaggerTransitions(self):
        if False:
            return 10
        lexicon_dir = tempfile.mkdtemp()

        def write_lines(filename, lines):
            if False:
                print('Hello World!')
            with open(os.path.join(lexicon_dir, filename), 'w') as f:
                f.write(''.join(('{}\n'.format(line) for line in lines)))
        write_lines('label-map', ['0'])
        write_lines('word-map', ['2', 'miranda 1', 'rights 1'])
        write_lines('tag-map', ['2', 'NN 1', 'NNP 1'])
        write_lines('tag-to-category', ['NN\tNOUN', 'NNP\tNOUN'])
        tagger = spec_builder.ComponentSpecBuilder('tagger')
        tagger.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes='256')
        tagger.set_transition_system(name='tagger')
        tagger.add_fixed_feature(name='words', fml='input.word', embedding_dim=64)
        tagger.add_rnn_link(embedding_dim=-1)
        tagger.fill_from_resources(lexicon_dir)
        (fixed_feature,) = tagger.spec.fixed_feature
        (linked_feature,) = tagger.spec.linked_feature
        self.assertEqual(fixed_feature.vocabulary_size, 5)
        self.assertEqual(fixed_feature.size, 1)
        self.assertEqual(fixed_feature.size, 1)
        self.assertEqual(linked_feature.size, 1)
        self.assertEqual(tagger.spec.num_actions, 2)
if __name__ == '__main__':
    tf.test.main()