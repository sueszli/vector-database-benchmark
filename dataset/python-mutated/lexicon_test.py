"""Tests for SyntaxNet lexicon."""
import os
import os.path
import tensorflow as tf
from google.protobuf import text_format
from dragnn.python import lexicon
from syntaxnet import parser_trainer
from syntaxnet import task_spec_pb2
from syntaxnet import test_flags
_EXPECTED_CONTEXT = '\ninput { name: "word-map" Part { file_pattern: "/tmp/word-map" } }\ninput { name: "tag-map" Part { file_pattern: "/tmp/tag-map" } }\ninput { name: "tag-to-category" Part { file_pattern: "/tmp/tag-to-category" } }\ninput { name: "lcword-map" Part { file_pattern: "/tmp/lcword-map" } }\ninput { name: "category-map" Part { file_pattern: "/tmp/category-map" } }\ninput { name: "char-map" Part { file_pattern: "/tmp/char-map" } }\ninput { name: "char-ngram-map" Part { file_pattern: "/tmp/char-ngram-map" } }\ninput { name: "label-map" Part { file_pattern: "/tmp/label-map" } }\ninput { name: "prefix-table" Part { file_pattern: "/tmp/prefix-table" } }\ninput { name: "suffix-table" Part { file_pattern: "/tmp/suffix-table" } }\ninput { name: "known-word-map" Part { file_pattern: "/tmp/known-word-map" } }\n'

class LexiconTest(tf.test.TestCase):

    def testCreateLexiconContext(self):
        if False:
            while True:
                i = 10
        expected_context = task_spec_pb2.TaskSpec()
        text_format.Parse(_EXPECTED_CONTEXT, expected_context)
        self.assertProtoEquals(lexicon.create_lexicon_context('/tmp'), expected_context)

    def testBuildLexicon(self):
        if False:
            print('Hello World!')
        empty_input_path = os.path.join(test_flags.temp_dir(), 'empty-input')
        lexicon_output_path = os.path.join(test_flags.temp_dir(), 'lexicon-output')
        with open(empty_input_path, 'w'):
            pass
        if not os.path.exists(lexicon_output_path):
            os.mkdir(lexicon_output_path)
        lexicon.build_lexicon(lexicon_output_path, empty_input_path)
if __name__ == '__main__':
    tf.test.main()