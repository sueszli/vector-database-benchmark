"""Utilities for reading and writing sentences in dragnn."""
import tensorflow as tf
from syntaxnet.ops import gen_parser_ops

class FormatSentenceReader(object):
    """A reader for formatted files, with optional projectivizing."""

    def __init__(self, filepath, record_format, batch_size=32, check_well_formed=False, projectivize=False, morph_to_pos=False):
        if False:
            i = 10
            return i + 15
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        task_context_str = "\n          input {\n            name: 'documents'\n            record_format: '%s'\n            Part {\n             file_pattern: '%s'\n            }\n          }" % (record_format, filepath)
        if morph_to_pos:
            task_context_str += '\n          Parameter {\n            name: "join_category_to_pos"\n            value: "true"\n          }\n          Parameter {\n            name: "add_pos_as_attribute"\n            value: "true"\n          }\n          Parameter {\n            name: "serialize_morph_to_pos"\n            value: "true"\n          }\n          '
        with self._graph.as_default():
            (self._source, self._is_last) = gen_parser_ops.document_source(task_context_str=task_context_str, batch_size=batch_size)
            if check_well_formed:
                self._source = gen_parser_ops.well_formed_filter(self._source)
            if projectivize:
                self._source = gen_parser_ops.projectivize_filter(self._source)

    def read(self):
        if False:
            return 10
        'Reads a single batch of sentences.'
        if self._session:
            (sentences, is_last) = self._session.run([self._source, self._is_last])
            if is_last:
                self._session.close()
                self._session = None
        else:
            (sentences, is_last) = ([], True)
        return (sentences, is_last)

    def corpus(self):
        if False:
            while True:
                i = 10
        'Reads the entire corpus, and returns in a list.'
        tf.logging.info('Reading corpus...')
        corpus = []
        while True:
            (sentences, is_last) = self.read()
            corpus.extend(sentences)
            if is_last:
                break
        tf.logging.info('Read %d sentences.' % len(corpus))
        return corpus

class ConllSentenceReader(FormatSentenceReader):
    """A sentence reader that uses an underlying 'conll-sentence' reader."""

    def __init__(self, filepath, batch_size=32, projectivize=False, morph_to_pos=False):
        if False:
            print('Hello World!')
        super(ConllSentenceReader, self).__init__(filepath, 'conll-sentence', check_well_formed=True, batch_size=batch_size, projectivize=projectivize, morph_to_pos=morph_to_pos)