import os
import tensorflow as tf
from dragnn.python import dragnn_ops
from dragnn.python import sentence_io
from syntaxnet import sentence_pb2
from syntaxnet import test_flags

class ConllSentenceReaderTest(tf.test.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.filepath = os.path.join(test_flags.source_root(), 'syntaxnet/testdata/mini-training-set')
        self.batch_size = 20

    def assertParseable(self, reader, expected_num, expected_last):
        if False:
            print('Hello World!')
        (sentences, last) = reader.read()
        self.assertEqual(expected_num, len(sentences))
        self.assertEqual(expected_last, last)
        for s in sentences:
            pb = sentence_pb2.Sentence()
            pb.ParseFromString(s)
            self.assertGreater(len(pb.token), 0)

    def testReadFirstSentence(self):
        if False:
            print('Hello World!')
        reader = sentence_io.ConllSentenceReader(self.filepath, 1)
        (sentences, last) = reader.read()
        self.assertEqual(1, len(sentences))
        pb = sentence_pb2.Sentence()
        pb.ParseFromString(sentences[0])
        self.assertFalse(last)
        self.assertEqual(u'I knew I could do it properly if given the right kind of support .', pb.text)

    def testReadFromTextFile(self):
        if False:
            while True:
                i = 10
        reader = sentence_io.ConllSentenceReader(self.filepath, self.batch_size)
        self.assertParseable(reader, self.batch_size, False)
        self.assertParseable(reader, self.batch_size, False)
        self.assertParseable(reader, 14, True)
        self.assertParseable(reader, 0, True)
        self.assertParseable(reader, 0, True)

    def testReadAndProjectivize(self):
        if False:
            for i in range(10):
                print('nop')
        reader = sentence_io.ConllSentenceReader(self.filepath, self.batch_size, projectivize=True)
        self.assertParseable(reader, self.batch_size, False)
        self.assertParseable(reader, self.batch_size, False)
        self.assertParseable(reader, 14, True)
        self.assertParseable(reader, 0, True)
        self.assertParseable(reader, 0, True)
if __name__ == '__main__':
    tf.test.main()