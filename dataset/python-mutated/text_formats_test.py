"""Tests for english_tokenizer."""
import os.path
import tensorflow as tf
import syntaxnet.load_parser_ops
from tensorflow.python.platform import tf_logging as logging
from syntaxnet import sentence_pb2
from syntaxnet import task_spec_pb2
from syntaxnet import test_flags
from syntaxnet.ops import gen_parser_ops

class TextFormatsTest(tf.test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.corpus_file = os.path.join(test_flags.temp_dir(), 'documents.conll')
        self.context_file = os.path.join(test_flags.temp_dir(), 'context.pbtxt')

    def AddInput(self, name, file_pattern, record_format, context):
        if False:
            while True:
                i = 10
        inp = context.input.add()
        inp.name = name
        inp.record_format.append(record_format)
        inp.part.add().file_pattern = file_pattern

    def AddParameter(self, name, value, context):
        if False:
            i = 10
            return i + 15
        param = context.parameter.add()
        param.name = name
        param.value = value

    def WriteContext(self, corpus_format):
        if False:
            for i in range(10):
                print('nop')
        context = task_spec_pb2.TaskSpec()
        self.AddInput('documents', self.corpus_file, corpus_format, context)
        for name in ('word-map', 'lcword-map', 'tag-map', 'category-map', 'label-map', 'prefix-table', 'suffix-table', 'tag-to-category'):
            self.AddInput(name, os.path.join(test_flags.temp_dir(), name), '', context)
        logging.info('Writing context to: %s', self.context_file)
        with open(self.context_file, 'w') as f:
            f.write(str(context))

    def ReadNextDocument(self, sess, sentence):
        if False:
            while True:
                i = 10
        (sentence_str,) = sess.run([sentence])
        if sentence_str:
            sentence_doc = sentence_pb2.Sentence()
            sentence_doc.ParseFromString(sentence_str[0])
        else:
            sentence_doc = None
        return sentence_doc

    def CheckTokenization(self, sentence, tokenization):
        if False:
            for i in range(10):
                print('nop')
        self.WriteContext('english-text')
        logging.info('Writing text file to: %s', self.corpus_file)
        with open(self.corpus_file, 'w') as f:
            f.write(sentence)
        (sentence, _) = gen_parser_ops.document_source(task_context=self.context_file, batch_size=1)
        with self.test_session() as sess:
            sentence_doc = self.ReadNextDocument(sess, sentence)
            self.assertEqual(' '.join([t.word for t in sentence_doc.token]), tokenization)

    def CheckUntokenizedDoc(self, sentence, words, starts, ends):
        if False:
            print('Hello World!')
        self.WriteContext('untokenized-text')
        logging.info('Writing text file to: %s', self.corpus_file)
        with open(self.corpus_file, 'w') as f:
            f.write(sentence)
        (sentence, _) = gen_parser_ops.document_source(task_context=self.context_file, batch_size=1)
        with self.test_session() as sess:
            sentence_doc = self.ReadNextDocument(sess, sentence)
            self.assertEqual(len(sentence_doc.token), len(words))
            self.assertEqual(len(sentence_doc.token), len(starts))
            self.assertEqual(len(sentence_doc.token), len(ends))
            for (i, token) in enumerate(sentence_doc.token):
                self.assertEqual(token.word.encode('utf-8'), words[i])
                self.assertEqual(token.start, starts[i])
                self.assertEqual(token.end, ends[i])

    def testUntokenized(self):
        if False:
            i = 10
            return i + 15
        self.CheckUntokenizedDoc('一个测试', ['一', '个', '测', '试'], [0, 3, 6, 9], [2, 5, 8, 11])
        self.CheckUntokenizedDoc('Hello ', ['H', 'e', 'l', 'l', 'o', ' '], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])

    def testConllSentence(self):
        if False:
            for i in range(10):
                print('nop')
        test_sentence = "\n1-2\tWe've\t_\n1\tWe\twe\tPRON\tPRP\tCase=Nom\t3\tnsubj\t_\tSpaceAfter=No\n2\t've\thave\tAUX\tVBP\tMood=Ind\t3\taux\t_\t_\n3\tmoved\tmove\tVERB\tVBN\tTense=Past\t0\troot\t_\t_\n4\ton\ton\tADV\tRB\t_\t3\tadvmod\t_\tSpaceAfter=No|foobar=baz\n4.1\tignored\tignore\tVERB\tVBN\tTense=Past\t0\t_\t_\t_\n5\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_\n"
        with open(self.corpus_file, 'w') as f:
            f.write(test_sentence)
        self.WriteContext('conll-sentence')
        (sentence, _) = gen_parser_ops.document_source(task_context=self.context_file, batch_size=1)
        expected_text = u"We've moved on."
        expected_words = [u'We', u"'ve", u'moved', u'on', u'.']
        expected_starts = [0, 2, 6, 12, 14]
        expected_ends = [1, 4, 10, 13, 14]
        with self.test_session() as sess:
            sentence_doc = self.ReadNextDocument(sess, sentence)
            self.assertEqual(expected_text, sentence_doc.text)
            self.assertEqual(expected_words, [t.word for t in sentence_doc.token])
            self.assertEqual(expected_starts, [t.start for t in sentence_doc.token])
            self.assertEqual(expected_ends, [t.end for t in sentence_doc.token])

    def testSentencePrototext(self):
        if False:
            return 10
        test_sentence = '\ntext: "fair enough; you people have eaten me."\ntoken {\n  word: "fair"\n  start: 0\n  end: 3\n  break_level: NO_BREAK\n}\ntoken {\n  word: "enough"\n  start: 5\n  end: 10\n  head: 0\n  break_level: SPACE_BREAK\n}\n'.lstrip()
        with open(self.corpus_file, 'w') as f:
            f.write(test_sentence)
        self.WriteContext('sentence-prototext')
        (sentence, _) = gen_parser_ops.document_source(task_context=self.context_file, batch_size=1)
        expected_text = u'fair enough; you people have eaten me.'
        expected_words = [u'fair', u'enough']
        expected_starts = [0, 5]
        expected_ends = [3, 10]
        with self.test_session() as sess:
            sentence_doc = self.ReadNextDocument(sess, sentence)
            self.assertEqual(expected_text, sentence_doc.text)
            self.assertEqual(expected_words, [t.word for t in sentence_doc.token])
            self.assertEqual(expected_starts, [t.start for t in sentence_doc.token])
            self.assertEqual(expected_ends, [t.end for t in sentence_doc.token])

    def testSegmentationTrainingData(self):
        if False:
            return 10
        doc1_lines = ['测试\tNO_SPACE\n', '的\tNO_SPACE\n', '句子\tNO_SPACE']
        doc1_text = '测试的句子'
        doc1_tokens = ['测', '试', '的', '句', '子']
        doc1_break_levles = [1, 0, 1, 1, 0]
        doc2_lines = ['That\tNO_SPACE\n', "'s\tSPACE\n", 'a\tSPACE\n', 'good\tSPACE\n', 'point\tNO_SPACE\n', '.\tNO_SPACE']
        doc2_text = "That's a good point."
        doc2_tokens = ['T', 'h', 'a', 't', "'", 's', ' ', 'a', ' ', 'g', 'o', 'o', 'd', ' ', 'p', 'o', 'i', 'n', 't', '.']
        doc2_break_levles = [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        self.CheckSegmentationTrainingData(doc1_lines, doc1_text, doc1_tokens, doc1_break_levles)
        self.CheckSegmentationTrainingData(doc2_lines, doc2_text, doc2_tokens, doc2_break_levles)

    def CheckSegmentationTrainingData(self, doc_lines, doc_text, doc_words, break_levels):
        if False:
            i = 10
            return i + 15
        self.WriteContext('segment-train-data')
        with open(self.corpus_file, 'w') as f:
            f.write(''.join(doc_lines))
        (sentence, _) = gen_parser_ops.document_source(task_context=self.context_file, batch_size=1)
        with self.test_session() as sess:
            sentence_doc = self.ReadNextDocument(sess, sentence)
            self.assertEqual(doc_text.decode('utf-8'), sentence_doc.text)
            self.assertEqual([t.decode('utf-8') for t in doc_words], [t.word for t in sentence_doc.token])
            self.assertEqual(break_levels, [t.break_level for t in sentence_doc.token])

    def testSimple(self):
        if False:
            print('Hello World!')
        self.CheckTokenization('Hello, world!', 'Hello , world !')
        self.CheckTokenization('"Hello"', "`` Hello ''")
        self.CheckTokenization('{"Hello@#$', '-LRB- `` Hello @ # $')
        self.CheckTokenization('"Hello..."', "`` Hello ... ''")
        self.CheckTokenization('()[]{}<>', '-LRB- -RRB- -LRB- -RRB- -LRB- -RRB- < >')
        self.CheckTokenization('Hello--world', 'Hello -- world')
        self.CheckTokenization("Isn't", "Is n't")
        self.CheckTokenization("n't", "n't")
        self.CheckTokenization('Hello Mr. Smith.', 'Hello Mr. Smith .')
        self.CheckTokenization("It's Mr. Smith's.", "It 's Mr. Smith 's .")
        self.CheckTokenization("It's the Smiths'.", "It 's the Smiths ' .")
        self.CheckTokenization('Gotta go', 'Got ta go')
        self.CheckTokenization('50-year-old', '50-year-old')

    def testUrl(self):
        if False:
            i = 10
            return i + 15
        self.CheckTokenization('http://www.google.com/news is down', 'http : //www.google.com/news is down')
if __name__ == '__main__':
    tf.test.main()