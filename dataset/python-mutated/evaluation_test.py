"""Tests for parser evaluation."""
import tensorflow as tf
from dragnn.python import evaluation
from syntaxnet import sentence_pb2

class EvaluationTest(tf.test.TestCase):

    def _add_sentence(self, tags, heads, labels, corpus):
        if False:
            print('Hello World!')
        'Adds a sentence to the corpus.'
        sentence = sentence_pb2.Sentence()
        for (tag, head, label) in zip(tags, heads, labels):
            sentence.token.add(word='x', start=0, end=0, tag=tag, head=head, label=label)
        corpus.append(sentence.SerializeToString())

    def setUp(self):
        if False:
            print('Hello World!')
        self._gold_corpus = []
        self._test_corpus = []
        self._add_sentence(['DT'], [-1], ['ROOT'], self._gold_corpus)
        self._add_sentence(['DT'], [-1], ['ROOT'], self._test_corpus)
        self._add_sentence(['DT', 'JJ', 'NN'], [2, 2, -1], ['det', 'amod', 'ROOT'], self._gold_corpus)
        self._add_sentence(['xx', 'JJ', 'NN'], [1, 0, -1], ['det', 'amod', 'xxxx'], self._test_corpus)

    def testCalculateParseMetrics(self):
        if False:
            for i in range(10):
                print('nop')
        (pos, uas, las) = evaluation.calculate_parse_metrics(self._gold_corpus, self._test_corpus)
        self.assertEqual(75, pos)
        self.assertEqual(50, uas)
        self.assertEqual(25, las)

    def testCalculateSegmentationMetrics(self):
        if False:
            while True:
                i = 10
        self._gold_corpus = []
        self._test_corpus = []

        def add_sentence_for_segment_eval(starts, ends, corpus):
            if False:
                for i in range(10):
                    print('nop')
            'Adds a sentence to the corpus.'
            sentence = sentence_pb2.Sentence()
            for (start, end) in zip(starts, ends):
                sentence.token.add(word='x', start=start, end=end)
            corpus.append(sentence.SerializeToString())
        add_sentence_for_segment_eval([0, 5, 8, 10, 15], [3, 6, 8, 13, 22], self._gold_corpus)
        add_sentence_for_segment_eval([0, 8, 10, 15], [6, 8, 13, 22], self._test_corpus)
        add_sentence_for_segment_eval([0, 8, 13], [6, 11, 20], self._gold_corpus)
        add_sentence_for_segment_eval([0, 8, 13, 17, 21], [6, 11, 15, 19, 22], self._test_corpus)
        (prec, rec, f1) = evaluation.calculate_segmentation_metrics(self._gold_corpus, self._test_corpus)
        self.assertEqual(55.56, prec)
        self.assertEqual(62.5, rec)
        self.assertEqual(58.82, f1)
        summaries = evaluation.segmentation_summaries(self._gold_corpus, self._test_corpus)
        self.assertEqual({'precision': 55.56, 'recall': 62.5, 'f1': 58.82, 'eval_metric': 58.82}, summaries)

    def testParserSummaries(self):
        if False:
            print('Hello World!')
        summaries = evaluation.parser_summaries(self._gold_corpus, self._test_corpus)
        self.assertEqual({'POS': 75, 'UAS': 50, 'LAS': 25, 'eval_metric': 25}, summaries)
if __name__ == '__main__':
    tf.test.main()