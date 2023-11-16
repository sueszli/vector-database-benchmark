from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tempfile
import six
import tensorflow as tf
from official.nlp.bert import tokenization

class TokenizationTest(tf.test.TestCase):
    """Tokenization test.

    The implementation is forked from
    https://github.com/google-research/bert/blob/master/tokenization_test.py."
  """

    def test_full_tokenizer(self):
        if False:
            while True:
                i = 10
        vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing', ',']
        with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
            if six.PY2:
                vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))
            else:
                vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]).encode('utf-8'))
            vocab_file = vocab_writer.name
        tokenizer = tokenization.FullTokenizer(vocab_file)
        os.unlink(vocab_file)
        tokens = tokenizer.tokenize(u'UNwantéd,running')
        self.assertAllEqual(tokens, ['un', '##want', '##ed', ',', 'runn', '##ing'])
        self.assertAllEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_chinese(self):
        if False:
            while True:
                i = 10
        tokenizer = tokenization.BasicTokenizer()
        self.assertAllEqual(tokenizer.tokenize(u'ah博推zz'), [u'ah', u'博', u'推', u'zz'])

    def test_basic_tokenizer_lower(self):
        if False:
            while True:
                i = 10
        tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
        self.assertAllEqual(tokenizer.tokenize(u' \tHeLLo!how  \n Are yoU?  '), ['hello', '!', 'how', 'are', 'you', '?'])
        self.assertAllEqual(tokenizer.tokenize(u'Héllo'), ['hello'])

    def test_basic_tokenizer_no_lower(self):
        if False:
            print('Hello World!')
        tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
        self.assertAllEqual(tokenizer.tokenize(u' \tHeLLo!how  \n Are yoU?  '), ['HeLLo', '!', 'how', 'Are', 'yoU', '?'])

    def test_wordpiece_tokenizer(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing']
        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)
        self.assertAllEqual(tokenizer.tokenize(''), [])
        self.assertAllEqual(tokenizer.tokenize('unwanted running'), ['un', '##want', '##ed', 'runn', '##ing'])
        self.assertAllEqual(tokenizer.tokenize('unwantedX running'), ['[UNK]', 'runn', '##ing'])

    def test_convert_tokens_to_ids(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing']
        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        self.assertAllEqual(tokenization.convert_tokens_to_ids(vocab, ['un', '##want', '##ed', 'runn', '##ing']), [7, 4, 5, 8, 9])

    def test_is_whitespace(self):
        if False:
            print('Hello World!')
        self.assertTrue(tokenization._is_whitespace(u' '))
        self.assertTrue(tokenization._is_whitespace(u'\t'))
        self.assertTrue(tokenization._is_whitespace(u'\r'))
        self.assertTrue(tokenization._is_whitespace(u'\n'))
        self.assertTrue(tokenization._is_whitespace(u'\xa0'))
        self.assertFalse(tokenization._is_whitespace(u'A'))
        self.assertFalse(tokenization._is_whitespace(u'-'))

    def test_is_control(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(tokenization._is_control(u'\x05'))
        self.assertFalse(tokenization._is_control(u'A'))
        self.assertFalse(tokenization._is_control(u' '))
        self.assertFalse(tokenization._is_control(u'\t'))
        self.assertFalse(tokenization._is_control(u'\r'))
        self.assertFalse(tokenization._is_control(u'💩'))

    def test_is_punctuation(self):
        if False:
            while True:
                i = 10
        self.assertTrue(tokenization._is_punctuation(u'-'))
        self.assertTrue(tokenization._is_punctuation(u'$'))
        self.assertTrue(tokenization._is_punctuation(u'`'))
        self.assertTrue(tokenization._is_punctuation(u'.'))
        self.assertFalse(tokenization._is_punctuation(u'A'))
        self.assertFalse(tokenization._is_punctuation(u' '))
if __name__ == '__main__':
    tf.test.main()