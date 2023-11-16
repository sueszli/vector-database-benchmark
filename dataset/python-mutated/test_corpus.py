import os
import io
from unittest import TestCase
from chatterbot import corpus

class CorpusLoadingTestCase(TestCase):

    def test_load_corpus_chinese(self):
        if False:
            i = 10
            return i + 15
        data_files = corpus.list_corpus_files('chatterbot.corpus.chinese')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_english(self):
        if False:
            print('Hello World!')
        data_files = corpus.list_corpus_files('chatterbot.corpus.english')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_english_greetings(self):
        if False:
            i = 10
            return i + 15
        data_files = corpus.list_corpus_files('chatterbot.corpus.english.greetings')
        corpus_data = list(corpus.load_corpus(*data_files))
        self.assertEqual(len(corpus_data), 1)
        (conversations, categories, file_path) = corpus_data[0]
        self.assertIn(['Hi', 'Hello'], conversations)
        self.assertEqual(['greetings'], categories)
        self.assertIn('chatterbot_corpus/data/english/greetings.yml', file_path)

    def test_load_corpus_english_categories(self):
        if False:
            return 10
        data_files = corpus.list_corpus_files('chatterbot.corpus.english.greetings')
        corpus_data = list(corpus.load_corpus(*data_files))
        self.assertEqual(len(corpus_data), 1)
        for (_conversation, categories, _file_path) in corpus_data:
            self.assertIn('greetings', categories)

    def test_load_corpus_french(self):
        if False:
            i = 10
            return i + 15
        data_files = corpus.list_corpus_files('chatterbot.corpus.french')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_german(self):
        if False:
            i = 10
            return i + 15
        data_files = corpus.list_corpus_files('chatterbot.corpus.german')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_hindi(self):
        if False:
            print('Hello World!')
        data_files = corpus.list_corpus_files('chatterbot.corpus.hindi')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_indonesian(self):
        if False:
            print('Hello World!')
        data_files = corpus.list_corpus_files('chatterbot.corpus.indonesian')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_italian(self):
        if False:
            for i in range(10):
                print('nop')
        data_files = corpus.list_corpus_files('chatterbot.corpus.italian')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_marathi(self):
        if False:
            print('Hello World!')
        data_files = corpus.list_corpus_files('chatterbot.corpus.marathi')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_portuguese(self):
        if False:
            i = 10
            return i + 15
        data_files = corpus.list_corpus_files('chatterbot.corpus.portuguese')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_russian(self):
        if False:
            for i in range(10):
                print('nop')
        data_files = corpus.list_corpus_files('chatterbot.corpus.russian')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_spanish(self):
        if False:
            return 10
        data_files = corpus.list_corpus_files('chatterbot.corpus.spanish')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

    def test_load_corpus_telugu(self):
        if False:
            i = 10
            return i + 15
        data_files = corpus.list_corpus_files('chatterbot.corpus.telugu')
        corpus_data = corpus.load_corpus(*data_files)
        self.assertTrue(len(list(corpus_data)))

class CorpusUtilsTestCase(TestCase):

    def test_get_file_path(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that a dotted path is properly converted to a file address.\n        '
        path = corpus.get_file_path('chatterbot.corpus.english')
        self.assertIn(os.path.join('chatterbot_corpus', 'data', 'english'), path)

    def test_read_english_corpus(self):
        if False:
            i = 10
            return i + 15
        corpus_path = os.path.join(corpus.DATA_DIRECTORY, 'english', 'conversations.yml')
        data = corpus.read_corpus(corpus_path)
        self.assertIn('conversations', data)

    def test_list_english_corpus_files(self):
        if False:
            return 10
        data_files = corpus.list_corpus_files('chatterbot.corpus.english')
        for data_file in data_files:
            self.assertIn('.yml', data_file)

    def test_load_corpus(self):
        if False:
            print('Hello World!')
        '\n        Test loading the entire corpus of languages.\n        '
        corpus_files = corpus.list_corpus_files('chatterbot.corpus')
        corpus_data = corpus.load_corpus(*corpus_files)
        self.assertTrue(len(list(corpus_data)))

class CorpusFilePathTestCase(TestCase):

    def test_load_corpus_file(self):
        if False:
            print('Hello World!')
        '\n        Test that a file path can be specified for a corpus.\n        '
        file_path = './test_corpus.yml'
        with io.open(file_path, 'w') as test_corpus:
            yml_data = u'\n'.join(['conversations:', '- - Hello', '  - Hi', '- - Hi', '  - Hello'])
            test_corpus.write(yml_data)
        data_files = corpus.list_corpus_files(file_path)
        corpus_data = list(corpus.load_corpus(*data_files))
        if os.path.exists(file_path):
            os.remove(file_path)
        self.assertEqual(len(corpus_data), 1)
        (conversations, _categories, _file_path) = corpus_data[0]
        self.assertEqual(len(conversations[0]), 2)

    def test_load_corpus_file_non_existent(self):
        if False:
            return 10
        '\n        Test that a file path can be specified for a corpus.\n        '
        file_path = './test_corpus.yml'
        self.assertFalse(os.path.exists(file_path))
        with self.assertRaises(IOError):
            list(corpus.load_corpus(file_path))

    def test_load_corpus_english_greetings(self):
        if False:
            return 10
        file_path = os.path.join(corpus.DATA_DIRECTORY, 'english', 'greetings.yml')
        data_files = corpus.list_corpus_files(file_path)
        corpus_data = corpus.load_corpus(*data_files)
        self.assertEqual(len(list(corpus_data)), 1)

    def test_load_corpus_english(self):
        if False:
            i = 10
            return i + 15
        file_path = os.path.join(corpus.DATA_DIRECTORY, 'english')
        data_files = corpus.list_corpus_files(file_path)
        corpus_data = corpus.load_corpus(*data_files)
        self.assertGreater(len(list(corpus_data)), 1)

    def test_load_corpus_english_trailing_slash(self):
        if False:
            print('Hello World!')
        file_path = os.path.join(corpus.DATA_DIRECTORY, 'english') + '/'
        data_files = corpus.list_corpus_files(file_path)
        corpus_data = list(corpus.load_corpus(*data_files))
        self.assertGreater(len(list(corpus_data)), 1)