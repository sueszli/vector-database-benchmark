from unittest import TestCase, SkipTest
from chatterbot import ChatBot

class ChatBotTestCase(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.chatbot = ChatBot('Test Bot', **self.get_kwargs())

    def tearDown(self):
        if False:
            while True:
                i = 10
        '\n        Remove the test database.\n        '
        self.chatbot.storage.drop()

    def assertIsLength(self, item, length):
        if False:
            return 10
        '\n        Assert that an iterable has the given length.\n        '
        if len(item) != length:
            raise AssertionError('Length {} is not equal to {}'.format(len(item), length))

    def get_kwargs(self):
        if False:
            print('Hello World!')
        return {'database_uri': None, 'initialize': False}

class ChatBotMongoTestCase(ChatBotTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        from pymongo.errors import ServerSelectionTimeoutError
        from pymongo import MongoClient
        try:
            client = MongoClient(serverSelectionTimeoutMS=0.1)
            client.server_info()
        except ServerSelectionTimeoutError:
            raise SkipTest('Unable to connect to Mongo DB.')

    def get_kwargs(self):
        if False:
            return 10
        kwargs = super().get_kwargs()
        kwargs['database_uri'] = 'mongodb://localhost:27017/chatterbot_test_database'
        kwargs['storage_adapter'] = 'chatterbot.storage.MongoDatabaseAdapter'
        return kwargs

class ChatBotSQLTestCase(ChatBotTestCase):

    def get_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        kwargs = super().get_kwargs()
        kwargs['storage_adapter'] = 'chatterbot.storage.SQLStorageAdapter'
        return kwargs