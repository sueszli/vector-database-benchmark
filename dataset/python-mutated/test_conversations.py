from unittest import TestCase
from chatterbot.conversation import Statement

class StatementTests(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.statement = Statement(text='A test statement.')

    def test_serializer(self):
        if False:
            while True:
                i = 10
        data = self.statement.serialize()
        self.assertEqual(self.statement.text, data['text'])