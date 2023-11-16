from tests.base_case import ChatBotTestCase
from chatterbot.logic import SpecificResponseAdapter
from chatterbot.conversation import Statement

class SpecificResponseAdapterTestCase(ChatBotTestCase):
    """
    Test cases for the SpecificResponseAdapter
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.adapter = SpecificResponseAdapter(self.chatbot, input_text='Open sesame!', output_text='Your sesame seed hamburger roll is now open.')

    def test_exact_match(self):
        if False:
            print('Hello World!')
        '\n        Test the case that an exact match is given.\n        '
        statement = Statement(text='Open sesame!')
        match = self.adapter.process(statement)
        self.assertEqual(match.confidence, 1)
        self.assertEqual(match, self.adapter.response_statement)

    def test_not_exact_match(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the case that an exact match is not given.\n        '
        statement = Statement(text='Open says me!')
        match = self.adapter.process(statement)
        self.assertEqual(match.confidence, 0)
        self.assertEqual(match, self.adapter.response_statement)