from tests.base_case import ChatBotTestCase
from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement

class LogicAdapterTestCase(ChatBotTestCase):
    """
    This test case is for the LogicAdapter base class.
    Although this class is not intended for direct use,
    this test case ensures that exceptions requiring
    basic functionality are triggered when needed.
    """

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.adapter = LogicAdapter(self.chatbot)

    def test_class_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the logic adapter can return its own class name.\n        '
        self.assertEqual(self.adapter.class_name, 'LogicAdapter')

    def test_can_process(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method should return true by default.\n        '
        self.assertTrue(self.adapter.can_process(''))

    def test_process(self):
        if False:
            print('Hello World!')
        with self.assertRaises(LogicAdapter.AdapterMethodNotImplementedError):
            self.adapter.process('')

    def test_get_default_response(self):
        if False:
            i = 10
            return i + 15
        response = self.adapter.get_default_response(Statement(text='...'))
        self.assertEqual(response.text, '...')

    def test_get_default_response_from_options(self):
        if False:
            print('Hello World!')
        self.adapter.default_responses = [Statement(text='The default')]
        response = self.adapter.get_default_response(Statement(text='...'))
        self.assertEqual(response.text, 'The default')

    def test_get_default_response_from_database(self):
        if False:
            return 10
        self.chatbot.storage.create(text='The default')
        response = self.adapter.get_default_response(Statement(text='...'))
        self.assertEqual(response.text, 'The default')