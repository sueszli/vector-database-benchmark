from tests.base_case import ChatBotTestCase
from chatterbot.logic import LogicAdapter
from chatterbot.trainers import ListTrainer

class DummyMutatorLogicAdapter(LogicAdapter):
    """
    This is a dummy class designed to modify a
    the resulting statement before it is returned.
    """

    def process(self, statement, additional_response_selection_parameters=None):
        if False:
            return 10
        statement.add_tags('pos_tags:NN')
        self.chatbot.storage.update(statement)
        statement.confidence = 1
        return statement

class DataCachingTests(ChatBotTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.chatbot.logic_adapters = [DummyMutatorLogicAdapter(self.chatbot)]
        self.trainer = ListTrainer(self.chatbot, show_training_progress=False)
        self.trainer.train(['Hello', 'How are you?'])

    def test_additional_attributes_saved(self):
        if False:
            while True:
                i = 10
        '\n        Test that an additional data attribute can be added to the statement\n        and that this attribute is saved.\n        '
        self.chatbot.get_response('Hello', conversation='test')
        results = list(self.chatbot.storage.filter(text='Hello', in_response_to=None, conversation='test'))
        self.assertEqual(len(results), 1)
        self.assertIn('pos_tags:NN', results[0].get_tags())