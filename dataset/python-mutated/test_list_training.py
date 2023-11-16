from tests.base_case import ChatBotTestCase
from chatterbot.trainers import ListTrainer
from chatterbot import preprocessors

class ListTrainingTests(ChatBotTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.trainer = ListTrainer(self.chatbot, show_training_progress=False)

    def test_training_cleans_whitespace(self):
        if False:
            print('Hello World!')
        '\n        Test that the ``clean_whitespace`` preprocessor is used during\n        the training process.\n        '
        self.chatbot.preprocessors = [preprocessors.clean_whitespace]
        self.trainer.train(['Can I help you with anything?', 'No, I     think I am all set.', 'Okay, have a nice day.', 'Thank you, you too.'])
        response = self.chatbot.get_response('Can I help you with anything?')
        self.assertEqual(response.text, 'No, I think I am all set.')

    def test_training_adds_statements(self):
        if False:
            return 10
        '\n        Test that the training method adds statements\n        to the database.\n        '
        conversation = ['Hello', 'Hi there!', 'How are you doing?', "I'm great.", 'That is good to hear', 'Thank you.', 'You are welcome.', 'Sure, any time.', 'Yeah', 'Can I help you with anything?']
        self.trainer.train(conversation)
        response = self.chatbot.get_response('Thank you.')
        self.assertEqual(response.text, 'You are welcome.')

    def test_training_sets_in_response_to(self):
        if False:
            while True:
                i = 10
        conversation = ['Do you like my hat?', 'I do not like your hat.']
        self.trainer.train(conversation)
        statements = list(self.chatbot.storage.filter(in_response_to='Do you like my hat?'))
        self.assertIsLength(statements, 1)
        self.assertEqual(statements[0].in_response_to, 'Do you like my hat?')

    def test_training_sets_search_text(self):
        if False:
            return 10
        conversation = ['Do you like my hat?', 'I do not like your hat.']
        self.trainer.train(conversation)
        statements = list(self.chatbot.storage.filter(in_response_to='Do you like my hat?'))
        self.assertIsLength(statements, 1)
        self.assertEqual(statements[0].search_text, 'VERB:hat')

    def test_training_sets_search_in_response_to(self):
        if False:
            i = 10
            return i + 15
        conversation = ['Do you like my hat?', 'I do not like your hat.']
        self.trainer.train(conversation)
        statements = list(self.chatbot.storage.filter(in_response_to='Do you like my hat?'))
        self.assertIsLength(statements, 1)
        self.assertEqual(statements[0].search_in_response_to, 'VERB:hat')

    def test_database_has_correct_format(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the database maintains a valid format\n        when data is added and updated. This means that\n        after the training process, the database should\n        contain nine objects and eight of these objects\n        should list the previous member of the list as\n        a response.\n        '
        conversation = ['Hello sir!', 'Hi, can I help you?', 'Yes, I am looking for italian parsely.', 'Italian parsely is right over here in out produce department', 'Great, thank you for your help.', 'No problem, did you need help finding anything else?', 'Nope, that was it.', 'Alright, have a great day.', 'Thanks, you too.']
        self.trainer.train(conversation)
        self.assertEqual(self.chatbot.storage.count(), 9)
        first_statement = list(self.chatbot.storage.filter(text=conversation[0]))
        self.assertIsNone(first_statement[0].in_response_to)
        second_statement = list(self.chatbot.storage.filter(text=conversation[1]))
        self.assertEqual(second_statement[0].in_response_to, conversation[0])

    def test_training_with_unicode_characters(self):
        if False:
            while True:
                i = 10
        '\n        Ensure that the training method adds unicode statements\n        to the database.\n        '
        conversation = [u'¶ ∑ ∞ ∫ π ∈ ℝ² ∖ ⩆ ⩇ ⩈ ⩉ ⩊ ⩋ ⪽ ⪾ ⪿ ⫀ ⫁ ⫂ ⋒ ⋓', u'⊂ ⊃ ⊆ ⊇ ⊈ ⊉ ⊊ ⊋ ⊄ ⊅ ⫅ ⫆ ⫋ ⫌ ⫃ ⫄ ⫇ ⫈ ⫉ ⫊ ⟃ ⟄', u'∠ ∡ ⦛ ⦞ ⦟ ⦢ ⦣ ⦤ ⦥ ⦦ ⦧ ⦨ ⦩ ⦪ ⦫ ⦬ ⦭ ⦮ ⦯ ⦓ ⦔ ⦕ ⦖ ⟀', u'∫ ∬ ∭ ∮ ∯ ∰ ∱ ∲ ∳ ⨋ ⨌ ⨍ ⨎ ⨏ ⨐ ⨑ ⨒ ⨓ ⨔ ⨕ ⨖ ⨗ ⨘ ⨙ ⨚ ⨛ ⨜', u'≁ ≂ ≃ ≄ ⋍ ≅ ≆ ≇ ≈ ≉ ≊ ≋ ≌ ⩯ ⩰ ⫏ ⫐ ⫑ ⫒ ⫓ ⫔ ⫕ ⫖', u'¬ ⫬ ⫭ ⊨ ⊭ ∀ ∁ ∃ ∄ ∴ ∵ ⊦ ⊬ ⊧ ⊩ ⊮ ⊫ ⊯ ⊪ ⊰ ⊱ ⫗ ⫘', u'∧ ∨ ⊻ ⊼ ⊽ ⋎ ⋏ ⟑ ⟇ ⩑ ⩒ ⩓ ⩔ ⩕ ⩖ ⩗ ⩘ ⩙ ⩚ ⩛ ⩜ ⩝ ⩞ ⩟ ⩠ ⩢']
        self.trainer.train(conversation)
        response = self.chatbot.get_response(conversation[1])
        self.assertEqual(response.text, conversation[2])

    def test_training_with_emoji_characters(self):
        if False:
            i = 10
            return i + 15
        '\n        Ensure that the training method adds statements containing emojis.\n        '
        conversation = [u'Hi, how are you? 😃', u'I am just dandy 👍', u'Superb! 🎆']
        self.trainer.train(conversation)
        response = self.chatbot.get_response(conversation[1])
        self.assertEqual(response.text, conversation[2])

    def test_training_with_unicode_bytestring(self):
        if False:
            i = 10
            return i + 15
        '\n        Test training with an 8-bit bytestring.\n        '
        conversation = ['Hi, how are you?', 'ä½\xa0å¥½å\x90\x97', 'Superb!']
        self.trainer.train(conversation)
        response = self.chatbot.get_response(conversation[1])
        self.assertEqual(response.text, conversation[2])

    def test_similar_sentence_gets_same_response_multiple_times(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if the bot returns the same response for the same\n        question (which is similar to the one present in the training set)\n        when asked repeatedly.\n        '
        training_data = ['how do you login to gmail?', 'Goto gmail.com, enter your login information and hit enter!']
        similar_question = 'how do I login to gmail?'
        self.trainer.train(training_data)
        response_to_trained_set = self.chatbot.get_response(text='how do you login to gmail?', conversation='a')
        response1 = self.chatbot.get_response(text=similar_question, conversation='b')
        response2 = self.chatbot.get_response(text=similar_question, conversation='c')
        self.assertEqual(response_to_trained_set.text, training_data[1])
        self.assertEqual(response1.text, training_data[1])
        self.assertEqual(response2.text, training_data[1])

    def test_consecutive_trainings_same_responses_different_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test consecutive trainings with the same responses to different inputs.\n        '
        self.trainer.train(['A', 'B', 'C'])
        self.trainer.train(['B', 'C', 'D'])
        response1 = self.chatbot.get_response('B')
        response2 = self.chatbot.get_response('C')
        self.assertEqual(response1.text, 'C')
        self.assertEqual(response2.text, 'D')

class ChatterBotResponseTests(ChatBotTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        '\n        Set up a database for testing.\n        '
        self.trainer = ListTrainer(self.chatbot, show_training_progress=False)
        data1 = ['african or european?', "Huh? I... I don't know that.", 'How do you know so much about swallows?']
        data2 = ['Siri is adorable', 'Who is Seri?', 'Siri is my cat']
        data3 = ['What... is your quest?', 'To seek the Holy Grail.', 'What... is your favourite colour?', 'Blue.']
        self.trainer.train(data1)
        self.trainer.train(data2)
        self.trainer.train(data3)

    def test_answer_to_known_input(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that a matching response is returned\n        when an exact match exists.\n        '
        input_text = 'What... is your favourite colour?'
        response = self.chatbot.get_response(input_text)
        self.assertIn('Blue', response.text)

    def test_answer_close_to_known_input(self):
        if False:
            while True:
                i = 10
        input_text = 'What is your favourite colour?'
        response = self.chatbot.get_response(input_text)
        self.assertIn('Blue', response.text)

    def test_match_has_no_response(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure that the if the last line in a file\n        matches the input text then a index error does\n        not occur.\n        '
        input_text = 'Siri is my cat'
        response = self.chatbot.get_response(input_text)
        self.assertGreater(len(response.text), 0)

    def test_empty_input(self):
        if False:
            print('Hello World!')
        '\n        If empty input is provided, anything may be returned.\n        '
        response = self.chatbot.get_response('')
        self.assertTrue(len(response.text) >= 0)