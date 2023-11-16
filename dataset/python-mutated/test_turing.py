from unittest import TestCase, expectedFailure

class TuringTests(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        from chatterbot import ChatBot
        self.chatbot = ChatBot('Agent Jr.')

    @expectedFailure
    def test_ask_name(self):
        if False:
            print('Hello World!')
        response = self.chatbot.get_response('What is your name?')
        self.assertIn('Agent', response.text)

    @expectedFailure
    def test_repeat_information(self):
        if False:
            while True:
                i = 10
        '\n        Test if we can detect any repeat responses from the agent.\n        '
        self.fail('Condition not met.')

    @expectedFailure
    def test_repeat_input(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test what the responses are like if we keep giving the same input.\n        '
        self.fail('Condition not met.')

    @expectedFailure
    def test_contradicting_responses(self):
        if False:
            print('Hello World!')
        '\n        Test if we can get the agent to contradict themselves.\n        '
        self.fail('Condition not met.')

    @expectedFailure
    def test_mathematical_ability(self):
        if False:
            while True:
                i = 10
        '\n        The math questions inherently suggest that the agent\n        should get some math problems wrong in order to seem\n        more human. My view on this is that it is more useful\n        to have a bot that is good at math, which could just\n        as easily be a human.\n        '
        self.fail('Condition not met.')

    @expectedFailure
    def test_response_time(self):
        if False:
            print('Hello World!')
        '\n        Does the agent respond in a realistic amount of time?\n        '
        self.fail('Condition not met.')