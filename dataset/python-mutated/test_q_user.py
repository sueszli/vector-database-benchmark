from server.tests.utils import BaseTestCase

class TestQUserCompletions(BaseTestCase):

    def test_user(self):
        if False:
            while True:
                i = 10
        self.assert_state('q.user.')

    def test_user_bracket(self):
        if False:
            i = 10
            return i + 15
        self.assert_state("q.user['']")
        self.assert_state('q.user[""]')

    def test_autocomplete_stop(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.get_completions('q.user.user.')), 0)

    def test_autocomplete_stop_bracket(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.get_completions('q.user[""][""]')), 0)
        self.assertEqual(len(self.get_completions("q.user['']['']")), 0)

    def test_autocomplete_if_statement(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.get_completions('if q.user.')), 4)
        self.assertEqual(len(self.get_completions('if q.user[""]')), 4)
        self.assertEqual(len(self.get_completions("if q.user['']")), 4)

    def test_in_function_call(self):
        if False:
            return 10
        self.assertEqual(len(self.get_completions('print(q.user.)', typing_offset=1)), 4)
        self.assertEqual(len(self.get_completions('print(q.user.')), 4)