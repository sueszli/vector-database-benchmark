from server.tests.utils import BaseTestCase

class TestQClientCompletions(BaseTestCase):

    def test_incomplete_expr(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.get_completions('q.')), 0)

    def test_client(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_state('q.client.')
        print()

    def test_client_bracket(self):
        if False:
            return 10
        self.assert_state("q.client['']")
        self.assert_state('q.client[""]')

    def test_autocomplete_stop(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.get_completions('q.client.client.')), 0)

    def test_autocomplete_stop_bracket(self):
        if False:
            return 10
        self.assertEqual(len(self.get_completions('q.client[""][""]')), 0)
        self.assertEqual(len(self.get_completions("q.client['']['']")), 0)

    def test_autocomplete_if_statement(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.get_completions('if q.client.')), 4)
        self.assertEqual(len(self.get_completions('if q.client[""]')), 4)
        self.assertEqual(len(self.get_completions("if q.client['']")), 4)

    def test_in_function_call(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(self.get_completions('print(q.client.)', typing_offset=1)), 4)
        self.assertEqual(len(self.get_completions('print(q.client.')), 4)