from server.tests.utils import BaseTestCase

class TestQArgsCompletions(BaseTestCase):

    def test_args(self):
        if False:
            return 10
        self.assert_interaction('q.args.')

    def test_args_bracket(self):
        if False:
            print('Hello World!')
        self.assert_interaction("q.args['']")
        self.assert_interaction('q.args[""]')

    def test_autocomplete_stop(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.get_completions('q.args.args.')), 0)

    def test_autocomplete_stop_bracket(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(self.get_completions('q.args[""][""]')), 0)
        self.assertEqual(len(self.get_completions("q.args['']['']")), 0)

    def test_autocomplete_block_statements(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.get_completions('if q.args.')), 3)
        self.assertEqual(len(self.get_completions('if q.args[""]')), 3)
        self.assertEqual(len(self.get_completions("if q.args['']")), 3)
        self.assertEqual(len(self.get_completions('while q.args.')), 3)

    def test_in_function_call(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.get_completions('print(q.args.)', typing_offset=1)), 3)
        self.assertEqual(len(self.get_completions('print(q.args.')), 3)