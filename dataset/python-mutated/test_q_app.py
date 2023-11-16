from server.tests.utils import BaseTestCase

class TestQAppCompletions(BaseTestCase):

    def test_app(self):
        if False:
            i = 10
            return i + 15
        self.assert_state('q.app.')

    def test_app_bracket(self):
        if False:
            return 10
        self.assert_state("q.app['']")
        self.assert_state('q.app[""]')

    def test_autocomplete_stop(self):
        if False:
            return 10
        self.assertEqual(len(self.get_completions('q.app.app.')), 0)

    def test_autocomplete_stop_bracket(self):
        if False:
            return 10
        self.assertEqual(len(self.get_completions('q.app[""][""]')), 0)
        self.assertEqual(len(self.get_completions("q.app['']['']")), 0)

    def test_autocomplete_if_statement(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.get_completions('if q.app.')), 4)
        self.assertEqual(len(self.get_completions('if q.app[""]')), 4)
        self.assertEqual(len(self.get_completions("if q.app['']")), 4)

    def test_in_function_call(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.get_completions('print(q.app.)', typing_offset=1)), 4)
        self.assertEqual(len(self.get_completions('print(q.app.')), 4)