from runner.koan import *

class AboutStrings(Koan):

    def test_double_quoted_strings_are_strings(self):
        if False:
            return 10
        string = 'Hello, world.'
        self.assertEqual(__, isinstance(string, str))

    def test_single_quoted_strings_are_also_strings(self):
        if False:
            print('Hello World!')
        string = 'Goodbye, world.'
        self.assertEqual(__, isinstance(string, str))

    def test_triple_quote_strings_are_also_strings(self):
        if False:
            print('Hello World!')
        string = 'Howdy, world!'
        self.assertEqual(__, isinstance(string, str))

    def test_triple_single_quotes_work_too(self):
        if False:
            i = 10
            return i + 15
        string = 'Bonjour tout le monde!'
        self.assertEqual(__, isinstance(string, str))

    def test_raw_strings_are_also_strings(self):
        if False:
            return 10
        string = 'Konnichi wa, world!'
        self.assertEqual(__, isinstance(string, str))

    def test_use_single_quotes_to_create_string_with_double_quotes(self):
        if False:
            print('Hello World!')
        string = 'He said, "Go Away."'
        self.assertEqual(__, string)

    def test_use_double_quotes_to_create_strings_with_single_quotes(self):
        if False:
            i = 10
            return i + 15
        string = "Don't"
        self.assertEqual(__, string)

    def test_use_backslash_for_escaping_quotes_in_strings(self):
        if False:
            for i in range(10):
                print('nop')
        a = 'He said, "Don\'t"'
        b = 'He said, "Don\'t"'
        self.assertEqual(__, a == b)

    def test_use_backslash_at_the_end_of_a_line_to_continue_onto_the_next_line(self):
        if False:
            print('Hello World!')
        string = 'It was the best of times,\nIt was the worst of times.'
        self.assertEqual(__, len(string))

    def test_triple_quoted_strings_can_span_lines(self):
        if False:
            while True:
                i = 10
        string = '\nHowdy,\nworld!\n'
        self.assertEqual(__, len(string))

    def test_triple_quoted_strings_need_less_escaping(self):
        if False:
            return 10
        a = 'Hello "world".'
        b = 'Hello "world".'
        self.assertEqual(__, a == b)

    def test_escaping_quotes_at_the_end_of_triple_quoted_string(self):
        if False:
            return 10
        string = 'Hello "world"'
        self.assertEqual(__, string)

    def test_plus_concatenates_strings(self):
        if False:
            return 10
        string = 'Hello, ' + 'world'
        self.assertEqual(__, string)

    def test_adjacent_strings_are_concatenated_automatically(self):
        if False:
            return 10
        string = 'Hello, world'
        self.assertEqual(__, string)

    def test_plus_will_not_modify_original_strings(self):
        if False:
            print('Hello World!')
        hi = 'Hello, '
        there = 'world'
        string = hi + there
        self.assertEqual(__, hi)
        self.assertEqual(__, there)

    def test_plus_equals_will_append_to_end_of_string(self):
        if False:
            print('Hello World!')
        hi = 'Hello, '
        there = 'world'
        hi += there
        self.assertEqual(__, hi)

    def test_plus_equals_also_leaves_original_string_unmodified(self):
        if False:
            return 10
        original = 'Hello, '
        hi = original
        there = 'world'
        hi += there
        self.assertEqual(__, original)

    def test_most_strings_interpret_escape_characters(self):
        if False:
            while True:
                i = 10
        string = '\n'
        self.assertEqual('\n', string)
        self.assertEqual('\n', string)
        self.assertEqual(__, len(string))