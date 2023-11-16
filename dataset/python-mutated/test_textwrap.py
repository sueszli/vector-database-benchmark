import unittest
from textwrap import TextWrapper, wrap, fill, dedent, indent, shorten

class BaseTestCase(unittest.TestCase):
    """Parent class with utility methods for textwrap tests."""

    def show(self, textin):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(textin, list):
            result = []
            for i in range(len(textin)):
                result.append('  %d: %r' % (i, textin[i]))
            result = '\n'.join(result) if result else '  no lines'
        elif isinstance(textin, str):
            result = '  %s\n' % repr(textin)
        return result

    def check(self, result, expect):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(result, expect, 'expected:\n%s\nbut got:\n%s' % (self.show(expect), self.show(result)))

    def check_wrap(self, text, width, expect, **kwargs):
        if False:
            while True:
                i = 10
        result = wrap(text, width, **kwargs)
        self.check(result, expect)

    def check_split(self, text, expect):
        if False:
            for i in range(10):
                print('nop')
        result = self.wrapper._split(text)
        self.assertEqual(result, expect, '\nexpected %r\nbut got  %r' % (expect, result))

class WrapTestCase(BaseTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.wrapper = TextWrapper(width=45)

    def test_simple(self):
        if False:
            print('Hello World!')
        text = "Hello there, how are you this fine day?  I'm glad to hear it!"
        self.check_wrap(text, 12, ['Hello there,', 'how are you', 'this fine', "day?  I'm", 'glad to hear', 'it!'])
        self.check_wrap(text, 42, ['Hello there, how are you this fine day?', "I'm glad to hear it!"])
        self.check_wrap(text, 80, [text])

    def test_empty_string(self):
        if False:
            return 10
        self.check_wrap('', 6, [])
        self.check_wrap('', 6, [], drop_whitespace=False)

    def test_empty_string_with_initial_indent(self):
        if False:
            return 10
        self.check_wrap('', 6, [], initial_indent='++')
        self.check_wrap('', 6, [], initial_indent='++', drop_whitespace=False)

    def test_whitespace(self):
        if False:
            i = 10
            return i + 15
        text = 'This is a paragraph that already has\nline breaks.  But some of its lines are much longer than the others,\nso it needs to be wrapped.\nSome lines are \ttabbed too.\nWhat a mess!\n'
        expect = ['This is a paragraph that already has line', 'breaks.  But some of its lines are much', 'longer than the others, so it needs to be', 'wrapped.  Some lines are  tabbed too.  What a', 'mess!']
        wrapper = TextWrapper(45, fix_sentence_endings=True)
        result = wrapper.wrap(text)
        self.check(result, expect)
        result = wrapper.fill(text)
        self.check(result, '\n'.join(expect))
        text = '\tTest\tdefault\t\ttabsize.'
        expect = ['        Test    default         tabsize.']
        self.check_wrap(text, 80, expect)
        text = '\tTest\tcustom\t\ttabsize.'
        expect = ['    Test    custom      tabsize.']
        self.check_wrap(text, 80, expect, tabsize=4)

    def test_fix_sentence_endings(self):
        if False:
            print('Hello World!')
        wrapper = TextWrapper(60, fix_sentence_endings=True)
        text = 'A short line. Note the single space.'
        expect = ['A short line.  Note the single space.']
        self.check(wrapper.wrap(text), expect)
        text = 'Well, Doctor? What do you think?'
        expect = ['Well, Doctor?  What do you think?']
        self.check(wrapper.wrap(text), expect)
        text = 'Well, Doctor?\nWhat do you think?'
        self.check(wrapper.wrap(text), expect)
        text = 'I say, chaps! Anyone for "tennis?"\nHmmph!'
        expect = ['I say, chaps!  Anyone for "tennis?"  Hmmph!']
        self.check(wrapper.wrap(text), expect)
        wrapper.width = 20
        expect = ['I say, chaps!', 'Anyone for "tennis?"', 'Hmmph!']
        self.check(wrapper.wrap(text), expect)
        text = 'And she said, "Go to hell!"\nCan you believe that?'
        expect = ['And she said, "Go to', 'hell!"  Can you', 'believe that?']
        self.check(wrapper.wrap(text), expect)
        wrapper.width = 60
        expect = ['And she said, "Go to hell!"  Can you believe that?']
        self.check(wrapper.wrap(text), expect)
        text = 'File stdio.h is nice.'
        expect = ['File stdio.h is nice.']
        self.check(wrapper.wrap(text), expect)

    def test_wrap_short(self):
        if False:
            print('Hello World!')
        text = 'This is a\nshort paragraph.'
        self.check_wrap(text, 20, ['This is a short', 'paragraph.'])
        self.check_wrap(text, 40, ['This is a short paragraph.'])

    def test_wrap_short_1line(self):
        if False:
            i = 10
            return i + 15
        text = 'This is a short line.'
        self.check_wrap(text, 30, ['This is a short line.'])
        self.check_wrap(text, 30, ['(1) This is a short line.'], initial_indent='(1) ')

    def test_hyphenated(self):
        if False:
            while True:
                i = 10
        text = "this-is-a-useful-feature-for-reformatting-posts-from-tim-peters'ly"
        self.check_wrap(text, 40, ['this-is-a-useful-feature-for-', "reformatting-posts-from-tim-peters'ly"])
        self.check_wrap(text, 41, ['this-is-a-useful-feature-for-', "reformatting-posts-from-tim-peters'ly"])
        self.check_wrap(text, 42, ['this-is-a-useful-feature-for-reformatting-', "posts-from-tim-peters'ly"])
        expect = "this-|is-|a-|useful-|feature-|for-|reformatting-|posts-|from-|tim-|peters'ly".split('|')
        self.check_wrap(text, 1, expect, break_long_words=False)
        self.check_split(text, expect)
        self.check_split('e-mail', ['e-mail'])
        self.check_split('Jelly-O', ['Jelly-O'])
        self.check_split('half-a-crown', 'half-|a-|crown'.split('|'))

    def test_hyphenated_numbers(self):
        if False:
            for i in range(10):
                print('nop')
        text = 'Python 1.0.0 was released on 1994-01-26.  Python 1.0.1 was\nreleased on 1994-02-15.'
        self.check_wrap(text, 30, ['Python 1.0.0 was released on', '1994-01-26.  Python 1.0.1 was', 'released on 1994-02-15.'])
        self.check_wrap(text, 40, ['Python 1.0.0 was released on 1994-01-26.', 'Python 1.0.1 was released on 1994-02-15.'])
        self.check_wrap(text, 1, text.split(), break_long_words=False)
        text = 'I do all my shopping at 7-11.'
        self.check_wrap(text, 25, ['I do all my shopping at', '7-11.'])
        self.check_wrap(text, 27, ['I do all my shopping at', '7-11.'])
        self.check_wrap(text, 29, ['I do all my shopping at 7-11.'])
        self.check_wrap(text, 1, text.split(), break_long_words=False)

    def test_em_dash(self):
        if False:
            i = 10
            return i + 15
        text = 'Em-dashes should be written -- thus.'
        self.check_wrap(text, 25, ['Em-dashes should be', 'written -- thus.'])
        self.check_wrap(text, 29, ['Em-dashes should be written', '-- thus.'])
        expect = ['Em-dashes should be written --', 'thus.']
        self.check_wrap(text, 30, expect)
        self.check_wrap(text, 35, expect)
        self.check_wrap(text, 36, ['Em-dashes should be written -- thus.'])
        text = 'You can also do--this or even---this.'
        expect = ['You can also do', '--this or even', '---this.']
        self.check_wrap(text, 15, expect)
        self.check_wrap(text, 16, expect)
        expect = ['You can also do--', 'this or even---', 'this.']
        self.check_wrap(text, 17, expect)
        self.check_wrap(text, 19, expect)
        expect = ['You can also do--this or even', '---this.']
        self.check_wrap(text, 29, expect)
        self.check_wrap(text, 31, expect)
        expect = ['You can also do--this or even---', 'this.']
        self.check_wrap(text, 32, expect)
        self.check_wrap(text, 35, expect)
        text = "Here's an -- em-dash and--here's another---and another!"
        expect = ["Here's", ' ', 'an', ' ', '--', ' ', 'em-', 'dash', ' ', 'and', '--', "here's", ' ', 'another', '---', 'and', ' ', 'another!']
        self.check_split(text, expect)
        text = 'and then--bam!--he was gone'
        expect = ['and', ' ', 'then', '--', 'bam!', '--', 'he', ' ', 'was', ' ', 'gone']
        self.check_split(text, expect)

    def test_unix_options(self):
        if False:
            for i in range(10):
                print('nop')
        text = 'You should use the -n option, or --dry-run in its long form.'
        self.check_wrap(text, 20, ['You should use the', '-n option, or --dry-', 'run in its long', 'form.'])
        self.check_wrap(text, 21, ['You should use the -n', 'option, or --dry-run', 'in its long form.'])
        expect = ['You should use the -n option, or', '--dry-run in its long form.']
        self.check_wrap(text, 32, expect)
        self.check_wrap(text, 34, expect)
        self.check_wrap(text, 35, expect)
        self.check_wrap(text, 38, expect)
        expect = ['You should use the -n option, or --dry-', 'run in its long form.']
        self.check_wrap(text, 39, expect)
        self.check_wrap(text, 41, expect)
        expect = ['You should use the -n option, or --dry-run', 'in its long form.']
        self.check_wrap(text, 42, expect)
        text = 'the -n option, or --dry-run or --dryrun'
        expect = ['the', ' ', '-n', ' ', 'option,', ' ', 'or', ' ', '--dry-', 'run', ' ', 'or', ' ', '--dryrun']
        self.check_split(text, expect)

    def test_funky_hyphens(self):
        if False:
            while True:
                i = 10
        self.check_split('what the--hey!', ['what', ' ', 'the', '--', 'hey!'])
        self.check_split('what the--', ['what', ' ', 'the--'])
        self.check_split('what the--.', ['what', ' ', 'the--.'])
        self.check_split('--text--.', ['--text--.'])
        self.check_split('--option', ['--option'])
        self.check_split('--option-opt', ['--option-', 'opt'])
        self.check_split('foo --option-opt bar', ['foo', ' ', '--option-', 'opt', ' ', 'bar'])

    def test_punct_hyphens(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_split("the 'wibble-wobble' widget", ['the', ' ', "'wibble-", "wobble'", ' ', 'widget'])
        self.check_split('the "wibble-wobble" widget', ['the', ' ', '"wibble-', 'wobble"', ' ', 'widget'])
        self.check_split('the (wibble-wobble) widget', ['the', ' ', '(wibble-', 'wobble)', ' ', 'widget'])
        self.check_split("the ['wibble-wobble'] widget", ['the', ' ', "['wibble-", "wobble']", ' ', 'widget'])
        self.check_split("what-d'you-call-it.", "what-d'you-|call-|it.".split('|'))

    def test_funky_parens(self):
        if False:
            while True:
                i = 10
        self.check_split('foo (--option) bar', ['foo', ' ', '(--option)', ' ', 'bar'])
        self.check_split('foo (bar) baz', ['foo', ' ', '(bar)', ' ', 'baz'])
        self.check_split('blah (ding dong), wubba', ['blah', ' ', '(ding', ' ', 'dong),', ' ', 'wubba'])

    def test_drop_whitespace_false(self):
        if False:
            for i in range(10):
                print('nop')
        text = ' This is a    sentence with     much whitespace.'
        self.check_wrap(text, 10, [' This is a', '    ', 'sentence ', 'with     ', 'much white', 'space.'], drop_whitespace=False)

    def test_drop_whitespace_false_whitespace_only(self):
        if False:
            while True:
                i = 10
        self.check_wrap('   ', 6, ['   '], drop_whitespace=False)

    def test_drop_whitespace_false_whitespace_only_with_indent(self):
        if False:
            print('Hello World!')
        self.check_wrap('   ', 6, ['     '], drop_whitespace=False, initial_indent='  ')

    def test_drop_whitespace_whitespace_only(self):
        if False:
            i = 10
            return i + 15
        self.check_wrap('  ', 6, [])

    def test_drop_whitespace_leading_whitespace(self):
        if False:
            for i in range(10):
                print('nop')
        text = ' This is a sentence with leading whitespace.'
        self.check_wrap(text, 50, [' This is a sentence with leading whitespace.'])
        self.check_wrap(text, 30, [' This is a sentence with', 'leading whitespace.'])

    def test_drop_whitespace_whitespace_line(self):
        if False:
            i = 10
            return i + 15
        text = 'abcd    efgh'
        self.check_wrap(text, 6, ['abcd', '    ', 'efgh'], drop_whitespace=False)
        self.check_wrap(text, 6, ['abcd', 'efgh'])

    def test_drop_whitespace_whitespace_only_with_indent(self):
        if False:
            print('Hello World!')
        self.check_wrap('  ', 6, [], initial_indent='++')

    def test_drop_whitespace_whitespace_indent(self):
        if False:
            return 10
        self.check_wrap('abcd efgh', 6, ['  abcd', '  efgh'], initial_indent='  ', subsequent_indent='  ')

    def test_split(self):
        if False:
            for i in range(10):
                print('nop')
        text = 'Hello there -- you goof-ball, use the -b option!'
        result = self.wrapper._split(text)
        self.check(result, ['Hello', ' ', 'there', ' ', '--', ' ', 'you', ' ', 'goof-', 'ball,', ' ', 'use', ' ', 'the', ' ', '-b', ' ', 'option!'])

    def test_break_on_hyphens(self):
        if False:
            return 10
        text = 'yaba daba-doo'
        self.check_wrap(text, 10, ['yaba daba-', 'doo'], break_on_hyphens=True)
        self.check_wrap(text, 10, ['yaba', 'daba-doo'], break_on_hyphens=False)

    def test_bad_width(self):
        if False:
            i = 10
            return i + 15
        text = "Whatever, it doesn't matter."
        self.assertRaises(ValueError, wrap, text, 0)
        self.assertRaises(ValueError, wrap, text, -1)

    def test_no_split_at_umlaut(self):
        if False:
            while True:
                i = 10
        text = 'Die Empfänger-Auswahl'
        self.check_wrap(text, 13, ['Die', 'Empfänger-', 'Auswahl'])

    def test_umlaut_followed_by_dash(self):
        if False:
            while True:
                i = 10
        text = 'aa ää-ää'
        self.check_wrap(text, 7, ['aa ää-', 'ää'])

    def test_non_breaking_space(self):
        if False:
            i = 10
            return i + 15
        text = 'This is a sentence with non-breaking\xa0space.'
        self.check_wrap(text, 20, ['This is a sentence', 'with non-', 'breaking\xa0space.'], break_on_hyphens=True)
        self.check_wrap(text, 20, ['This is a sentence', 'with', 'non-breaking\xa0space.'], break_on_hyphens=False)

    def test_narrow_non_breaking_space(self):
        if False:
            for i in range(10):
                print('nop')
        text = 'This is a sentence with non-breaking\u202fspace.'
        self.check_wrap(text, 20, ['This is a sentence', 'with non-', 'breaking\u202fspace.'], break_on_hyphens=True)
        self.check_wrap(text, 20, ['This is a sentence', 'with', 'non-breaking\u202fspace.'], break_on_hyphens=False)

class MaxLinesTestCase(BaseTestCase):
    text = "Hello there, how are you this fine day?  I'm glad to hear it!"

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_wrap(self.text, 12, ['Hello [...]'], max_lines=0)
        self.check_wrap(self.text, 12, ['Hello [...]'], max_lines=1)
        self.check_wrap(self.text, 12, ['Hello there,', 'how [...]'], max_lines=2)
        self.check_wrap(self.text, 13, ['Hello there,', 'how are [...]'], max_lines=2)
        self.check_wrap(self.text, 80, [self.text], max_lines=1)
        self.check_wrap(self.text, 12, ['Hello there,', 'how are you', 'this fine', "day?  I'm", 'glad to hear', 'it!'], max_lines=6)

    def test_spaces(self):
        if False:
            while True:
                i = 10
        self.check_wrap(self.text, 12, ['Hello there,', 'how are you', 'this fine', 'day? [...]'], max_lines=4)
        self.check_wrap(self.text, 6, ['Hello', '[...]'], max_lines=2)
        self.check_wrap(self.text + ' ' * 10, 12, ['Hello there,', 'how are you', 'this fine', "day?  I'm", 'glad to hear', 'it!'], max_lines=6)

    def test_placeholder(self):
        if False:
            i = 10
            return i + 15
        self.check_wrap(self.text, 12, ['Hello...'], max_lines=1, placeholder='...')
        self.check_wrap(self.text, 12, ['Hello there,', 'how are...'], max_lines=2, placeholder='...')
        with self.assertRaises(ValueError):
            wrap(self.text, 16, initial_indent='    ', max_lines=1, placeholder=' [truncated]...')
        with self.assertRaises(ValueError):
            wrap(self.text, 16, subsequent_indent='    ', max_lines=2, placeholder=' [truncated]...')
        self.check_wrap(self.text, 16, ['    Hello there,', '  [truncated]...'], max_lines=2, initial_indent='    ', subsequent_indent='  ', placeholder=' [truncated]...')
        self.check_wrap(self.text, 16, ['  [truncated]...'], max_lines=1, initial_indent='  ', subsequent_indent='    ', placeholder=' [truncated]...')
        self.check_wrap(self.text, 80, [self.text], placeholder='.' * 1000)

    def test_placeholder_backtrack(self):
        if False:
            return 10
        text = 'Good grief Python features are advancing quickly!'
        self.check_wrap(text, 12, ['Good grief', 'Python*****'], max_lines=3, placeholder='*****')

class LongWordTestCase(BaseTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.wrapper = TextWrapper()
        self.text = 'Did you say "supercalifragilisticexpialidocious?"\nHow *do* you spell that odd word, anyways?\n'

    def test_break_long(self):
        if False:
            i = 10
            return i + 15
        self.check_wrap(self.text, 30, ['Did you say "supercalifragilis', 'ticexpialidocious?" How *do*', 'you spell that odd word,', 'anyways?'])
        self.check_wrap(self.text, 50, ['Did you say "supercalifragilisticexpialidocious?"', 'How *do* you spell that odd word, anyways?'])
        self.check_wrap('-' * 10 + 'hello', 10, ['----------', '               h', '               e', '               l', '               l', '               o'], subsequent_indent=' ' * 15)
        self.check_wrap(self.text, 12, ['Did you say ', '"supercalifr', 'agilisticexp', 'ialidocious?', '" How *do*', 'you spell', 'that odd', 'word,', 'anyways?'])

    def test_nobreak_long(self):
        if False:
            while True:
                i = 10
        self.wrapper.break_long_words = 0
        self.wrapper.width = 30
        expect = ['Did you say', '"supercalifragilisticexpialidocious?"', 'How *do* you spell that odd', 'word, anyways?']
        result = self.wrapper.wrap(self.text)
        self.check(result, expect)
        result = wrap(self.text, width=30, break_long_words=0)
        self.check(result, expect)

    def test_max_lines_long(self):
        if False:
            while True:
                i = 10
        self.check_wrap(self.text, 12, ['Did you say ', '"supercalifr', 'agilisticexp', '[...]'], max_lines=4)

class LongWordWithHyphensTestCase(BaseTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.wrapper = TextWrapper()
        self.text1 = 'We used enyzme 2-succinyl-6-hydroxy-2,4-cyclohexadiene-1-carboxylate synthase.\n'
        self.text2 = '1234567890-1234567890--this_is_a_very_long_option_indeed-good-bye"\n'

    def test_break_long_words_on_hyphen(self):
        if False:
            i = 10
            return i + 15
        expected = ['We used enyzme 2-succinyl-6-hydroxy-2,4-', 'cyclohexadiene-1-carboxylate synthase.']
        self.check_wrap(self.text1, 50, expected)
        expected = ['We used', 'enyzme 2-', 'succinyl-', '6-hydroxy-', '2,4-', 'cyclohexad', 'iene-1-', 'carboxylat', 'e', 'synthase.']
        self.check_wrap(self.text1, 10, expected)
        expected = ['1234567890', '-123456789', '0--this_is', '_a_very_lo', 'ng_option_', 'indeed-', 'good-bye"']
        self.check_wrap(self.text2, 10, expected)

    def test_break_long_words_not_on_hyphen(self):
        if False:
            i = 10
            return i + 15
        expected = ['We used enyzme 2-succinyl-6-hydroxy-2,4-cyclohexad', 'iene-1-carboxylate synthase.']
        self.check_wrap(self.text1, 50, expected, break_on_hyphens=False)
        expected = ['We used', 'enyzme 2-s', 'uccinyl-6-', 'hydroxy-2,', '4-cyclohex', 'adiene-1-c', 'arboxylate', 'synthase.']
        self.check_wrap(self.text1, 10, expected, break_on_hyphens=False)
        expected = ['1234567890', '-123456789', '0--this_is', '_a_very_lo', 'ng_option_', 'indeed-', 'good-bye"']
        self.check_wrap(self.text2, 10, expected)

    def test_break_on_hyphen_but_not_long_words(self):
        if False:
            return 10
        expected = ['We used enyzme', '2-succinyl-6-hydroxy-2,4-cyclohexadiene-1-carboxylate', 'synthase.']
        self.check_wrap(self.text1, 50, expected, break_long_words=False)
        expected = ['We used', 'enyzme', '2-succinyl-6-hydroxy-2,4-cyclohexadiene-1-carboxylate', 'synthase.']
        self.check_wrap(self.text1, 10, expected, break_long_words=False)
        expected = ['1234567890', '-123456789', '0--this_is', '_a_very_lo', 'ng_option_', 'indeed-', 'good-bye"']
        self.check_wrap(self.text2, 10, expected)

    def test_do_not_break_long_words_or_on_hyphens(self):
        if False:
            for i in range(10):
                print('nop')
        expected = ['We used enyzme', '2-succinyl-6-hydroxy-2,4-cyclohexadiene-1-carboxylate', 'synthase.']
        self.check_wrap(self.text1, 50, expected, break_long_words=False, break_on_hyphens=False)
        expected = ['We used', 'enyzme', '2-succinyl-6-hydroxy-2,4-cyclohexadiene-1-carboxylate', 'synthase.']
        self.check_wrap(self.text1, 10, expected, break_long_words=False, break_on_hyphens=False)
        expected = ['1234567890', '-123456789', '0--this_is', '_a_very_lo', 'ng_option_', 'indeed-', 'good-bye"']
        self.check_wrap(self.text2, 10, expected)

class IndentTestCases(BaseTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.text = 'This paragraph will be filled, first without any indentation,\nand then with some (including a hanging indent).'

    def test_fill(self):
        if False:
            print('Hello World!')
        expect = 'This paragraph will be filled, first\nwithout any indentation, and then with\nsome (including a hanging indent).'
        result = fill(self.text, 40)
        self.check(result, expect)

    def test_initial_indent(self):
        if False:
            i = 10
            return i + 15
        expect = ['     This paragraph will be filled,', 'first without any indentation, and then', 'with some (including a hanging indent).']
        result = wrap(self.text, 40, initial_indent='     ')
        self.check(result, expect)
        expect = '\n'.join(expect)
        result = fill(self.text, 40, initial_indent='     ')
        self.check(result, expect)

    def test_subsequent_indent(self):
        if False:
            return 10
        expect = '  * This paragraph will be filled, first\n    without any indentation, and then\n    with some (including a hanging\n    indent).'
        result = fill(self.text, 40, initial_indent='  * ', subsequent_indent='    ')
        self.check(result, expect)

class DedentTestCase(unittest.TestCase):

    def assertUnchanged(self, text):
        if False:
            i = 10
            return i + 15
        "assert that dedent() has no effect on 'text'"
        self.assertEqual(text, dedent(text))

    def test_dedent_nomargin(self):
        if False:
            print('Hello World!')
        text = "Hello there.\nHow are you?\nOh good, I'm glad."
        self.assertUnchanged(text)
        text = 'Hello there.\n\nBoo!'
        self.assertUnchanged(text)
        text = 'Hello there.\n  This is indented.'
        self.assertUnchanged(text)
        text = 'Hello there.\n\n  Boo!\n'
        self.assertUnchanged(text)

    def test_dedent_even(self):
        if False:
            print('Hello World!')
        text = '  Hello there.\n  How are ya?\n  Oh good.'
        expect = 'Hello there.\nHow are ya?\nOh good.'
        self.assertEqual(expect, dedent(text))
        text = '  Hello there.\n\n  How are ya?\n  Oh good.\n'
        expect = 'Hello there.\n\nHow are ya?\nOh good.\n'
        self.assertEqual(expect, dedent(text))
        text = '  Hello there.\n  \n  How are ya?\n  Oh good.\n'
        expect = 'Hello there.\n\nHow are ya?\nOh good.\n'
        self.assertEqual(expect, dedent(text))

    def test_dedent_uneven(self):
        if False:
            return 10
        text = '        def foo():\n            while 1:\n                return foo\n        '
        expect = 'def foo():\n    while 1:\n        return foo\n'
        self.assertEqual(expect, dedent(text))
        text = '  Foo\n    Bar\n\n   Baz\n'
        expect = 'Foo\n  Bar\n\n Baz\n'
        self.assertEqual(expect, dedent(text))
        text = '  Foo\n    Bar\n \n   Baz\n'
        expect = 'Foo\n  Bar\n\n Baz\n'
        self.assertEqual(expect, dedent(text))

    def test_dedent_declining(self):
        if False:
            i = 10
            return i + 15
        text = '     Foo\n    Bar\n'
        expect = ' Foo\nBar\n'
        self.assertEqual(expect, dedent(text))
        text = '     Foo\n\n    Bar\n'
        expect = ' Foo\n\nBar\n'
        self.assertEqual(expect, dedent(text))
        text = '     Foo\n    \n    Bar\n'
        expect = ' Foo\n\nBar\n'
        self.assertEqual(expect, dedent(text))

    def test_dedent_preserve_internal_tabs(self):
        if False:
            for i in range(10):
                print('nop')
        text = '  hello\tthere\n  how are\tyou?'
        expect = 'hello\tthere\nhow are\tyou?'
        self.assertEqual(expect, dedent(text))
        self.assertEqual(expect, dedent(expect))

    def test_dedent_preserve_margin_tabs(self):
        if False:
            i = 10
            return i + 15
        text = '  hello there\n\thow are you?'
        self.assertUnchanged(text)
        text = '        hello there\n\thow are you?'
        self.assertUnchanged(text)
        text = '\thello there\n\thow are you?'
        expect = 'hello there\nhow are you?'
        self.assertEqual(expect, dedent(text))
        text = '  \thello there\n  \thow are you?'
        self.assertEqual(expect, dedent(text))
        text = '  \t  hello there\n  \t  how are you?'
        self.assertEqual(expect, dedent(text))
        text = '  \thello there\n  \t  how are you?'
        expect = 'hello there\n  how are you?'
        self.assertEqual(expect, dedent(text))
        text = "  \thello there\n   \thow are you?\n \tI'm fine, thanks"
        expect = " \thello there\n  \thow are you?\n\tI'm fine, thanks"
        self.assertEqual(expect, dedent(text))

class IndentTestCase(unittest.TestCase):
    ROUNDTRIP_CASES = ('Hi.\nThis is a test.\nTesting.', 'Hi.\nThis is a test.\n\nTesting.', '\nHi.\nThis is a test.\nTesting.\n')
    CASES = ROUNDTRIP_CASES + ('Hi.\r\nThis is a test.\r\nTesting.\r\n', '\nHi.\r\nThis is a test.\n\r\nTesting.\r\n\n')

    def test_indent_nomargin_default(self):
        if False:
            return 10
        for text in self.CASES:
            self.assertEqual(indent(text, ''), text)

    def test_indent_nomargin_explicit_default(self):
        if False:
            i = 10
            return i + 15
        for text in self.CASES:
            self.assertEqual(indent(text, '', None), text)

    def test_indent_nomargin_all_lines(self):
        if False:
            for i in range(10):
                print('nop')
        predicate = lambda line: True
        for text in self.CASES:
            self.assertEqual(indent(text, '', predicate), text)

    def test_indent_no_lines(self):
        if False:
            return 10
        predicate = lambda line: False
        for text in self.CASES:
            self.assertEqual(indent(text, '    ', predicate), text)

    def test_roundtrip_spaces(self):
        if False:
            return 10
        for text in self.ROUNDTRIP_CASES:
            self.assertEqual(dedent(indent(text, '    ')), text)

    def test_roundtrip_tabs(self):
        if False:
            while True:
                i = 10
        for text in self.ROUNDTRIP_CASES:
            self.assertEqual(dedent(indent(text, '\t\t')), text)

    def test_roundtrip_mixed(self):
        if False:
            for i in range(10):
                print('nop')
        for text in self.ROUNDTRIP_CASES:
            self.assertEqual(dedent(indent(text, ' \t  \t ')), text)

    def test_indent_default(self):
        if False:
            i = 10
            return i + 15
        prefix = '  '
        expected = ('  Hi.\n  This is a test.\n  Testing.', '  Hi.\n  This is a test.\n\n  Testing.', '\n  Hi.\n  This is a test.\n  Testing.\n', '  Hi.\r\n  This is a test.\r\n  Testing.\r\n', '\n  Hi.\r\n  This is a test.\n\r\n  Testing.\r\n\n')
        for (text, expect) in zip(self.CASES, expected):
            self.assertEqual(indent(text, prefix), expect)

    def test_indent_explicit_default(self):
        if False:
            for i in range(10):
                print('nop')
        prefix = '  '
        expected = ('  Hi.\n  This is a test.\n  Testing.', '  Hi.\n  This is a test.\n\n  Testing.', '\n  Hi.\n  This is a test.\n  Testing.\n', '  Hi.\r\n  This is a test.\r\n  Testing.\r\n', '\n  Hi.\r\n  This is a test.\n\r\n  Testing.\r\n\n')
        for (text, expect) in zip(self.CASES, expected):
            self.assertEqual(indent(text, prefix, None), expect)

    def test_indent_all_lines(self):
        if False:
            return 10
        prefix = '  '
        expected = ('  Hi.\n  This is a test.\n  Testing.', '  Hi.\n  This is a test.\n  \n  Testing.', '  \n  Hi.\n  This is a test.\n  Testing.\n', '  Hi.\r\n  This is a test.\r\n  Testing.\r\n', '  \n  Hi.\r\n  This is a test.\n  \r\n  Testing.\r\n  \n')
        predicate = lambda line: True
        for (text, expect) in zip(self.CASES, expected):
            self.assertEqual(indent(text, prefix, predicate), expect)

    def test_indent_empty_lines(self):
        if False:
            i = 10
            return i + 15
        prefix = '  '
        expected = ('Hi.\nThis is a test.\nTesting.', 'Hi.\nThis is a test.\n  \nTesting.', '  \nHi.\nThis is a test.\nTesting.\n', 'Hi.\r\nThis is a test.\r\nTesting.\r\n', '  \nHi.\r\nThis is a test.\n  \r\nTesting.\r\n  \n')
        predicate = lambda line: not line.strip()
        for (text, expect) in zip(self.CASES, expected):
            self.assertEqual(indent(text, prefix, predicate), expect)

class ShortenTestCase(BaseTestCase):

    def check_shorten(self, text, width, expect, **kwargs):
        if False:
            i = 10
            return i + 15
        result = shorten(text, width, **kwargs)
        self.check(result, expect)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        text = "Hello there, how are you this fine day? I'm glad to hear it!"
        self.check_shorten(text, 18, 'Hello there, [...]')
        self.check_shorten(text, len(text), text)
        self.check_shorten(text, len(text) - 1, "Hello there, how are you this fine day? I'm glad to [...]")

    def test_placeholder(self):
        if False:
            for i in range(10):
                print('nop')
        text = "Hello there, how are you this fine day? I'm glad to hear it!"
        self.check_shorten(text, 17, 'Hello there,$$', placeholder='$$')
        self.check_shorten(text, 18, 'Hello there, how$$', placeholder='$$')
        self.check_shorten(text, 18, 'Hello there, $$', placeholder=' $$')
        self.check_shorten(text, len(text), text, placeholder='$$')
        self.check_shorten(text, len(text) - 1, "Hello there, how are you this fine day? I'm glad to hear$$", placeholder='$$')

    def test_empty_string(self):
        if False:
            while True:
                i = 10
        self.check_shorten('', 6, '')

    def test_whitespace(self):
        if False:
            print('Hello World!')
        text = '\n            This is a  paragraph that  already has\n            line breaks and \t tabs too.'
        self.check_shorten(text, 62, 'This is a paragraph that already has line breaks and tabs too.')
        self.check_shorten(text, 61, 'This is a paragraph that already has line breaks and [...]')
        self.check_shorten('hello      world!  ', 12, 'hello world!')
        self.check_shorten('hello      world!  ', 11, 'hello [...]')
        self.check_shorten('hello      world!  ', 10, '[...]')

    def test_width_too_small_for_placeholder(self):
        if False:
            return 10
        shorten('x' * 20, width=8, placeholder='(......)')
        with self.assertRaises(ValueError):
            shorten('x' * 20, width=8, placeholder='(.......)')

    def test_first_word_too_long_but_placeholder_fits(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_shorten('Helloo', 5, '[...]')
if __name__ == '__main__':
    unittest.main()