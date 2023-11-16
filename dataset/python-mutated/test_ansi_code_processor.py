import unittest
from qtconsole.ansi_code_processor import AnsiCodeProcessor

class TestAnsiCodeProcessor(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.processor = AnsiCodeProcessor()

    def test_clear(self):
        if False:
            i = 10
            return i + 15
        ' Do control sequences for clearing the console work?\n        '
        string = '\x1b[2J\x1b[K'
        i = -1
        for (i, substring) in enumerate(self.processor.split_string(string)):
            if i == 0:
                self.assertEqual(len(self.processor.actions), 1)
                action = self.processor.actions[0]
                self.assertEqual(action.action, 'erase')
                self.assertEqual(action.area, 'screen')
                self.assertEqual(action.erase_to, 'all')
            elif i == 1:
                self.assertEqual(len(self.processor.actions), 1)
                action = self.processor.actions[0]
                self.assertEqual(action.action, 'erase')
                self.assertEqual(action.area, 'line')
                self.assertEqual(action.erase_to, 'end')
            else:
                self.fail('Too many substrings.')
        self.assertEqual(i, 1, 'Too few substrings.')

    def test_colors(self):
        if False:
            print('Hello World!')
        ' Do basic controls sequences for colors work?\n        '
        string = 'first\x1b[34mblue\x1b[0mlast'
        i = -1
        for (i, substring) in enumerate(self.processor.split_string(string)):
            if i == 0:
                self.assertEqual(substring, 'first')
                self.assertEqual(self.processor.foreground_color, None)
            elif i == 1:
                self.assertEqual(substring, 'blue')
                self.assertEqual(self.processor.foreground_color, 4)
            elif i == 2:
                self.assertEqual(substring, 'last')
                self.assertEqual(self.processor.foreground_color, None)
            else:
                self.fail('Too many substrings.')
        self.assertEqual(i, 2, 'Too few substrings.')

    def test_colors_xterm(self):
        if False:
            for i in range(10):
                print('nop')
        ' Do xterm-specific control sequences for colors work?\n        '
        string = '\x1b]4;20;rgb:ff/ff/ff\x1b\x1b]4;25;rgbi:1.0/1.0/1.0\x1b'
        substrings = list(self.processor.split_string(string))
        desired = {20: (255, 255, 255), 25: (255, 255, 255)}
        self.assertEqual(self.processor.color_map, desired)
        string = '\x1b[38;5;20m\x1b[48;5;25m'
        substrings = list(self.processor.split_string(string))
        self.assertEqual(self.processor.foreground_color, 20)
        self.assertEqual(self.processor.background_color, 25)

    def test_true_color(self):
        if False:
            i = 10
            return i + 15
        'Do 24bit True Color control sequences?\n        '
        string = '\x1b[38;2;255;100;0m\x1b[48;2;100;100;100m'
        substrings = list(self.processor.split_string(string))
        self.assertEqual(self.processor.foreground_color, [255, 100, 0])
        self.assertEqual(self.processor.background_color, [100, 100, 100])

    def test_scroll(self):
        if False:
            for i in range(10):
                print('nop')
        ' Do control sequences for scrolling the buffer work?\n        '
        string = '\x1b[5S\x1b[T'
        i = -1
        for (i, substring) in enumerate(self.processor.split_string(string)):
            if i == 0:
                self.assertEqual(len(self.processor.actions), 1)
                action = self.processor.actions[0]
                self.assertEqual(action.action, 'scroll')
                self.assertEqual(action.dir, 'up')
                self.assertEqual(action.unit, 'line')
                self.assertEqual(action.count, 5)
            elif i == 1:
                self.assertEqual(len(self.processor.actions), 1)
                action = self.processor.actions[0]
                self.assertEqual(action.action, 'scroll')
                self.assertEqual(action.dir, 'down')
                self.assertEqual(action.unit, 'line')
                self.assertEqual(action.count, 1)
            else:
                self.fail('Too many substrings.')
        self.assertEqual(i, 1, 'Too few substrings.')

    def test_formfeed(self):
        if False:
            for i in range(10):
                print('nop')
        ' Are formfeed characters processed correctly?\n        '
        string = '\x0c'
        self.assertEqual(list(self.processor.split_string(string)), [''])
        self.assertEqual(len(self.processor.actions), 1)
        action = self.processor.actions[0]
        self.assertEqual(action.action, 'scroll')
        self.assertEqual(action.dir, 'down')
        self.assertEqual(action.unit, 'page')
        self.assertEqual(action.count, 1)

    def test_carriage_return(self):
        if False:
            i = 10
            return i + 15
        ' Are carriage return characters processed correctly?\n        '
        string = 'foo\rbar'
        splits = []
        actions = []
        for split in self.processor.split_string(string):
            splits.append(split)
            actions.append([action.action for action in self.processor.actions])
        self.assertEqual(splits, ['foo', None, 'bar'])
        self.assertEqual(actions, [[], ['carriage-return'], []])

    def test_carriage_return_newline(self):
        if False:
            while True:
                i = 10
        'transform CRLF to LF'
        string = 'foo\rbar\r\ncat\r\n\n'
        splits = []
        actions = []
        for split in self.processor.split_string(string):
            splits.append(split)
            actions.append([action.action for action in self.processor.actions])
        self.assertEqual(splits, ['foo', None, 'bar', '\r\n', 'cat', '\r\n', '\n'])
        self.assertEqual(actions, [[], ['carriage-return'], [], ['newline'], [], ['newline'], ['newline']])

    def test_beep(self):
        if False:
            print('Hello World!')
        ' Are beep characters processed correctly?\n        '
        string = 'foo\x07bar'
        splits = []
        actions = []
        for split in self.processor.split_string(string):
            splits.append(split)
            actions.append([action.action for action in self.processor.actions])
        self.assertEqual(splits, ['foo', None, 'bar'])
        self.assertEqual(actions, [[], ['beep'], []])

    def test_backspace(self):
        if False:
            print('Hello World!')
        ' Are backspace characters processed correctly?\n        '
        string = 'foo\x08bar'
        splits = []
        actions = []
        for split in self.processor.split_string(string):
            splits.append(split)
            actions.append([action.action for action in self.processor.actions])
        self.assertEqual(splits, ['foo', None, 'bar'])
        self.assertEqual(actions, [[], ['backspace'], []])

    def test_combined(self):
        if False:
            return 10
        ' Are CR and BS characters processed correctly in combination?\n\n        BS is treated as a change in print position, rather than a\n        backwards character deletion.  Therefore a BS at EOL is\n        effectively ignored.\n        '
        string = 'abc\rdef\x08'
        splits = []
        actions = []
        for split in self.processor.split_string(string):
            splits.append(split)
            actions.append([action.action for action in self.processor.actions])
        self.assertEqual(splits, ['abc', None, 'def', None])
        self.assertEqual(actions, [[], ['carriage-return'], [], ['backspace']])
if __name__ == '__main__':
    unittest.main()