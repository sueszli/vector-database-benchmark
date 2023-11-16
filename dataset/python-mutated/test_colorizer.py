"""Test colorizer, coverage 99%."""
from idlelib import colorizer
from test.support import requires
import unittest
from unittest import mock
from idlelib.idle_test.tkinter_testing_utils import run_in_tk_mainloop
from functools import partial
import textwrap
from tkinter import Tk, Text
from idlelib import config
from idlelib.percolator import Percolator
usercfg = colorizer.idleConf.userCfg
testcfg = {'main': config.IdleUserConfParser(''), 'highlight': config.IdleUserConfParser(''), 'keys': config.IdleUserConfParser(''), 'extensions': config.IdleUserConfParser('')}
source = textwrap.dedent('    if True: int (\'1\') # keyword, builtin, string, comment\n    elif False: print(0)  # \'string\' in comment\n    else: float(None)  # if in comment\n    if iF + If + IF: \'keyword matching must respect case\'\n    if\'\': x or\'\'  # valid keyword-string no-space combinations\n    async def f(): await g()\n    # Strings should be entirely colored, including quotes.\n    \'x\', \'\'\'x\'\'\', "x", """x"""\n    \'abc\\\n    def\'\n    \'\'\'abc\\\n    def\'\'\'\n    # All valid prefixes for unicode and byte strings should be colored.\n    r\'x\', u\'x\', R\'x\', U\'x\', f\'x\', F\'x\'\n    fr\'x\', Fr\'x\', fR\'x\', FR\'x\', rf\'x\', rF\'x\', Rf\'x\', RF\'x\'\n    b\'x\',B\'x\', br\'x\',Br\'x\',bR\'x\',BR\'x\', rb\'x\', rB\'x\',Rb\'x\',RB\'x\'\n    # Invalid combinations of legal characters should be half colored.\n    ur\'x\', ru\'x\', uf\'x\', fu\'x\', UR\'x\', ufr\'x\', rfu\'x\', xf\'x\', fx\'x\'\n    match point:\n        case (x, 0) as _:\n            print(f"X={x}")\n        case [_, [_], "_",\n                _]:\n            pass\n        case _ if ("a" if _ else set()): pass\n        case _:\n            raise ValueError("Not a point _")\n    \'\'\'\n    case _:\'\'\'\n    "match x:"\n    ')

def setUpModule():
    if False:
        while True:
            i = 10
    colorizer.idleConf.userCfg = testcfg

def tearDownModule():
    if False:
        while True:
            i = 10
    colorizer.idleConf.userCfg = usercfg

class FunctionTest(unittest.TestCase):

    def test_any(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(colorizer.any('test', ('a', 'b', 'cd')), '(?P<test>a|b|cd)')

    def test_make_pat(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(colorizer.make_pat())

    def test_prog(self):
        if False:
            return 10
        prog = colorizer.prog
        eq = self.assertEqual
        line = 'def f():\n    print("hello")\n'
        m = prog.search(line)
        eq(m.groupdict()['KEYWORD'], 'def')
        m = prog.search(line, m.end())
        eq(m.groupdict()['SYNC'], '\n')
        m = prog.search(line, m.end())
        eq(m.groupdict()['BUILTIN'], 'print')
        m = prog.search(line, m.end())
        eq(m.groupdict()['STRING'], '"hello"')
        m = prog.search(line, m.end())
        eq(m.groupdict()['SYNC'], '\n')

    def test_idprog(self):
        if False:
            print('Hello World!')
        idprog = colorizer.idprog
        m = idprog.match('nospace')
        self.assertIsNone(m)
        m = idprog.match(' space')
        self.assertEqual(m.group(0), ' space')

class ColorConfigTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        requires('gui')
        root = cls.root = Tk()
        root.withdraw()
        cls.text = Text(root)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        del cls.text
        cls.root.update_idletasks()
        cls.root.destroy()
        del cls.root

    def test_color_config(self):
        if False:
            return 10
        text = self.text
        eq = self.assertEqual
        colorizer.color_config(text)
        eq(text['background'], '#ffffff')
        eq(text['foreground'], '#000000')
        eq(text['selectbackground'], 'gray')
        eq(text['selectforeground'], '#000000')
        eq(text['insertbackground'], 'black')
        eq(text['inactiveselectbackground'], 'gray')

class ColorDelegatorInstantiationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        requires('gui')
        root = cls.root = Tk()
        root.withdraw()
        cls.text = Text(root)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        del cls.text
        cls.root.update_idletasks()
        cls.root.destroy()
        del cls.root

    def setUp(self):
        if False:
            return 10
        self.color = colorizer.ColorDelegator()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.color.close()
        self.text.delete('1.0', 'end')
        self.color.resetcache()
        del self.color

    def test_init(self):
        if False:
            print('Hello World!')
        color = self.color
        self.assertIsInstance(color, colorizer.ColorDelegator)

    def test_init_state(self):
        if False:
            print('Hello World!')
        color = self.color
        self.assertIsNone(color.after_id)
        self.assertTrue(color.allow_colorizing)
        self.assertFalse(color.colorizing)
        self.assertFalse(color.stop_colorizing)

class ColorDelegatorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        requires('gui')
        root = cls.root = Tk()
        root.withdraw()
        text = cls.text = Text(root)
        cls.percolator = Percolator(text)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        cls.percolator.close()
        del cls.percolator, cls.text
        cls.root.update_idletasks()
        cls.root.destroy()
        del cls.root

    def setUp(self):
        if False:
            while True:
                i = 10
        self.color = colorizer.ColorDelegator()
        self.percolator.insertfilter(self.color)

    def tearDown(self):
        if False:
            return 10
        self.color.close()
        self.percolator.removefilter(self.color)
        self.text.delete('1.0', 'end')
        self.color.resetcache()
        del self.color

    def test_setdelegate(self):
        if False:
            i = 10
            return i + 15
        color = self.color
        self.assertIsInstance(color.delegate, colorizer.Delegator)
        self.assertEqual(self.root.tk.call('after', 'info', color.after_id)[1], 'timer')

    def test_LoadTagDefs(self):
        if False:
            while True:
                i = 10
        highlight = partial(config.idleConf.GetHighlight, theme='IDLE Classic')
        for (tag, colors) in self.color.tagdefs.items():
            with self.subTest(tag=tag):
                self.assertIn('background', colors)
                self.assertIn('foreground', colors)
                if tag not in ('SYNC', 'TODO'):
                    self.assertEqual(colors, highlight(element=tag.lower()))

    def test_config_colors(self):
        if False:
            for i in range(10):
                print('nop')
        text = self.text
        highlight = partial(config.idleConf.GetHighlight, theme='IDLE Classic')
        for tag in self.color.tagdefs:
            for plane in ('background', 'foreground'):
                with self.subTest(tag=tag, plane=plane):
                    if tag in ('SYNC', 'TODO'):
                        self.assertEqual(text.tag_cget(tag, plane), '')
                    else:
                        self.assertEqual(text.tag_cget(tag, plane), highlight(element=tag.lower())[plane])
        self.assertEqual(text.tag_names()[-1], 'sel')

    @mock.patch.object(colorizer.ColorDelegator, 'notify_range')
    def test_insert(self, mock_notify):
        if False:
            print('Hello World!')
        text = self.text
        text.insert('insert', 'foo')
        self.assertEqual(text.get('1.0', 'end'), 'foo\n')
        mock_notify.assert_called_with('1.0', '1.0+3c')
        text.insert('insert', 'barbaz')
        self.assertEqual(text.get('1.0', 'end'), 'foobarbaz\n')
        mock_notify.assert_called_with('1.3', '1.3+6c')

    @mock.patch.object(colorizer.ColorDelegator, 'notify_range')
    def test_delete(self, mock_notify):
        if False:
            return 10
        text = self.text
        text.insert('insert', 'abcdefghi')
        self.assertEqual(text.get('1.0', 'end'), 'abcdefghi\n')
        text.delete('1.7')
        self.assertEqual(text.get('1.0', 'end'), 'abcdefgi\n')
        mock_notify.assert_called_with('1.7')
        text.delete('1.3', '1.6')
        self.assertEqual(text.get('1.0', 'end'), 'abcgi\n')
        mock_notify.assert_called_with('1.3')

    def test_notify_range(self):
        if False:
            for i in range(10):
                print('nop')
        text = self.text
        color = self.color
        eq = self.assertEqual
        save_id = color.after_id
        eq(self.root.tk.call('after', 'info', save_id)[1], 'timer')
        self.assertFalse(color.colorizing)
        self.assertFalse(color.stop_colorizing)
        self.assertTrue(color.allow_colorizing)
        color.colorizing = True
        color.notify_range('1.0', 'end')
        self.assertFalse(color.stop_colorizing)
        eq(color.after_id, save_id)
        text.after_cancel(save_id)
        color.after_id = None
        color.notify_range('1.0', '1.0+3c')
        self.assertTrue(color.stop_colorizing)
        self.assertIsNotNone(color.after_id)
        eq(self.root.tk.call('after', 'info', color.after_id)[1], 'timer')
        self.assertNotEqual(color.after_id, save_id)
        text.after_cancel(color.after_id)
        color.after_id = None
        color.allow_colorizing = False
        color.notify_range('1.4', '1.4+10c')
        self.assertIsNone(color.after_id)

    def test_toggle_colorize_event(self):
        if False:
            print('Hello World!')
        color = self.color
        eq = self.assertEqual
        self.assertFalse(color.colorizing)
        self.assertFalse(color.stop_colorizing)
        self.assertTrue(color.allow_colorizing)
        eq(self.root.tk.call('after', 'info', color.after_id)[1], 'timer')
        color.toggle_colorize_event()
        self.assertIsNone(color.after_id)
        self.assertFalse(color.colorizing)
        self.assertFalse(color.stop_colorizing)
        self.assertFalse(color.allow_colorizing)
        color.colorizing = True
        color.toggle_colorize_event()
        self.assertIsNone(color.after_id)
        self.assertTrue(color.colorizing)
        self.assertFalse(color.stop_colorizing)
        self.assertTrue(color.allow_colorizing)
        color.toggle_colorize_event()
        self.assertIsNone(color.after_id)
        self.assertTrue(color.colorizing)
        self.assertTrue(color.stop_colorizing)
        self.assertFalse(color.allow_colorizing)
        color.colorizing = False
        color.toggle_colorize_event()
        eq(self.root.tk.call('after', 'info', color.after_id)[1], 'timer')
        self.assertFalse(color.colorizing)
        self.assertTrue(color.stop_colorizing)
        self.assertTrue(color.allow_colorizing)

    @mock.patch.object(colorizer.ColorDelegator, 'recolorize_main')
    def test_recolorize(self, mock_recmain):
        if False:
            for i in range(10):
                print('nop')
        text = self.text
        color = self.color
        eq = self.assertEqual
        text.after_cancel(color.after_id)
        save_delegate = color.delegate
        color.delegate = None
        color.recolorize()
        mock_recmain.assert_not_called()
        color.delegate = save_delegate
        color.allow_colorizing = False
        color.recolorize()
        mock_recmain.assert_not_called()
        color.allow_colorizing = True
        color.colorizing = True
        color.recolorize()
        mock_recmain.assert_not_called()
        color.colorizing = False
        color.recolorize()
        self.assertFalse(color.stop_colorizing)
        self.assertFalse(color.colorizing)
        mock_recmain.assert_called()
        eq(mock_recmain.call_count, 1)
        eq(self.root.tk.call('after', 'info', color.after_id)[1], 'timer')
        text.tag_remove('TODO', '1.0', 'end')
        color.recolorize()
        self.assertFalse(color.stop_colorizing)
        self.assertFalse(color.colorizing)
        mock_recmain.assert_called()
        eq(mock_recmain.call_count, 2)
        self.assertIsNone(color.after_id)

    @mock.patch.object(colorizer.ColorDelegator, 'notify_range')
    def test_recolorize_main(self, mock_notify):
        if False:
            return 10
        text = self.text
        color = self.color
        eq = self.assertEqual
        text.insert('insert', source)
        expected = (('1.0', ('KEYWORD',)), ('1.2', ()), ('1.3', ('KEYWORD',)), ('1.7', ()), ('1.9', ('BUILTIN',)), ('1.14', ('STRING',)), ('1.19', ('COMMENT',)), ('2.1', ('KEYWORD',)), ('2.18', ()), ('2.25', ('COMMENT',)), ('3.6', ('BUILTIN',)), ('3.12', ('KEYWORD',)), ('3.21', ('COMMENT',)), ('4.0', ('KEYWORD',)), ('4.3', ()), ('4.6', ()), ('5.2', ('STRING',)), ('5.8', ('KEYWORD',)), ('5.10', ('STRING',)), ('6.0', ('KEYWORD',)), ('6.10', ('DEFINITION',)), ('6.11', ()), ('8.0', ('STRING',)), ('8.4', ()), ('8.5', ('STRING',)), ('8.12', ()), ('8.14', ('STRING',)), ('19.0', ('KEYWORD',)), ('20.4', ('KEYWORD',)), ('20.16', ('KEYWORD',)), ('24.8', ('KEYWORD',)), ('25.4', ('KEYWORD',)), ('25.9', ('KEYWORD',)), ('25.11', ('KEYWORD',)), ('25.15', ('STRING',)), ('25.19', ('KEYWORD',)), ('25.22', ()), ('25.24', ('KEYWORD',)), ('25.29', ('BUILTIN',)), ('25.37', ('KEYWORD',)), ('26.4', ('KEYWORD',)), ('26.9', ('KEYWORD',)), ('27.25', ('STRING',)), ('27.38', ('STRING',)), ('29.0', ('STRING',)), ('30.1', ('STRING',)), ('1.55', ('SYNC',)), ('2.50', ('SYNC',)), ('3.34', ('SYNC',)))
        text.tag_remove('TODO', '1.0', 'end')
        color.recolorize_main()
        for tag in text.tag_names():
            with self.subTest(tag=tag):
                eq(text.tag_ranges(tag), ())
        text.tag_add('TODO', '1.0', 'end')
        color.recolorize_main()
        for (index, expected_tags) in expected:
            with self.subTest(index=index):
                eq(text.tag_names(index), expected_tags)
        eq(text.tag_nextrange('TODO', '1.0'), ())
        eq(text.tag_nextrange('KEYWORD', '1.0'), ('1.0', '1.2'))
        eq(text.tag_nextrange('COMMENT', '2.0'), ('2.22', '2.43'))
        eq(text.tag_nextrange('SYNC', '2.0'), ('2.43', '3.0'))
        eq(text.tag_nextrange('STRING', '2.0'), ('4.17', '4.53'))
        eq(text.tag_nextrange('STRING', '8.0'), ('8.0', '8.3'))
        eq(text.tag_nextrange('STRING', '8.3'), ('8.5', '8.12'))
        eq(text.tag_nextrange('STRING', '8.12'), ('8.14', '8.17'))
        eq(text.tag_nextrange('STRING', '8.17'), ('8.19', '8.26'))
        eq(text.tag_nextrange('SYNC', '8.0'), ('8.26', '9.0'))
        eq(text.tag_nextrange('SYNC', '30.0'), ('30.10', '32.0'))

    def _assert_highlighting(self, source, tag_ranges):
        if False:
            for i in range(10):
                print('nop')
        "Check highlighting of a given piece of code.\n\n        This inserts just this code into the Text widget. It will then\n        check that the resulting highlighting tag ranges exactly match\n        those described in the given `tag_ranges` dict.\n\n        Note that the irrelevant tags 'sel', 'TODO' and 'SYNC' are\n        ignored.\n        "
        text = self.text
        with mock.patch.object(colorizer.ColorDelegator, 'notify_range'):
            text.delete('1.0', 'end-1c')
            text.insert('insert', source)
            text.tag_add('TODO', '1.0', 'end-1c')
            self.color.recolorize_main()
        text_tag_ranges = {}
        for tag in set(text.tag_names()) - {'sel', 'TODO', 'SYNC'}:
            indexes = [rng.string for rng in text.tag_ranges(tag)]
            for index_pair in zip(indexes[::2], indexes[1::2]):
                text_tag_ranges.setdefault(tag, []).append(index_pair)
        self.assertEqual(text_tag_ranges, tag_ranges)
        with mock.patch.object(colorizer.ColorDelegator, 'notify_range'):
            text.delete('1.0', 'end-1c')

    def test_def_statement(self):
        if False:
            print('Hello World!')
        self._assert_highlighting('def', {'KEYWORD': [('1.0', '1.3')]})
        self._assert_highlighting('def foo:', {'KEYWORD': [('1.0', '1.3')], 'DEFINITION': [('1.4', '1.7')]})
        self._assert_highlighting('def fo', {'KEYWORD': [('1.0', '1.3')], 'DEFINITION': [('1.4', '1.6')]})
        self._assert_highlighting('def ++', {'KEYWORD': [('1.0', '1.3')]})

    def test_match_soft_keyword(self):
        if False:
            while True:
                i = 10
        self._assert_highlighting('match', {'KEYWORD': [('1.0', '1.5')]})
        self._assert_highlighting('match fo', {'KEYWORD': [('1.0', '1.5')]})
        self._assert_highlighting('match foo:', {'KEYWORD': [('1.0', '1.5')]})
        self._assert_highlighting('match and', {'KEYWORD': [('1.6', '1.9')]})
        self._assert_highlighting('match int:', {'KEYWORD': [('1.0', '1.5')], 'BUILTIN': [('1.6', '1.9')]})
        self._assert_highlighting('match^', {})
        self._assert_highlighting('match @', {})
        self._assert_highlighting('match :', {})
        self._assert_highlighting('match\t,', {})
        self._assert_highlighting('match _:', {'KEYWORD': [('1.0', '1.5')]})

    def test_case_soft_keyword(self):
        if False:
            for i in range(10):
                print('nop')
        self._assert_highlighting('case', {'KEYWORD': [('1.0', '1.4')]})
        self._assert_highlighting('case fo', {'KEYWORD': [('1.0', '1.4')]})
        self._assert_highlighting('case foo:', {'KEYWORD': [('1.0', '1.4')]})
        self._assert_highlighting('case and', {'KEYWORD': [('1.5', '1.8')]})
        self._assert_highlighting('case int:', {'KEYWORD': [('1.0', '1.4')], 'BUILTIN': [('1.5', '1.8')]})
        self._assert_highlighting('case^', {})
        self._assert_highlighting('case @', {})
        self._assert_highlighting('case :', {})
        self._assert_highlighting('case\t,', {})
        self._assert_highlighting('case _:', {'KEYWORD': [('1.0', '1.4'), ('1.5', '1.6')]})

    def test_long_multiline_string(self):
        if False:
            while True:
                i = 10
        source = textwrap.dedent('            """a\n            b\n            c\n            d\n            e"""\n            ')
        self._assert_highlighting(source, {'STRING': [('1.0', '5.4')]})

    @run_in_tk_mainloop(delay=50)
    def test_incremental_editing(self):
        if False:
            i = 10
            return i + 15
        text = self.text
        eq = self.assertEqual
        text.insert('insert', 'i')
        yield
        eq(text.tag_nextrange('BUILTIN', '1.0'), ())
        eq(text.tag_nextrange('KEYWORD', '1.0'), ())
        text.insert('insert', 'n')
        yield
        eq(text.tag_nextrange('BUILTIN', '1.0'), ())
        eq(text.tag_nextrange('KEYWORD', '1.0'), ('1.0', '1.2'))
        text.insert('insert', 't')
        yield
        eq(text.tag_nextrange('BUILTIN', '1.0'), ('1.0', '1.3'))
        eq(text.tag_nextrange('KEYWORD', '1.0'), ())
        text.insert('insert', 'e')
        yield
        eq(text.tag_nextrange('BUILTIN', '1.0'), ())
        eq(text.tag_nextrange('KEYWORD', '1.0'), ())
        text.delete('insert-1c', 'insert')
        yield
        eq(text.tag_nextrange('BUILTIN', '1.0'), ('1.0', '1.3'))
        eq(text.tag_nextrange('KEYWORD', '1.0'), ())
        text.delete('insert-1c', 'insert')
        yield
        eq(text.tag_nextrange('BUILTIN', '1.0'), ())
        eq(text.tag_nextrange('KEYWORD', '1.0'), ('1.0', '1.2'))
        text.delete('insert-1c', 'insert')
        yield
        eq(text.tag_nextrange('BUILTIN', '1.0'), ())
        eq(text.tag_nextrange('KEYWORD', '1.0'), ())

    @mock.patch.object(colorizer.ColorDelegator, 'recolorize')
    @mock.patch.object(colorizer.ColorDelegator, 'notify_range')
    def test_removecolors(self, mock_notify, mock_recolorize):
        if False:
            i = 10
            return i + 15
        text = self.text
        color = self.color
        text.insert('insert', source)
        color.recolorize_main()
        text.tag_add('ERROR', '1.0')
        text.tag_add('TODO', '1.0')
        text.tag_add('hit', '1.0')
        for tag in color.tagdefs:
            with self.subTest(tag=tag):
                self.assertNotEqual(text.tag_ranges(tag), ())
        color.removecolors()
        for tag in color.tagdefs:
            with self.subTest(tag=tag):
                self.assertEqual(text.tag_ranges(tag), ())
if __name__ == '__main__':
    unittest.main(verbosity=2)