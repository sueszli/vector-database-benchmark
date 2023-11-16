import os
import sys
from textwrap import dedent
from twisted.persisted import sob
from twisted.persisted.styles import Ephemeral
from twisted.python import components
from twisted.trial import unittest

class Dummy(components.Componentized):
    pass
objects = [1, 'hello', (1, 'hello'), [1, 'hello'], {1: 'hello'}]

class FakeModule:
    pass

class PersistTests(unittest.TestCase):

    def testStyles(self):
        if False:
            for i in range(10):
                print('nop')
        for o in objects:
            p = sob.Persistent(o, '')
            for style in 'source pickle'.split():
                p.setStyle(style)
                p.save(filename='persisttest.' + style)
                o1 = sob.load('persisttest.' + style, style)
                self.assertEqual(o, o1)

    def testStylesBeingSet(self):
        if False:
            for i in range(10):
                print('nop')
        o = Dummy()
        o.foo = 5
        o.setComponent(sob.IPersistable, sob.Persistent(o, 'lala'))
        for style in 'source pickle'.split():
            sob.IPersistable(o).setStyle(style)
            sob.IPersistable(o).save(filename='lala.' + style)
            o1 = sob.load('lala.' + style, style)
            self.assertEqual(o.foo, o1.foo)
            self.assertEqual(sob.IPersistable(o1).style, style)

    def testPassphraseError(self):
        if False:
            while True:
                i = 10
        '\n        Calling save() with a passphrase is an error.\n        '
        p = sob.Persistant(None, 'object')
        self.assertRaises(TypeError, p.save, 'filename.pickle', passphrase='abc')

    def testNames(self):
        if False:
            i = 10
            return i + 15
        o = [1, 2, 3]
        p = sob.Persistent(o, 'object')
        for style in 'source pickle'.split():
            p.setStyle(style)
            p.save()
            o1 = sob.load('object.ta' + style[0], style)
            self.assertEqual(o, o1)
            for tag in 'lala lolo'.split():
                p.save(tag)
                o1 = sob.load('object-' + tag + '.ta' + style[0], style)
                self.assertEqual(o, o1)

    def testPython(self):
        if False:
            for i in range(10):
                print('nop')
        with open('persisttest.python', 'w') as f:
            f.write('foo=[1,2,3] ')
        o = sob.loadValueFromFile('persisttest.python', 'foo')
        self.assertEqual(o, [1, 2, 3])

    def testTypeGuesser(self):
        if False:
            print('Hello World!')
        self.assertRaises(KeyError, sob.guessType, 'file.blah')
        self.assertEqual('python', sob.guessType('file.py'))
        self.assertEqual('python', sob.guessType('file.tac'))
        self.assertEqual('python', sob.guessType('file.etac'))
        self.assertEqual('pickle', sob.guessType('file.tap'))
        self.assertEqual('pickle', sob.guessType('file.etap'))
        self.assertEqual('source', sob.guessType('file.tas'))
        self.assertEqual('source', sob.guessType('file.etas'))

    def testEverythingEphemeralGetattr(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_EverythingEphermal.__getattr__} will proxy the __main__ module as an\n        L{Ephemeral} object, and during load will be transparent, but after\n        load will return L{Ephemeral} objects from any accessed attributes.\n        '
        self.fakeMain.testMainModGetattr = 1
        dirname = self.mktemp()
        os.mkdir(dirname)
        filename = os.path.join(dirname, 'persisttest.ee_getattr')
        global mainWhileLoading
        mainWhileLoading = None
        with open(filename, 'w') as f:
            f.write(dedent('\n            app = []\n            import __main__\n            app.append(__main__.testMainModGetattr == 1)\n            try:\n                __main__.somethingElse\n            except AttributeError:\n                app.append(True)\n            else:\n                app.append(False)\n            from twisted.test import test_sob\n            test_sob.mainWhileLoading = __main__\n            '))
        loaded = sob.load(filename, 'source')
        self.assertIsInstance(loaded, list)
        self.assertTrue(loaded[0], 'Expected attribute not set.')
        self.assertTrue(loaded[1], 'Unexpected attribute set.')
        self.assertIsInstance(mainWhileLoading, Ephemeral)
        self.assertIsInstance(mainWhileLoading.somethingElse, Ephemeral)
        del mainWhileLoading

    def testEverythingEphemeralSetattr(self):
        if False:
            i = 10
            return i + 15
        "\n        Verify that _EverythingEphemeral.__setattr__ won't affect __main__.\n        "
        self.fakeMain.testMainModSetattr = 1
        dirname = self.mktemp()
        os.mkdir(dirname)
        filename = os.path.join(dirname, 'persisttest.ee_setattr')
        with open(filename, 'w') as f:
            f.write('import __main__\n')
            f.write('__main__.testMainModSetattr = 2\n')
            f.write('app = None\n')
        sob.load(filename, 'source')
        self.assertEqual(self.fakeMain.testMainModSetattr, 1)

    def testEverythingEphemeralException(self):
        if False:
            print('Hello World!')
        "\n        Test that an exception during load() won't cause _EE to mask __main__\n        "
        dirname = self.mktemp()
        os.mkdir(dirname)
        filename = os.path.join(dirname, 'persisttest.ee_exception')
        with open(filename, 'w') as f:
            f.write('raise ValueError\n')
        self.assertRaises(ValueError, sob.load, filename, 'source')
        self.assertEqual(type(sys.modules['__main__']), FakeModule)

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Replace the __main__ module with a fake one, so that it can be mutated\n        in tests\n        '
        self.realMain = sys.modules['__main__']
        self.fakeMain = sys.modules['__main__'] = FakeModule()

    def tearDown(self):
        if False:
            return 10
        '\n        Restore __main__ to its original value\n        '
        sys.modules['__main__'] = self.realMain