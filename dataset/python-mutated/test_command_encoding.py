"""Tests for the Command.encoding_type interface."""
from bzrlib.tests import TestCase
from bzrlib.commands import Command, register_command, plugin_cmds

class cmd_echo_exact(Command):
    """This command just repeats what it is given.

    It decodes the argument, and then writes it to stdout.
    """
    takes_args = ['text']
    encoding_type = 'exact'

    def run(self, text=None):
        if False:
            i = 10
            return i + 15
        self.outf.write(text)

class cmd_echo_strict(cmd_echo_exact):
    """Raise a UnicodeError for unrepresentable characters."""
    encoding_type = 'strict'

class cmd_echo_replace(cmd_echo_exact):
    """Replace bogus unicode characters."""
    encoding_type = 'replace'

class TestCommandEncoding(TestCase):

    def test_exact(self):
        if False:
            while True:
                i = 10

        def bzr(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return self.run_bzr(*args, **kwargs)[0]
        register_command(cmd_echo_exact)
        try:
            self.assertEqual('foo', bzr('echo-exact foo'))
            self.assertRaises(UnicodeEncodeError, bzr, ['echo-exact', u'fooµ'])
        finally:
            plugin_cmds.remove('echo-exact')

    def test_strict_utf8(self):
        if False:
            for i in range(10):
                print('nop')

        def bzr(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            kwargs['encoding'] = 'utf-8'
            return self.run_bzr(*args, **kwargs)[0]
        register_command(cmd_echo_strict)
        try:
            self.assertEqual('foo', bzr('echo-strict foo'))
            self.assertEqual(u'fooµ'.encode('utf-8'), bzr(['echo-strict', u'fooµ']))
        finally:
            plugin_cmds.remove('echo-strict')

    def test_strict_ascii(self):
        if False:
            while True:
                i = 10

        def bzr(*args, **kwargs):
            if False:
                while True:
                    i = 10
            kwargs['encoding'] = 'ascii'
            return self.run_bzr(*args, **kwargs)[0]
        register_command(cmd_echo_strict)
        try:
            self.assertEqual('foo', bzr('echo-strict foo'))
            self.assertRaises(UnicodeEncodeError, bzr, ['echo-strict', u'fooµ'])
        finally:
            plugin_cmds.remove('echo-strict')

    def test_replace_utf8(self):
        if False:
            i = 10
            return i + 15

        def bzr(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            kwargs['encoding'] = 'utf-8'
            return self.run_bzr(*args, **kwargs)[0]
        register_command(cmd_echo_replace)
        try:
            self.assertEqual('foo', bzr('echo-replace foo'))
            self.assertEqual(u'fooµ'.encode('utf-8'), bzr(['echo-replace', u'fooµ']))
        finally:
            plugin_cmds.remove('echo-replace')

    def test_replace_ascii(self):
        if False:
            for i in range(10):
                print('nop')

        def bzr(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            kwargs['encoding'] = 'ascii'
            return self.run_bzr(*args, **kwargs)[0]
        register_command(cmd_echo_replace)
        try:
            self.assertEqual('foo', bzr('echo-replace foo'))
            self.assertEqual('foo?', bzr(['echo-replace', u'fooµ']))
        finally:
            plugin_cmds.remove('echo-replace')