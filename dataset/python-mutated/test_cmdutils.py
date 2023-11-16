"""Tests for qutebrowser.api.cmdutils."""
import sys
import logging
import types
import enum
import textwrap
import pytest
from qutebrowser.misc import objects
from qutebrowser.commands import cmdexc, argparser, command
from qutebrowser.api import cmdutils
from qutebrowser.utils import usertypes

@pytest.fixture(autouse=True)
def clear_globals(monkeypatch):
    if False:
        return 10
    monkeypatch.setattr(objects, 'commands', {})

def _get_cmd(*args, **kwargs):
    if False:
        return 10
    'Get a command object created via @cmdutils.register.\n\n    Args:\n        Passed to @cmdutils.register decorator\n    '

    @cmdutils.register(*args, **kwargs)
    def fun():
        if False:
            for i in range(10):
                print('nop')
        'Blah.'
    return objects.commands['fun']

class TestCheckOverflow:

    def test_good(self):
        if False:
            i = 10
            return i + 15
        cmdutils.check_overflow(1, 'int')

    def test_bad(self):
        if False:
            while True:
                i = 10
        int32_max = 2 ** 31 - 1
        with pytest.raises(cmdutils.CommandError, match='Numeric argument is too large for internal int representation.'):
            cmdutils.check_overflow(int32_max + 1, 'int')

class TestCheckExclusive:

    @pytest.mark.parametrize('flags', [[], [False, True], [False, False]])
    def test_good(self, flags):
        if False:
            return 10
        cmdutils.check_exclusive(flags, [])

    def test_bad(self):
        if False:
            print('Hello World!')
        with pytest.raises(cmdutils.CommandError, match='Only one of -x/-y/-z can be given!'):
            cmdutils.check_exclusive([True, True], 'xyz')

class TestRegister:

    def test_simple(self):
        if False:
            i = 10
            return i + 15

        @cmdutils.register()
        def fun():
            if False:
                i = 10
                return i + 15
            'Blah.'
        cmd = objects.commands['fun']
        assert cmd.handler is fun
        assert cmd.name == 'fun'
        assert len(objects.commands) == 1

    def test_underlines(self):
        if False:
            return 10
        'Make sure the function name is normalized correctly (_ -> -).'

        @cmdutils.register()
        def eggs_bacon():
            if False:
                i = 10
                return i + 15
            'Blah.'
        assert objects.commands['eggs-bacon'].name == 'eggs-bacon'
        assert 'eggs_bacon' not in objects.commands

    def test_lowercasing(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure the function name is normalized correctly (uppercase).'

        @cmdutils.register()
        def Test():
            if False:
                for i in range(10):
                    print('nop')
            'Blah.'
        assert objects.commands['test'].name == 'test'
        assert 'Test' not in objects.commands

    def test_explicit_name(self):
        if False:
            i = 10
            return i + 15
        'Test register with explicit name.'

        @cmdutils.register(name='foobar')
        def fun():
            if False:
                i = 10
                return i + 15
            'Blah.'
        assert objects.commands['foobar'].name == 'foobar'
        assert 'fun' not in objects.commands
        assert len(objects.commands) == 1

    def test_multiple_registrations(self):
        if False:
            while True:
                i = 10
        'Make sure registering the same name twice raises ValueError.'

        @cmdutils.register(name='foobar')
        def fun():
            if False:
                return 10
            'Blah.'
        with pytest.raises(ValueError):

            @cmdutils.register(name='foobar')
            def fun2():
                if False:
                    while True:
                        i = 10
                'Blah.'

    def test_instance(self):
        if False:
            return 10
        'Make sure the instance gets passed to Command.'

        @cmdutils.register(instance='foobar')
        def fun(self):
            if False:
                i = 10
                return i + 15
            'Blah.'
        assert objects.commands['fun']._instance == 'foobar'

    def test_star_args(self):
        if False:
            for i in range(10):
                print('nop')
        'Check handling of *args.'

        @cmdutils.register()
        def fun(*args):
            if False:
                return 10
            'Blah.'
            assert args == ['one', 'two']
        objects.commands['fun'].parser.parse_args(['one', 'two'])

    def test_star_args_empty(self):
        if False:
            return 10
        'Check handling of *args without any value.'

        @cmdutils.register()
        def fun(*args):
            if False:
                return 10
            'Blah.'
            assert not args
        with pytest.raises(argparser.ArgumentParserError):
            objects.commands['fun'].parser.parse_args([])

    def test_star_args_type(self):
        if False:
            for i in range(10):
                print('nop')
        "Check handling of *args with a type.\n\n        This isn't implemented, so be sure we catch it.\n        "
        with pytest.raises(TypeError):

            @cmdutils.register()
            def fun(*args: int):
                if False:
                    for i in range(10):
                        print('nop')
                'Blah.'

    def test_star_args_optional(self):
        if False:
            for i in range(10):
                print('nop')
        'Check handling of *args withstar_args_optional.'

        @cmdutils.register(star_args_optional=True)
        def fun(*args):
            if False:
                while True:
                    i = 10
            'Blah.'
            assert not args
        cmd = objects.commands['fun']
        cmd.namespace = cmd.parser.parse_args([])
        (args, kwargs) = cmd._get_call_args(win_id=0)
        fun(*args, **kwargs)

    def test_star_args_optional_annotated(self):
        if False:
            for i in range(10):
                print('nop')

        @cmdutils.register(star_args_optional=True)
        def fun(*args: str):
            if False:
                while True:
                    i = 10
            'Blah.'
        cmd = objects.commands['fun']
        cmd.namespace = cmd.parser.parse_args([])
        cmd._get_call_args(win_id=0)

    @pytest.mark.parametrize('inp, expected', [(['--arg'], True), (['-a'], True), ([], False)])
    def test_flag(self, inp, expected):
        if False:
            return 10

        @cmdutils.register()
        def fun(arg=False):
            if False:
                i = 10
                return i + 15
            'Blah.'
            assert arg == expected
        cmd = objects.commands['fun']
        cmd.namespace = cmd.parser.parse_args(inp)
        assert cmd.namespace.arg == expected

    def test_flag_argument(self):
        if False:
            return 10

        @cmdutils.register()
        @cmdutils.argument('arg', flag='b')
        def fun(arg=False):
            if False:
                for i in range(10):
                    print('nop')
            'Blah.'
            assert arg
        cmd = objects.commands['fun']
        with pytest.raises(argparser.ArgumentParserError):
            cmd.parser.parse_args(['-a'])
        cmd.namespace = cmd.parser.parse_args(['-b'])
        assert cmd.namespace.arg
        (args, kwargs) = cmd._get_call_args(win_id=0)
        fun(*args, **kwargs)

    def test_self_without_instance(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError, match='fun is a class method, but instance was not given!'):

            @cmdutils.register()
            def fun(self):
                if False:
                    while True:
                        i = 10
                'Blah.'

    def test_instance_without_self(self):
        if False:
            return 10
        with pytest.raises(TypeError, match='fun is not a class method, but instance was given!'):

            @cmdutils.register(instance='inst')
            def fun():
                if False:
                    while True:
                        i = 10
                'Blah.'

    def test_var_kw(self):
        if False:
            return 10
        with pytest.raises(TypeError, match='fun: functions with varkw arguments are not supported!'):

            @cmdutils.register()
            def fun(**kwargs):
                if False:
                    while True:
                        i = 10
                'Blah.'

    def test_partial_arg(self):
        if False:
            return 10
        'Test with only some arguments decorated with @cmdutils.argument.'

        @cmdutils.register()
        @cmdutils.argument('arg1', flag='b')
        def fun(arg1=False, arg2=False):
            if False:
                i = 10
                return i + 15
            'Blah.'

    def test_win_id(self):
        if False:
            while True:
                i = 10

        @cmdutils.register()
        @cmdutils.argument('win_id', value=cmdutils.Value.win_id)
        def fun(win_id):
            if False:
                return 10
            'Blah.'
        assert objects.commands['fun']._get_call_args(42) == ([42], {})

    def test_count(self):
        if False:
            while True:
                i = 10

        @cmdutils.register()
        @cmdutils.argument('count', value=cmdutils.Value.count)
        def fun(count=0):
            if False:
                print('Hello World!')
            'Blah.'
        assert objects.commands['fun']._get_call_args(42) == ([0], {})

    def test_fill_self(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError, match="fun: Can't fill 'self' with value!"):

            @cmdutils.register(instance='foobar')
            @cmdutils.argument('self', value=cmdutils.Value.count)
            def fun(self):
                if False:
                    for i in range(10):
                        print('nop')
                'Blah.'

    def test_fill_invalid(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError, match="fun: Invalid value='foo' for argument 'arg'!"):

            @cmdutils.register()
            @cmdutils.argument('arg', value='foo')
            def fun(arg):
                if False:
                    print('Hello World!')
                'Blah.'

    def test_count_without_default(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError, match='fun: handler has count parameter without default!'):

            @cmdutils.register()
            @cmdutils.argument('count', value=cmdutils.Value.count)
            def fun(count):
                if False:
                    for i in range(10):
                        print('nop')
                'Blah.'

    @pytest.mark.parametrize('hide', [True, False])
    def test_pos_args(self, hide):
        if False:
            for i in range(10):
                print('nop')

        @cmdutils.register()
        @cmdutils.argument('arg', hide=hide)
        def fun(arg):
            if False:
                return 10
            'Blah.'
        pos_args = objects.commands['fun'].pos_args
        if hide:
            assert pos_args == []
        else:
            assert pos_args == [('arg', 'arg')]

    class Enum(enum.Enum):
        x = enum.auto()
        y = enum.auto()

    @pytest.mark.parametrize('annotation, inp, choices, expected', [('int', '42', None, 42), ('int', 'x', None, cmdexc.ArgumentTypeError), ('str', 'foo', None, 'foo'), ('Union[str, int]', 'foo', None, 'foo'), ('Union[str, int]', '42', None, 42), ('str', 'foo', ['foo'], 'foo'), ('str', 'bar', ['foo'], cmdexc.ArgumentTypeError), ('Union[str, int]', 'foo', ['foo'], 'foo'), ('Union[str, int]', 'bar', ['foo'], cmdexc.ArgumentTypeError), ('Union[str, int]', '42', ['foo'], 42), ('Enum', 'x', None, Enum.x), ('Enum', 'z', None, cmdexc.ArgumentTypeError)])
    def test_typed_args(self, annotation, inp, choices, expected):
        if False:
            print('Hello World!')
        src = textwrap.dedent("\n        from typing import Union\n        from qutebrowser.api import cmdutils\n\n        @cmdutils.register()\n        @cmdutils.argument('arg', choices=choices)\n        def fun(arg: {annotation}):\n            '''Blah.'''\n            return arg\n        ".format(annotation=annotation))
        code = compile(src, '<string>', 'exec')
        print(src)
        ns = {'choices': choices, 'Enum': self.Enum}
        exec(code, ns, ns)
        fun = ns['fun']
        cmd = objects.commands['fun']
        cmd.namespace = cmd.parser.parse_args([inp])
        if expected is cmdexc.ArgumentTypeError:
            with pytest.raises(cmdexc.ArgumentTypeError):
                cmd._get_call_args(win_id=0)
        else:
            (args, kwargs) = cmd._get_call_args(win_id=0)
            assert args == [expected]
            assert kwargs == {}
            ret = fun(*args, **kwargs)
            assert ret == expected

    def test_choices_no_annotation(self):
        if False:
            while True:
                i = 10

        @cmdutils.register()
        @cmdutils.argument('arg', choices=['foo', 'bar'])
        def fun(arg):
            if False:
                i = 10
                return i + 15
            'Blah.'
        cmd = objects.commands['fun']
        cmd.namespace = cmd.parser.parse_args(['fish'])
        with pytest.raises(cmdexc.ArgumentTypeError):
            cmd._get_call_args(win_id=0)

    def test_choices_no_annotation_kwonly(self):
        if False:
            return 10

        @cmdutils.register()
        @cmdutils.argument('arg', choices=['foo', 'bar'])
        def fun(*, arg='foo'):
            if False:
                while True:
                    i = 10
            'Blah.'
        cmd = objects.commands['fun']
        cmd.namespace = cmd.parser.parse_args(['--arg=fish'])
        with pytest.raises(cmdexc.ArgumentTypeError):
            cmd._get_call_args(win_id=0)

    def test_pos_arg_info(self):
        if False:
            return 10

        @cmdutils.register()
        @cmdutils.argument('foo', choices=('a', 'b'))
        @cmdutils.argument('bar', choices=('x', 'y'))
        @cmdutils.argument('opt')
        def fun(foo, bar, opt=False):
            if False:
                while True:
                    i = 10
            'Blah.'
        cmd = objects.commands['fun']
        assert cmd.get_pos_arg_info(0) == command.ArgInfo(choices=('a', 'b'))
        assert cmd.get_pos_arg_info(1) == command.ArgInfo(choices=('x', 'y'))
        with pytest.raises(IndexError):
            cmd.get_pos_arg_info(2)

    def test_keyword_only_without_default(self):
        if False:
            return 10

        def fun(*, target):
            if False:
                print('Hello World!')
            'Blah.'
        with pytest.raises(TypeError, match="fun: handler has keyword only argument 'target' without default!"):
            fun = cmdutils.register()(fun)

    def test_typed_keyword_only_without_default(self):
        if False:
            while True:
                i = 10

        def fun(*, target: int):
            if False:
                i = 10
                return i + 15
            'Blah.'
        with pytest.raises(TypeError, match="fun: handler has keyword only argument 'target' without default!"):
            fun = cmdutils.register()(fun)

class TestArgument:
    """Test the @cmdutils.argument decorator."""

    def test_invalid_argument(self):
        if False:
            return 10
        with pytest.raises(ValueError, match='fun has no argument foo!'):

            @cmdutils.argument('foo')
            def fun(bar):
                if False:
                    return 10
                'Blah.'

    def test_storage(self):
        if False:
            return 10

        @cmdutils.argument('foo', flag='x')
        @cmdutils.argument('bar', flag='y')
        def fun(foo, bar):
            if False:
                return 10
            'Blah.'
        expected = {'foo': command.ArgInfo(flag='x'), 'bar': command.ArgInfo(flag='y')}
        assert fun.qute_args == expected

    def test_arginfo_boolean(self):
        if False:
            i = 10
            return i + 15

        @cmdutils.argument('special1', value=cmdutils.Value.count)
        @cmdutils.argument('special2', value=cmdutils.Value.win_id)
        @cmdutils.argument('normal')
        def fun(special1, special2, normal):
            if False:
                while True:
                    i = 10
            'Blah.'
        assert fun.qute_args['special1'].value
        assert fun.qute_args['special2'].value
        assert not fun.qute_args['normal'].value

    def test_wrong_order(self):
        if False:
            while True:
                i = 10
        'When @cmdutils.argument is used above (after) @register, fail.'
        with pytest.raises(ValueError, match='@cmdutils.argument got called above \\(after\\) @cmdutils.register for fun!'):

            @cmdutils.argument('bar', flag='y')
            @cmdutils.register()
            def fun(bar):
                if False:
                    i = 10
                    return i + 15
                'Blah.'

    def test_no_docstring(self, caplog):
        if False:
            for i in range(10):
                print('nop')
        with caplog.at_level(logging.WARNING):

            @cmdutils.register()
            def fun():
                if False:
                    while True:
                        i = 10
                pass
        assert len(caplog.records) == 1
        assert caplog.messages[0].endswith('test_cmdutils.py has no docstring')

    def test_no_docstring_with_optimize(self, monkeypatch):
        if False:
            return 10
        "With -OO we'd get a warning on start, but no warning afterwards."
        monkeypatch.setattr(sys, 'flags', types.SimpleNamespace(optimize=2))

        @cmdutils.register()
        def fun():
            if False:
                i = 10
                return i + 15
            pass

class TestRun:

    @pytest.fixture(autouse=True)
    def patch_backend(self, mode_manager, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.setattr(command.objects, 'backend', usertypes.Backend.QtWebKit)

    @pytest.mark.parametrize('backend, used, ok', [(usertypes.Backend.QtWebEngine, usertypes.Backend.QtWebEngine, True), (usertypes.Backend.QtWebEngine, usertypes.Backend.QtWebKit, False), (usertypes.Backend.QtWebKit, usertypes.Backend.QtWebEngine, False), (usertypes.Backend.QtWebKit, usertypes.Backend.QtWebKit, True), (None, usertypes.Backend.QtWebEngine, True), (None, usertypes.Backend.QtWebKit, True)])
    def test_backend(self, monkeypatch, backend, used, ok):
        if False:
            print('Hello World!')
        monkeypatch.setattr(command.objects, 'backend', used)
        cmd = _get_cmd(backend=backend)
        if ok:
            cmd.run(win_id=0)
        else:
            with pytest.raises(cmdexc.PrerequisitesError, match='.* backend\\.'):
                cmd.run(win_id=0)

    def test_no_args(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = _get_cmd()
        cmd.run(win_id=0)

    def test_instance_unavailable_with_backend(self, monkeypatch):
        if False:
            return 10
        "Test what happens when a backend doesn't have an objreg object.\n\n        For example, QtWebEngine doesn't have 'hintmanager' registered. We make\n        sure the backend checking happens before resolving the instance, so we\n        display an error instead of crashing.\n        "

        @cmdutils.register(instance='doesnotexist', backend=usertypes.Backend.QtWebEngine)
        def fun(self):
            if False:
                print('Hello World!')
            'Blah.'
        monkeypatch.setattr(command.objects, 'backend', usertypes.Backend.QtWebKit)
        cmd = objects.commands['fun']
        with pytest.raises(cmdexc.PrerequisitesError, match='.* backend\\.'):
            cmd.run(win_id=0)

    def test_deprecated(self, caplog, message_mock):
        if False:
            while True:
                i = 10
        cmd = _get_cmd(deprecated='use something else')
        with caplog.at_level(logging.WARNING):
            cmd.run(win_id=0)
        msg = message_mock.getmsg(usertypes.MessageLevel.warning)
        assert msg.text == 'fun is deprecated - use something else'

    def test_deprecated_name(self, caplog, message_mock):
        if False:
            return 10

        @cmdutils.register(deprecated_name='dep')
        def fun():
            if False:
                return 10
            'Blah.'
        original_cmd = objects.commands['fun']
        original_cmd.run(win_id=0)
        deprecated_cmd = objects.commands['dep']
        with caplog.at_level(logging.WARNING):
            deprecated_cmd.run(win_id=0)
        msg = message_mock.getmsg(usertypes.MessageLevel.warning)
        assert msg.text == 'dep is deprecated - use fun instead'