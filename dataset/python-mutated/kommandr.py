"""
Simple module helps you convert your ordinary functions into cool
command-line interfaces using :py:module:``argparse`` in backyard.
"""
import sys
import inspect
import argparse
try:
    from itertools import izip_longest
except ImportError:
    from itertools import zip_longest as izip_longest
import collections
import six
from .parsers.storm_config_parser import get_storm_config
from . import __version__

class AliasedSubParsersAction(argparse._SubParsersAction):

    class _AliasedPseudoAction(argparse.Action):

        def __init__(self, name, aliases, help):
            if False:
                i = 10
                return i + 15
            dest = name
            if aliases:
                dest += ' (%s)' % '-'.join(aliases)
            sup = super(AliasedSubParsersAction._AliasedPseudoAction, self)
            sup.__init__(option_strings=[], dest=dest, help=help)

    def add_parser(self, name, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'aliases' in kwargs:
            aliases = kwargs['aliases']
            del kwargs['aliases']
        else:
            aliases = []
        parser = super(AliasedSubParsersAction, self).add_parser(name, **kwargs)
        if aliases:
            for alias in aliases:
                self._name_parser_map[alias] = parser
        if 'help' in kwargs:
            help = kwargs.pop('help')
            self._choices_actions.pop()
            pseudo_action = self._AliasedPseudoAction(name, aliases, help)
            self._choices_actions.append(pseudo_action)
        return parser

class prog(object):
    """Class to hold an isolated command namespace"""
    _COMMAND_FLAG = '_command'
    _POSITIONAL = type('_positional', (object,), {})

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        'Constructor\n\n        :param version: program version\n        :param type: str\n\n        :param **kwargs: keyword arguments those passed through to\n                         :py:class:``argparse.ArgumentParser`` constructor\n        :param type: dict\n\n        '
        kwargs.update({'formatter_class': argparse.RawTextHelpFormatter, 'epilog': 'storm is a command line tool to manage ssh connections.\nget more information at: github.com/emre/storm'})
        self.parser = argparse.ArgumentParser(**kwargs)
        self.parser.register('action', 'parsers', AliasedSubParsersAction)
        self.parser.formatter_class.width = 300
        self.parser.add_argument('-v', '--version', action='version', version=__version__)
        self.subparsers = self.parser.add_subparsers(title='commands', metavar='COMMAND')
        self.subparsers.required = True

    def command(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Convenient decorator simply creates corresponding command'
        if len(args) == 1 and isinstance(args[0], collections.Callable):
            return self._generate_command(args[0])
        else:

            def _command(func):
                if False:
                    return 10
                return self._generate_command(func, *args, **kwargs)
            return _command

    def arg(self, arg_name, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Decorator function configures any arg by given ``arg_name`` with\n        supplied ``args`` and ``kwargs`` passing them transparently to\n        :py:func:``argparse.ArgumentParser.add_argument`` function\n\n        :param arg_name: arg name to configure\n        :param type: str\n\n        '

        def wrapper(func):
            if False:
                i = 10
                return i + 15
            if not getattr(func, 'argopts', None):
                func.argopts = {}
            func.argopts[arg_name] = (args, kwargs)
            return func
        return wrapper

    def _generate_command(self, func, name=None, **kwargs):
        if False:
            return 10
        'Generates a command parser for given func.\n\n        :param func: func to generate related command parser\n        :param type: function\n\n        :param name: command name\n        :param type: str\n\n        :param **kwargs: keyword arguments those passed through to\n                         :py:class:``argparse.ArgumentParser.add_parser``\n        :param type: dict\n\n        '
        func_pointer = name or func.__name__
        storm_config = get_storm_config()
        (aliases, additional_kwarg) = (None, None)
        if 'aliases' in storm_config:
            for (command, alias_list) in six.iteritems(storm_config.get('aliases')):
                if func_pointer == command:
                    aliases = alias_list
                    break
        func_help = func.__doc__ and func.__doc__.strip()
        subparser = self.subparsers.add_parser(name or func.__name__, aliases=aliases, help=func_help)
        spec = inspect.getargspec(func)
        opts = reversed(list(izip_longest(reversed(spec.args or []), reversed(spec.defaults or []), fillvalue=self._POSITIONAL())))
        for (k, v) in opts:
            argopts = getattr(func, 'argopts', {})
            (args, kwargs) = argopts.get(k, ([], {}))
            args = list(args)
            is_positional = isinstance(v, self._POSITIONAL)
            options = [arg for arg in args if arg.startswith('-')]
            if isinstance(v, list):
                kwargs.update({'action': 'append'})
            if is_positional:
                if options:
                    args = options
                    kwargs.update({'required': True, 'dest': k})
                else:
                    args = [k]
            else:
                args = options or ['--%s' % k]
                kwargs.update({'default': v, 'dest': k})
            arg = subparser.add_argument(*args, **kwargs)
        subparser.set_defaults(**{self._COMMAND_FLAG: func})
        return func

    def execute(self, arg_list):
        if False:
            for i in range(10):
                print('nop')
        'Main function to parse and dispatch commands by given ``arg_list``\n\n        :param arg_list: all arguments provided by the command line\n        :param type: list\n\n        '
        arg_map = self.parser.parse_args(arg_list).__dict__
        command = arg_map.pop(self._COMMAND_FLAG)
        return command(**arg_map)

    def __call__(self):
        if False:
            print('Hello World!')
        'Calls :py:func:``execute`` with :py:class:``sys.argv`` excluding\n        script name which comes first.\n\n        '
        self.execute(sys.argv[1:])
main = prog()
arg = main.arg
command = main.command
execute = main.execute