import argparse
import os
import sys
import warnings
from gettext import gettext
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import final
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import _pytest._io
from _pytest.config.exceptions import UsageError
from _pytest.deprecated import ARGUMENT_PERCENT_DEFAULT
from _pytest.deprecated import ARGUMENT_TYPE_STR
from _pytest.deprecated import ARGUMENT_TYPE_STR_CHOICE
from _pytest.deprecated import check_ispytest
FILE_OR_DIR = 'file_or_dir'

class NotSet:

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '<notset>'
NOT_SET = NotSet()

@final
class Parser:
    """Parser for command line arguments and ini-file values.

    :ivar extra_info: Dict of generic param -> value to display in case
        there's an error processing the command line arguments.
    """
    prog: Optional[str] = None

    def __init__(self, usage: Optional[str]=None, processopt: Optional[Callable[['Argument'], None]]=None, *, _ispytest: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        check_ispytest(_ispytest)
        self._anonymous = OptionGroup('Custom options', parser=self, _ispytest=True)
        self._groups: List[OptionGroup] = []
        self._processopt = processopt
        self._usage = usage
        self._inidict: Dict[str, Tuple[str, Optional[str], Any]] = {}
        self._ininames: List[str] = []
        self.extra_info: Dict[str, Any] = {}

    def processoption(self, option: 'Argument') -> None:
        if False:
            return 10
        if self._processopt:
            if option.dest:
                self._processopt(option)

    def getgroup(self, name: str, description: str='', after: Optional[str]=None) -> 'OptionGroup':
        if False:
            i = 10
            return i + 15
        'Get (or create) a named option Group.\n\n        :param name: Name of the option group.\n        :param description: Long description for --help output.\n        :param after: Name of another group, used for ordering --help output.\n        :returns: The option group.\n\n        The returned group object has an ``addoption`` method with the same\n        signature as :func:`parser.addoption <pytest.Parser.addoption>` but\n        will be shown in the respective group in the output of\n        ``pytest --help``.\n        '
        for group in self._groups:
            if group.name == name:
                return group
        group = OptionGroup(name, description, parser=self, _ispytest=True)
        i = 0
        for (i, grp) in enumerate(self._groups):
            if grp.name == after:
                break
        self._groups.insert(i + 1, group)
        return group

    def addoption(self, *opts: str, **attrs: Any) -> None:
        if False:
            print('Hello World!')
        'Register a command line option.\n\n        :param opts:\n            Option names, can be short or long options.\n        :param attrs:\n            Same attributes as the argparse library\'s :py:func:`add_argument()\n            <argparse.ArgumentParser.add_argument>` function accepts.\n\n        After command line parsing, options are available on the pytest config\n        object via ``config.option.NAME`` where ``NAME`` is usually set\n        by passing a ``dest`` attribute, for example\n        ``addoption("--long", dest="NAME", ...)``.\n        '
        self._anonymous.addoption(*opts, **attrs)

    def parse(self, args: Sequence[Union[str, 'os.PathLike[str]']], namespace: Optional[argparse.Namespace]=None) -> argparse.Namespace:
        if False:
            return 10
        from _pytest._argcomplete import try_argcomplete
        self.optparser = self._getparser()
        try_argcomplete(self.optparser)
        strargs = [os.fspath(x) for x in args]
        return self.optparser.parse_args(strargs, namespace=namespace)

    def _getparser(self) -> 'MyOptionParser':
        if False:
            while True:
                i = 10
        from _pytest._argcomplete import filescompleter
        optparser = MyOptionParser(self, self.extra_info, prog=self.prog)
        groups = self._groups + [self._anonymous]
        for group in groups:
            if group.options:
                desc = group.description or group.name
                arggroup = optparser.add_argument_group(desc)
                for option in group.options:
                    n = option.names()
                    a = option.attrs()
                    arggroup.add_argument(*n, **a)
        file_or_dir_arg = optparser.add_argument(FILE_OR_DIR, nargs='*')
        file_or_dir_arg.completer = filescompleter
        return optparser

    def parse_setoption(self, args: Sequence[Union[str, 'os.PathLike[str]']], option: argparse.Namespace, namespace: Optional[argparse.Namespace]=None) -> List[str]:
        if False:
            i = 10
            return i + 15
        parsedoption = self.parse(args, namespace=namespace)
        for (name, value) in parsedoption.__dict__.items():
            setattr(option, name, value)
        return cast(List[str], getattr(parsedoption, FILE_OR_DIR))

    def parse_known_args(self, args: Sequence[Union[str, 'os.PathLike[str]']], namespace: Optional[argparse.Namespace]=None) -> argparse.Namespace:
        if False:
            while True:
                i = 10
        'Parse the known arguments at this point.\n\n        :returns: An argparse namespace object.\n        '
        return self.parse_known_and_unknown_args(args, namespace=namespace)[0]

    def parse_known_and_unknown_args(self, args: Sequence[Union[str, 'os.PathLike[str]']], namespace: Optional[argparse.Namespace]=None) -> Tuple[argparse.Namespace, List[str]]:
        if False:
            for i in range(10):
                print('nop')
        'Parse the known arguments at this point, and also return the\n        remaining unknown arguments.\n\n        :returns:\n            A tuple containing an argparse namespace object for the known\n            arguments, and a list of the unknown arguments.\n        '
        optparser = self._getparser()
        strargs = [os.fspath(x) for x in args]
        return optparser.parse_known_args(strargs, namespace=namespace)

    def addini(self, name: str, help: str, type: Optional[Literal['string', 'paths', 'pathlist', 'args', 'linelist', 'bool']]=None, default: Any=NOT_SET) -> None:
        if False:
            while True:
                i = 10
        'Register an ini-file option.\n\n        :param name:\n            Name of the ini-variable.\n        :param type:\n            Type of the variable. Can be:\n\n                * ``string``: a string\n                * ``bool``: a boolean\n                * ``args``: a list of strings, separated as in a shell\n                * ``linelist``: a list of strings, separated by line breaks\n                * ``paths``: a list of :class:`pathlib.Path`, separated as in a shell\n                * ``pathlist``: a list of ``py.path``, separated as in a shell\n\n            .. versionadded:: 7.0\n                The ``paths`` variable type.\n\n            Defaults to ``string`` if ``None`` or not passed.\n        :param default:\n            Default value if no ini-file option exists but is queried.\n\n        The value of ini-variables can be retrieved via a call to\n        :py:func:`config.getini(name) <pytest.Config.getini>`.\n        '
        assert type in (None, 'string', 'paths', 'pathlist', 'args', 'linelist', 'bool')
        if default is NOT_SET:
            default = get_ini_default_for_type(type)
        self._inidict[name] = (help, type, default)
        self._ininames.append(name)

def get_ini_default_for_type(type: Optional[Literal['string', 'paths', 'pathlist', 'args', 'linelist', 'bool']]) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\n    Used by addini to get the default value for a given ini-option type, when\n    default is not supplied.\n    '
    if type is None:
        return ''
    elif type in ('paths', 'pathlist', 'args', 'linelist'):
        return []
    elif type == 'bool':
        return False
    else:
        return ''

class ArgumentError(Exception):
    """Raised if an Argument instance is created with invalid or
    inconsistent arguments."""

    def __init__(self, msg: str, option: Union['Argument', str]) -> None:
        if False:
            i = 10
            return i + 15
        self.msg = msg
        self.option_id = str(option)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        if self.option_id:
            return f'option {self.option_id}: {self.msg}'
        else:
            return self.msg

class Argument:
    """Class that mimics the necessary behaviour of optparse.Option.

    It's currently a least effort implementation and ignoring choices
    and integer prefixes.

    https://docs.python.org/3/library/optparse.html#optparse-standard-option-types
    """
    _typ_map = {'int': int, 'string': str, 'float': float, 'complex': complex}

    def __init__(self, *names: str, **attrs: Any) -> None:
        if False:
            print('Hello World!')
        'Store params in private vars for use in add_argument.'
        self._attrs = attrs
        self._short_opts: List[str] = []
        self._long_opts: List[str] = []
        if '%default' in (attrs.get('help') or ''):
            warnings.warn(ARGUMENT_PERCENT_DEFAULT, stacklevel=3)
        try:
            typ = attrs['type']
        except KeyError:
            pass
        else:
            if isinstance(typ, str):
                if typ == 'choice':
                    warnings.warn(ARGUMENT_TYPE_STR_CHOICE.format(typ=typ, names=names), stacklevel=4)
                    attrs['type'] = type(attrs['choices'][0])
                else:
                    warnings.warn(ARGUMENT_TYPE_STR.format(typ=typ, names=names), stacklevel=4)
                    attrs['type'] = Argument._typ_map[typ]
                self.type = attrs['type']
            else:
                self.type = typ
        try:
            self.default = attrs['default']
        except KeyError:
            pass
        self._set_opt_strings(names)
        dest: Optional[str] = attrs.get('dest')
        if dest:
            self.dest = dest
        elif self._long_opts:
            self.dest = self._long_opts[0][2:].replace('-', '_')
        else:
            try:
                self.dest = self._short_opts[0][1:]
            except IndexError as e:
                self.dest = '???'
                raise ArgumentError('need a long or short option', self) from e

    def names(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self._short_opts + self._long_opts

    def attrs(self) -> Mapping[str, Any]:
        if False:
            return 10
        attrs = 'default dest help'.split()
        attrs.append(self.dest)
        for attr in attrs:
            try:
                self._attrs[attr] = getattr(self, attr)
            except AttributeError:
                pass
        if self._attrs.get('help'):
            a = self._attrs['help']
            a = a.replace('%default', '%(default)s')
            self._attrs['help'] = a
        return self._attrs

    def _set_opt_strings(self, opts: Sequence[str]) -> None:
        if False:
            while True:
                i = 10
        'Directly from optparse.\n\n        Might not be necessary as this is passed to argparse later on.\n        '
        for opt in opts:
            if len(opt) < 2:
                raise ArgumentError('invalid option string %r: must be at least two characters long' % opt, self)
            elif len(opt) == 2:
                if not (opt[0] == '-' and opt[1] != '-'):
                    raise ArgumentError('invalid short option string %r: must be of the form -x, (x any non-dash char)' % opt, self)
                self._short_opts.append(opt)
            else:
                if not (opt[0:2] == '--' and opt[2] != '-'):
                    raise ArgumentError('invalid long option string %r: must start with --, followed by non-dash' % opt, self)
                self._long_opts.append(opt)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        args: List[str] = []
        if self._short_opts:
            args += ['_short_opts: ' + repr(self._short_opts)]
        if self._long_opts:
            args += ['_long_opts: ' + repr(self._long_opts)]
        args += ['dest: ' + repr(self.dest)]
        if hasattr(self, 'type'):
            args += ['type: ' + repr(self.type)]
        if hasattr(self, 'default'):
            args += ['default: ' + repr(self.default)]
        return 'Argument({})'.format(', '.join(args))

class OptionGroup:
    """A group of options shown in its own section."""

    def __init__(self, name: str, description: str='', parser: Optional[Parser]=None, *, _ispytest: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        check_ispytest(_ispytest)
        self.name = name
        self.description = description
        self.options: List[Argument] = []
        self.parser = parser

    def addoption(self, *opts: str, **attrs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Add an option to this group.\n\n        If a shortened version of a long option is specified, it will\n        be suppressed in the help. ``addoption('--twowords', '--two-words')``\n        results in help showing ``--two-words`` only, but ``--twowords`` gets\n        accepted **and** the automatic destination is in ``args.twowords``.\n\n        :param opts:\n            Option names, can be short or long options.\n        :param attrs:\n            Same attributes as the argparse library's :py:func:`add_argument()\n            <argparse.ArgumentParser.add_argument>` function accepts.\n        "
        conflict = set(opts).intersection((name for opt in self.options for name in opt.names()))
        if conflict:
            raise ValueError('option names %s already added' % conflict)
        option = Argument(*opts, **attrs)
        self._addoption_instance(option, shortupper=False)

    def _addoption(self, *opts: str, **attrs: Any) -> None:
        if False:
            while True:
                i = 10
        option = Argument(*opts, **attrs)
        self._addoption_instance(option, shortupper=True)

    def _addoption_instance(self, option: 'Argument', shortupper: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not shortupper:
            for opt in option._short_opts:
                if opt[0] == '-' and opt[1].islower():
                    raise ValueError('lowercase shortoptions reserved')
        if self.parser:
            self.parser.processoption(option)
        self.options.append(option)

class MyOptionParser(argparse.ArgumentParser):

    def __init__(self, parser: Parser, extra_info: Optional[Dict[str, Any]]=None, prog: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._parser = parser
        super().__init__(prog=prog, usage=parser._usage, add_help=False, formatter_class=DropShorterLongHelpFormatter, allow_abbrev=False)
        self.extra_info = extra_info if extra_info else {}

    def error(self, message: str) -> NoReturn:
        if False:
            return 10
        'Transform argparse error message into UsageError.'
        msg = f'{self.prog}: error: {message}'
        if hasattr(self._parser, '_config_source_hint'):
            msg = f'{msg} ({self._parser._config_source_hint})'
        raise UsageError(self.format_usage() + msg)

    def parse_args(self, args: Optional[Sequence[str]]=None, namespace: Optional[argparse.Namespace]=None) -> argparse.Namespace:
        if False:
            i = 10
            return i + 15
        'Allow splitting of positional arguments.'
        (parsed, unrecognized) = self.parse_known_args(args, namespace)
        if unrecognized:
            for arg in unrecognized:
                if arg and arg[0] == '-':
                    lines = ['unrecognized arguments: %s' % ' '.join(unrecognized)]
                    for (k, v) in sorted(self.extra_info.items()):
                        lines.append(f'  {k}: {v}')
                    self.error('\n'.join(lines))
            getattr(parsed, FILE_OR_DIR).extend(unrecognized)
        return parsed
    if sys.version_info[:2] < (3, 9):

        def _parse_optional(self, arg_string: str) -> Optional[Tuple[Optional[argparse.Action], str, Optional[str]]]:
            if False:
                i = 10
                return i + 15
            if not arg_string:
                return None
            if not arg_string[0] in self.prefix_chars:
                return None
            if arg_string in self._option_string_actions:
                action = self._option_string_actions[arg_string]
                return (action, arg_string, None)
            if len(arg_string) == 1:
                return None
            if '=' in arg_string:
                (option_string, explicit_arg) = arg_string.split('=', 1)
                if option_string in self._option_string_actions:
                    action = self._option_string_actions[option_string]
                    return (action, option_string, explicit_arg)
            if self.allow_abbrev or not arg_string.startswith('--'):
                option_tuples = self._get_option_tuples(arg_string)
                if len(option_tuples) > 1:
                    msg = gettext('ambiguous option: %(option)s could match %(matches)s')
                    options = ', '.join((option for (_, option, _) in option_tuples))
                    self.error(msg % {'option': arg_string, 'matches': options})
                elif len(option_tuples) == 1:
                    (option_tuple,) = option_tuples
                    return option_tuple
            if self._negative_number_matcher.match(arg_string):
                if not self._has_negative_number_optionals:
                    return None
            if ' ' in arg_string:
                return None
            return (None, arg_string, None)

class DropShorterLongHelpFormatter(argparse.HelpFormatter):
    """Shorten help for long options that differ only in extra hyphens.

    - Collapse **long** options that are the same except for extra hyphens.
    - Shortcut if there are only two options and one of them is a short one.
    - Cache result on the action object as this is called at least 2 times.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        if 'width' not in kwargs:
            kwargs['width'] = _pytest._io.get_terminal_width()
        super().__init__(*args, **kwargs)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if False:
            print('Hello World!')
        orgstr = super()._format_action_invocation(action)
        if orgstr and orgstr[0] != '-':
            return orgstr
        res: Optional[str] = getattr(action, '_formatted_action_invocation', None)
        if res:
            return res
        options = orgstr.split(', ')
        if len(options) == 2 and (len(options[0]) == 2 or len(options[1]) == 2):
            action._formatted_action_invocation = orgstr
            return orgstr
        return_list = []
        short_long: Dict[str, str] = {}
        for option in options:
            if len(option) == 2 or option[2] == ' ':
                continue
            if not option.startswith('--'):
                raise ArgumentError('long optional argument without "--": [%s]' % option, option)
            xxoption = option[2:]
            shortened = xxoption.replace('-', '')
            if shortened not in short_long or len(short_long[shortened]) < len(xxoption):
                short_long[shortened] = xxoption
        for option in options:
            if len(option) == 2 or option[2] == ' ':
                return_list.append(option)
            if option[2:] == short_long.get(option.replace('-', '')):
                return_list.append(option.replace(' ', '=', 1))
        formatted_action_invocation = ', '.join(return_list)
        action._formatted_action_invocation = formatted_action_invocation
        return formatted_action_invocation

    def _split_lines(self, text, width):
        if False:
            for i in range(10):
                print('nop')
        'Wrap lines after splitting on original newlines.\n\n        This allows to have explicit line breaks in the help text.\n        '
        import textwrap
        lines = []
        for line in text.splitlines():
            lines.extend(textwrap.wrap(line.strip(), width))
        return lines