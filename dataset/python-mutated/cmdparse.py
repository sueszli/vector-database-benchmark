import itertools
import re
import shlex
from pipenv.vendor import tomlkit

class ScriptEmptyError(ValueError):
    pass

class ScriptParseError(ValueError):
    pass

def _quote_if_contains(value, pattern):
    if False:
        for i in range(10):
            print('nop')
    if next(iter(re.finditer(pattern, value)), None):
        return '"{}"'.format(re.sub('(\\\\*)"', '\\1\\1\\\\"', value))
    return value

def _parse_toml_inline_table(value: tomlkit.items.InlineTable) -> str:
    if False:
        return 10
    'parses the [scripts] in pipfile and converts: `{call = "package.module:func(\'arg\')"}` into an executable command'
    keys_list = list(value.keys())
    if len(keys_list) > 1:
        raise ScriptParseError('More than 1 key in toml script line')
    cmd_key = keys_list[0]
    if cmd_key not in Script.script_types:
        raise ScriptParseError(f'Not an accepted script callabale, options are: {Script.script_types}')
    if cmd_key == 'call':
        (module, _, func) = str(value['call']).partition(':')
        if not module or not func:
            raise ScriptParseError('Callable must be like: name = {call = "package.module:func(\'arg\')"}')
        if re.search('\\(.*?\\)', func) is None:
            func += '()'
        return f'python -c "import {module} as _m; _m.{func}"'

class Script:
    """Parse a script line (in Pipfile's [scripts] section).

    This always works in POSIX mode, even on Windows.
    """
    script_types = ['call']

    def __init__(self, command, args=None):
        if False:
            print('Hello World!')
        self._parts = [command]
        if args:
            self._parts.extend(args)

    @classmethod
    def parse(cls, value):
        if False:
            i = 10
            return i + 15
        if isinstance(value, tomlkit.items.InlineTable):
            cmd_string = _parse_toml_inline_table(value)
            value = shlex.split(cmd_string)
        elif isinstance(value, str):
            value = shlex.split(value)
        if not value:
            raise ScriptEmptyError(value)
        return cls(value[0], value[1:])

    def __repr__(self):
        if False:
            return 10
        return f'Script({self._parts!r})'

    @property
    def command(self):
        if False:
            while True:
                i = 10
        return self._parts[0]

    @property
    def args(self):
        if False:
            print('Hello World!')
        return self._parts[1:]

    @property
    def cmd_args(self):
        if False:
            for i in range(10):
                print('nop')
        return self._parts

    def extend(self, extra_args):
        if False:
            for i in range(10):
                print('nop')
        self._parts.extend(extra_args)

    def cmdify(self):
        if False:
            for i in range(10):
                print('nop')
        'Encode into a cmd-executable string.\n\n        This re-implements CreateProcess\'s quoting logic to turn a list of\n        arguments into one single string for the shell to interpret.\n\n        * All double quotes are escaped with a backslash.\n        * Existing backslashes before a quote are doubled, so they are all\n          escaped properly.\n        * Backslashes elsewhere are left as-is; cmd will interpret them\n          literally.\n\n        The result is then quoted into a pair of double quotes to be grouped.\n\n        An argument is intentionally not quoted if it does not contain\n        foul characters. This is done to be compatible with Windows built-in\n        commands that don\'t work well with quotes, e.g. everything with `echo`,\n        and DOS-style (forward slash) switches.\n\n        Foul characters include:\n\n        * Whitespaces.\n        * Carets (^). (pypa/pipenv#3307)\n        * Parentheses in the command. (pypa/pipenv#3168)\n\n        Carets introduce a difficult situation since they are essentially\n        "lossy" when parsed. Consider this in cmd.exe::\n\n            > echo "foo^bar"\n            "foo^bar"\n            > echo foo^^bar\n            foo^bar\n\n        The two commands produce different results, but are both parsed by the\n        shell as `foo^bar`, and there\'s essentially no sensible way to tell\n        what was actually passed in. This implementation assumes the quoted\n        variation (the first) since it is easier to implement, and arguably\n        the more common case.\n\n        The intended use of this function is to pre-process an argument list\n        before passing it into ``subprocess.Popen(..., shell=True)``.\n\n        See also: https://docs.python.org/3/library/subprocess.html#converting-argument-sequence\n        '
        return ' '.join(itertools.chain([_quote_if_contains(self.command, '[\\s^()]')], (_quote_if_contains(arg, '[\\s^]') for arg in self.args)))