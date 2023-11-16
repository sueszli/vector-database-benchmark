"""
Here is a minimal example of usage:
.. code-block:: python
    >>> from openquake.baselib import sap
    >>> def fun(input, inplace, output=None, out='/tmp'):
    ...     'Example'
    ...     for item in sorted(locals().items()):
    ...         print('%s = %s' % item)
    >>> p = sap.script(fun)
    >>> p.arg('input', 'input file or archive')
    >>> p.flg('inplace', 'convert inplace')
    >>> p.arg('output', 'output archive')
    >>> p.opt('out', 'optional output file')
    >>> p.callfunc(['a'])
    inplace = False
    input = a
    out = /tmp
    output = None
    >>> p.callfunc(['a', 'b', '-i', '-o', 'OUT'])
    inplace = True
    input = a
    out = OUT
    output = b
Parsers can be composed too.
"""
import sys
import inspect
import argparse
NODEFAULT = object()
registry = {}

def get_parentparser(parser, description=None, help=True):
    if False:
        while True:
            i = 10
    '\n    :param parser: :class:`argparse.ArgumentParser` instance or None\n    :param description: string used to build a new parser if parser is None\n    :param help: flag used to build a new parser if parser is None\n    :returns: if parser is None the new parser; otherwise the `.parentparser`\n              attribute (if set) or the parser itself (if not set)\n    '
    if parser is None:
        return argparse.ArgumentParser(description=description, add_help=help)
    elif hasattr(parser, 'parentparser'):
        return parser.parentparser
    else:
        return parser

def str_choices(choices):
    if False:
        return 10
    'Returns {choice1, ..., choiceN} or the empty string'
    if choices:
        return '{%s}' % ', '.join(choices)
    return ''

class Script(object):
    """
    A simple way to define command processors based on argparse.
    Each parser is associated to a function and parsers can be
    composed together, by dispatching on a given name (if not given,
    the function name is used).
    """

    def __init__(self, func, name=None, parentparser=None, help=True):
        if False:
            return 10
        self.func = func
        self.name = name or func.__name__
        (args, self.varargs, varkw, defaults) = inspect.getfullargspec(func)[:4]
        assert self.varargs is None, self.varargs
        defaults = defaults or ()
        nodefaults = len(args) - len(defaults)
        alldefaults = (NODEFAULT,) * nodefaults + defaults
        self.argdict = dict(zip(args, alldefaults))
        self.description = descr = func.__doc__ if func.__doc__ else None
        self.parentparser = get_parentparser(parentparser, descr, help)
        self.names = []
        self.all_arguments = []
        self._group = self.parentparser
        self._argno = 0
        self.checked = False
        registry['%s.%s' % (func.__module__, func.__name__)] = self

    def group(self, descr):
        if False:
            return 10
        'Added a new group of arguments with the given description'
        self._group = self.parentparser.add_argument_group(descr)

    def _add(self, name, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add an argument to the underlying parser and grow the list\n        .all_arguments and the set .names\n        '
        argname = list(self.argdict)[self._argno]
        if argname != name:
            raise NameError('Setting argument %s, but it should be %s' % (name, argname))
        self._group.add_argument(*args, **kw)
        self.all_arguments.append((args, kw))
        self.names.append(name)
        self._argno += 1

    def arg(self, name, help, type=None, choices=None, metavar=None, nargs=None):
        if False:
            return 10
        'Describe a positional argument'
        kw = dict(help=help, type=type, choices=choices, metavar=metavar, nargs=nargs)
        default = self.argdict[name]
        if default is not NODEFAULT:
            kw['nargs'] = nargs or '?'
            kw['default'] = default
            kw['help'] = kw['help'] + ' [default: %s]' % repr(default)
        self._add(name, name, **kw)

    def opt(self, name, help, abbrev=None, type=None, choices=None, metavar=None, nargs=None):
        if False:
            while True:
                i = 10
        'Describe an option'
        kw = dict(help=help, type=type, choices=choices, metavar=metavar, nargs=nargs)
        default = self.argdict[name]
        if default is not NODEFAULT:
            kw['default'] = default
            kw['metavar'] = metavar or str_choices(choices) or str(default)
        abbrev = abbrev or '-' + name[0]
        abbrevs = set((args[0] for (args, kw) in self.all_arguments))
        longname = '--' + name.replace('_', '-')
        if abbrev == '-h' or abbrev in abbrevs:
            self._add(name, longname, **kw)
        else:
            self._add(name, abbrev, longname, **kw)

    def flg(self, name, help, abbrev=None):
        if False:
            print('Hello World!')
        'Describe a flag'
        abbrev = abbrev or '-' + name[0]
        longname = '--' + name.replace('_', '-')
        self._add(name, abbrev, longname, action='store_true', help=help)

    def check_arguments(self):
        if False:
            return 10
        'Make sure all arguments have a specification'
        for (name, default) in self.argdict.items():
            if name not in self.names and default is NODEFAULT:
                raise NameError('Missing argparse specification for %r' % name)

    def callfunc(self, argv=None):
        if False:
            i = 10
            return i + 15
        '\n        Parse the argv list and extract a dictionary of arguments which\n        is then passed to  the function underlying the script.\n        '
        if not self.checked:
            self.check_arguments()
            self.checked = True
        namespace = self.parentparser.parse_args(argv or sys.argv[1:])
        return self.func(**vars(namespace))

    def help(self):
        if False:
            print('Hello World!')
        '\n        Return the help message as a string\n        '
        return self.parentparser.format_help()

    def __repr__(self):
        if False:
            while True:
                i = 10
        args = ', '.join(self.names)
        return '<%s %s(%s)>' % (self.__class__.__name__, self.name, args)

def script(func):
    if False:
        for i in range(10):
            print('nop')
    s = Script(func)
    func.arg = s.arg
    func.opt = s.opt
    func.flg = s.flg
    func.group = s.group
    func._add = s._add
    func.callfunc = s.callfunc
    return func

def compose(scripts, name='main', description=None, prog=None, version=None):
    if False:
        print('Hello World!')
    '\n    Collects together different scripts and builds a single\n    script dispatching to the subparsers depending on\n    the first argument, i.e. the name of the subparser to invoke.\n    :param scripts: a list of script instances\n    :param name: the name of the composed parser\n    :param description: description of the composed parser\n    :param prog: name of the script printed in the usage message\n    :param version: version of the script printed with --version\n    '
    assert len(scripts) >= 1, scripts
    parentparser = argparse.ArgumentParser(description=description, add_help=False)
    parentparser.add_argument('--version', '-v', action='version', version=version)
    subparsers = parentparser.add_subparsers(help='available subcommands; use %s help <subcmd>' % prog, prog=prog)

    def gethelp(cmd=None):
        if False:
            print('Hello World!')
        if cmd is None:
            print(parentparser.format_help())
            return
        subp = subparsers._name_parser_map.get(cmd)
        if subp is None:
            print('No help for unknown command %r' % cmd)
        else:
            print(subp.format_help())
    help_script = Script(gethelp, 'help', help=False)
    progname = '%s ' % prog if prog else ''
    help_script.arg('cmd', progname + 'subcommand')
    for s in list(scripts) + [help_script]:
        subp = subparsers.add_parser(s.name, description=s.description)
        for (args, kw) in s.all_arguments:
            subp.add_argument(*args, **kw)
        subp.set_defaults(_func=s.func)

    def main(**kw):
        if False:
            for i in range(10):
                print('nop')
        try:
            func = kw.pop('_func')
        except KeyError:
            parentparser.print_usage()
        else:
            return func(**kw)
    main.__name__ = name
    return Script(main, name, parentparser)