"""
Provides the basis for all command-line options (flags) in sacred.

It defines the base class CommandLineOption and the standard supported flags.
Some further options that add observers to the run are defined alongside those.
"""
from typing import Callable
import inspect
import re
from sacred.run import Run
from sacred.commands import print_config
from sacred.utils import convert_camel_case_to_snake_case
CLIFunction = Callable[[str, Run], None]

class CLIOption:

    def __init__(self, apply_function: CLIFunction, short_flag: str, long_flag: str, is_flag: bool):
        if False:
            print('Hello World!')
        if not re.match('-\\w$', short_flag):
            raise ValueError('Short flag malformed. One correct short flag would be: `-j`.')
        if not re.match('--[\\w-]+\\w$', long_flag):
            raise ValueError('Long flag malformed. One correct long flag would be: `--my-pretty-flag`')
        self.apply_function = apply_function
        self.short_flag = short_flag
        self.long_flag = long_flag
        self.is_flag = is_flag
        self.arg = None if is_flag else 'VALUE'
        self.arg_description = None if is_flag else ''

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        return self.apply_function(*args, **kwargs)

    def get_flag(self):
        if False:
            return 10
        'Legacy function. Should be removed at some point.'
        return self.long_flag

    def get_short_flag(self):
        if False:
            while True:
                i = 10
        'Legacy function. Should be removed at some point.'
        return self.short_flag

    def get_flags(self):
        if False:
            while True:
                i = 10
        'Legacy function. Should be removed at some point.'
        return (self.short_flag, self.long_flag)

    def apply(self, args, run):
        if False:
            for i in range(10):
                print('nop')
        'Legacy function. Should be removed at some point.'
        return self.apply_function(args, run)

    def get_name(self):
        if False:
            while True:
                i = 10
        return self.apply_function.__name__

    def get_description(self):
        if False:
            while True:
                i = 10
        return inspect.getdoc(self.apply_function) or ''

def cli_option(short_flag: str, long_flag: str, is_flag=False):
    if False:
        while True:
            i = 10

    def wrapper(f: CLIFunction):
        if False:
            print('Hello World!')
        return CLIOption(f, short_flag, long_flag, is_flag)
    return wrapper

class CommandLineOption:
    """
    Base class for all command-line options.

    To implement a new command-line option just inherit from this class.
    Then add the `flag` class-attribute to specify the name and a class
    docstring with the description.
    If your command-line option should take an argument you must also provide
    its name via the `arg` class attribute and its description as
    `arg_description`.
    Finally you need to implement the `execute` classmethod. It receives the
    value of the argument (if applicable) and the current run. You can modify
    the run object in any way.

    If the command line option depends on one or more installed packages, those
    should be imported in the `apply` method to get a proper ImportError
    if the packages are not available.
    """
    _enabled = True
    short_flag = None
    ' The (one-letter) short form (defaults to first letter of flag) '
    arg = None
    ' Name of the argument (optional) '
    arg_description = None
    ' Description of the argument (optional) '

    @classmethod
    def get_flag(cls):
        if False:
            while True:
                i = 10
        flag = cls.__name__
        if flag.endswith('Option'):
            flag = flag[:-6]
        return '--' + convert_camel_case_to_snake_case(flag)

    @classmethod
    def get_short_flag(cls):
        if False:
            while True:
                i = 10
        if cls.short_flag is None:
            return '-' + cls.get_flag()[2]
        else:
            return '-' + cls.short_flag

    @classmethod
    def get_flags(cls):
        if False:
            while True:
                i = 10
        "\n        Return the short and the long version of this option.\n\n        The long flag (e.g. '--foo_bar'; used on the command-line like this:\n        --foo_bar[=ARGS]) is derived from the class-name by stripping away any\n        -Option suffix and converting the rest to snake_case.\n\n        The short flag (e.g. '-f'; used on the command-line like this:\n        -f [ARGS]) the short_flag class-member if that is set, or the first\n        letter of the long flag otherwise.\n\n        Returns\n        -------\n        (str, str)\n            tuple of short-flag, and long-flag\n\n        "
        return (cls.get_short_flag(), cls.get_flag())

    @classmethod
    def apply(cls, args, run):
        if False:
            i = 10
            return i + 15
        '\n        Modify the current Run base on this command-line option.\n\n        This function is executed after constructing the Run object, but\n        before actually starting it.\n\n        Parameters\n        ----------\n        args : bool | str\n            If this command-line option accepts an argument this will be value\n            of that argument if set or None.\n            Otherwise it is either True or False.\n        run :  sacred.run.Run\n            The current run to be modified\n\n        '
        pass

def get_name(option):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(option, CLIOption):
        return option.get_name()
    else:
        return option.__name__

@cli_option('-h', '--help', is_flag=True)
def help_option(args, run):
    if False:
        return 10
    'Print this help message and exit.'
    pass

@cli_option('-d', '--debug', is_flag=True)
def debug_option(args, run):
    if False:
        return 10
    "\n    Set this run to debug mode.\n\n    Suppress warnings about missing observers and don't filter the stacktrace.\n    Also enables usage with ipython `--pdb`.\n    "
    run.debug = True

@cli_option('-D', '--pdb', is_flag=True)
def pdb_option(args, run):
    if False:
        i = 10
        return i + 15
    'Automatically enter post-mortem debugging with pdb on failure.'
    run.pdb = True

@cli_option('-l', '--loglevel')
def loglevel_option(args, run):
    if False:
        return 10
    '\n    Set the LogLevel.\n\n    Loglevel either as 0 - 50 or as string: DEBUG(10),\n    INFO(20), WARNING(30), ERROR(40), CRITICAL(50)\n    '
    try:
        lvl = int(args)
    except ValueError:
        lvl = args
    run.root_logger.setLevel(lvl)

@cli_option('-c', '--comment')
def comment_option(args, run):
    if False:
        print('Hello World!')
    'Add a comment to this run.'
    run.meta_info['comment'] = args

@cli_option('-b', '--beat-interval')
def beat_interval_option(args, run):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the heart-beat interval for this run.\n\n    Time between two heartbeat events is measured in seconds.\n    '
    run.beat_interval = float(args)

@cli_option('-u', '--unobserved', is_flag=True)
def unobserved_option(args, run):
    if False:
        while True:
            i = 10
    'Ignore all observers for this run.'
    run.unobserved = True

@cli_option('-q', '--queue', is_flag=True)
def queue_option(args, run):
    if False:
        while True:
            i = 10
    'Only queue this run, do not start it.'
    run.queue_only = True

@cli_option('-f', '--force', is_flag=True)
def force_option(args, run):
    if False:
        print('Hello World!')
    'Disable warnings about suspicious changes for this run.'
    run.force = True

@cli_option('-P', '--priority')
def priority_option(args, run):
    if False:
        print('Hello World!')
    'Sets the priority for a queued up experiment.\n\n    `--priority=NUMBER`\n    The number represent the priority for this run.\n    '
    try:
        priority = float(args)
    except ValueError as e:
        raise ValueError("The PRIORITY argument must be a number! (but was '{}')".format(args)) from e
    run.meta_info['priority'] = priority

@cli_option('-e', '--enforce_clean', is_flag=True)
def enforce_clean_option(args, run):
    if False:
        print('Hello World!')
    'Fail if any version control repository is dirty.'
    repos = run.experiment_info['repositories']
    if not repos:
        raise RuntimeError('No version control detected. Cannot enforce clean repository.\nMake sure that your sources under VCS and the corresponding python package is installed.')
    else:
        for repo in repos:
            if repo['dirty']:
                raise RuntimeError('EnforceClean: Uncommited changes in the "{}" repository.'.format(repo))

@cli_option('-p', '--print-config', is_flag=True)
def print_config_option(args, run):
    if False:
        i = 10
        return i + 15
    'Always print the configuration first.'
    print_config(run)
    print('-' * 79)

@cli_option('-i', '--id')
def id_option(args, run):
    if False:
        while True:
            i = 10
    'Set the id for this run.'
    run._id = args

@cli_option('-n', '--name')
def name_option(args, run):
    if False:
        while True:
            i = 10
    'Set the name for this run.'
    run.experiment_info['name'] = args
    run.run_logger = run.root_logger.getChild(args)

@cli_option('-C', '--capture')
def capture_option(args, run):
    if False:
        i = 10
        return i + 15
    '\n    Control the way stdout and stderr are captured.\n\n    The argument value must be one of [no, sys, fd]\n    '
    run.capture_mode = args