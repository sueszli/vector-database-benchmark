"""The Experiment class, which is central to sacred."""
import inspect
import os.path
import sys
import warnings
from collections import OrderedDict
from typing import Sequence, Optional, List
from docopt import docopt, printable_usage
from sacred import SETTINGS
from sacred.arg_parser import format_usage, get_config_updates
from sacred import commandline_options
from sacred.commandline_options import CLIOption
from sacred.commands import help_for_command, print_config, print_dependencies, save_config, print_named_configs
from sacred.observers.file_storage import file_storage_option
from sacred.observers.s3_observer import s3_option
from sacred.config.signature import Signature
from sacred.ingredient import Ingredient
from sacred.initialize import create_run
from sacred.observers.sql import sql_option
from sacred.observers.tinydb_hashfs import tiny_db_option
from sacred.run import Run
from sacred.host_info import check_additional_host_info, HostInfoGetter
from sacred.utils import print_filtered_stacktrace, ensure_wellformed_argv, SacredError, format_sacred_error, PathType, get_inheritors
from sacred.observers.mongo import mongo_db_option
__all__ = ('Experiment',)

class Experiment(Ingredient):
    """
    The central class for each experiment in Sacred.

    It manages the configuration, the main function, captured methods,
    observers, commands, and further ingredients.

    An Experiment instance should be created as one of the first
    things in any experiment-file.
    """

    def __init__(self, name: Optional[str]=None, ingredients: Sequence[Ingredient]=(), interactive: bool=False, base_dir: Optional[PathType]=None, additional_host_info: Optional[List[HostInfoGetter]]=None, additional_cli_options: Optional[Sequence[CLIOption]]=None, save_git_info: bool=True):
        if False:
            return 10
        "\n        Create a new experiment with the given name and optional ingredients.\n\n        Parameters\n        ----------\n        name\n            Optional name of this experiment, defaults to the filename.\n            (Required in interactive mode)\n\n        ingredients : list[sacred.Ingredient], optional\n            A list of ingredients to be used with this experiment.\n\n        interactive\n            If set to True will allow the experiment to be run in interactive\n            mode (e.g. IPython or Jupyter notebooks).\n            However, this mode is discouraged since it won't allow storing the\n            source-code or reliable reproduction of the runs.\n\n        base_dir\n            Optional full path to the base directory of this experiment. This\n            will set the scope for automatic source file discovery.\n\n        additional_host_info\n            Optional dictionary containing as keys the names of the pieces of\n            host info you want to collect, and as\n            values the functions collecting those pieces of information.\n\n        save_git_info:\n            Optionally save the git commit hash and the git state\n            (clean or dirty) for all source files. This requires the GitPython\n            package.\n        "
        self.additional_host_info = additional_host_info or []
        check_additional_host_info(self.additional_host_info)
        self.additional_cli_options = additional_cli_options or []
        self.all_cli_options = gather_command_line_options() + self.additional_cli_options
        caller_globals = inspect.stack()[1][0].f_globals
        if name is None:
            if interactive:
                raise RuntimeError('name is required in interactive mode.')
            mainfile = caller_globals.get('__file__')
            if mainfile is None:
                raise RuntimeError('No main-file found. Are you running in interactive mode? If so please provide a name and set interactive=True.')
            name = os.path.basename(mainfile)
            if name.endswith('.py'):
                name = name[:-3]
            elif name.endswith('.pyc'):
                name = name[:-4]
        super().__init__(path=name, ingredients=ingredients, interactive=interactive, base_dir=base_dir, _caller_globals=caller_globals, save_git_info=save_git_info)
        self.default_command = None
        self.command(print_config, unobserved=True)
        self.command(print_dependencies, unobserved=True)
        self.command(save_config, unobserved=True)
        self.command(print_named_configs(self), unobserved=True)
        self.observers = []
        self.current_run = None
        self.captured_out_filter = None
        'Filter function to be applied to captured output of a run'
        self.option_hooks = []

    def main(self, function):
        if False:
            for i in range(10):
                print('nop')
        '\n        Decorator to define the main function of the experiment.\n\n        The main function of an experiment is the default command that is being\n        run when no command is specified, or when calling the run() method.\n\n        Usually it is more convenient to use ``automain`` instead.\n        '
        captured = self.command(function)
        self.default_command = captured.__name__
        return captured

    def automain(self, function):
        if False:
            print('Hello World!')
        "\n        Decorator that defines *and runs* the main function of the experiment.\n\n        The decorated function is marked as the default command for this\n        experiment, and the command-line interface is automatically run when\n        the file is executed.\n\n        The method decorated by this should be last in the file because is\n        equivalent to:\n\n        Example\n        -------\n        ::\n\n            @ex.main\n            def my_main():\n                pass\n\n            if __name__ == '__main__':\n                ex.run_commandline()\n        "
        captured = self.main(function)
        if function.__module__ == '__main__':
            import inspect
            main_filename = inspect.getfile(function)
            if main_filename == '<stdin>' or (main_filename.startswith('<ipython-input-') and main_filename.endswith('>')):
                raise RuntimeError('Cannot use @ex.automain decorator in interactive mode. Use @ex.main instead.')
            self.run_commandline()
        return captured

    def option_hook(self, function):
        if False:
            while True:
                i = 10
        "\n        Decorator for adding an option hook function.\n\n        An option hook is a function that is called right before a run\n        is created. It receives (and potentially modifies) the options\n        dictionary. That is, the dictionary of commandline options used for\n        this run.\n\n        Notes\n        -----\n        The decorated function MUST have an argument called options.\n\n        The options also contain ``'COMMAND'`` and ``'UPDATE'`` entries,\n        but changing them has no effect. Only modification on\n        flags (entries starting with ``'--'``) are considered.\n        "
        sig = Signature(function)
        if 'options' not in sig.arguments:
            raise KeyError("option_hook functions must have an argument called 'options', but got {}".format(sig.arguments))
        self.option_hooks.append(function)
        return function

    def get_usage(self, program_name=None):
        if False:
            return 10
        'Get the commandline usage string for this experiment.'
        program_name = os.path.relpath(program_name or sys.argv[0] or 'Dummy', self.base_dir)
        commands = OrderedDict(self.gather_commands())
        long_usage = format_usage(program_name, self.doc, commands, self.all_cli_options)
        internal_usage = format_usage('dummy', self.doc, commands, self.all_cli_options)
        short_usage = printable_usage(long_usage)
        return (short_usage, long_usage, internal_usage)

    def run(self, command_name: Optional[str]=None, config_updates: Optional[dict]=None, named_configs: Sequence[str]=(), info: Optional[dict]=None, meta_info: Optional[dict]=None, options: Optional[dict]=None) -> Run:
        if False:
            return 10
        '\n        Run the main function of the experiment or a given command.\n\n        Parameters\n        ----------\n        command_name\n            Name of the command to be run. Defaults to main function.\n\n        config_updates\n            Changes to the configuration as a nested dictionary\n\n        named_configs\n            list of names of named_configs to use\n\n        info\n            Additional information for this run.\n\n        meta_info\n            Additional meta information for this run.\n\n        options\n            Dictionary of options to use\n\n        Returns\n        -------\n        The Run object corresponding to the finished run.\n        '
        run = self._create_run(command_name, config_updates, named_configs, info, meta_info, options)
        run()
        return run

    def run_commandline(self, argv=None) -> Optional[Run]:
        if False:
            i = 10
            return i + 15
        '\n        Run the command-line interface of this experiment.\n\n        If ``argv`` is omitted it defaults to ``sys.argv``.\n\n        Parameters\n        ----------\n        argv\n            Command-line as string or list of strings like ``sys.argv``.\n\n        Returns\n        -------\n        The Run object corresponding to the finished run.\n\n        '
        argv = ensure_wellformed_argv(argv)
        (short_usage, usage, internal_usage) = self.get_usage()
        args = docopt(internal_usage, [str(a) for a in argv[1:]], help=False)
        cmd_name = args.get('COMMAND') or self.default_command
        (config_updates, named_configs) = get_config_updates(args['UPDATE'])
        err = self._check_command(cmd_name)
        if not args['help'] and err:
            print(short_usage)
            print(err)
            sys.exit(1)
        if self._handle_help(args, usage):
            sys.exit()
        try:
            return self.run(cmd_name, config_updates, named_configs, info={}, meta_info={}, options=args)
        except Exception as e:
            if self.current_run:
                debug = self.current_run.debug
            else:
                debug = args.get('--debug', False)
            if debug:
                raise
            elif self.current_run and self.current_run.pdb:
                import traceback
                import pdb
                traceback.print_exception(*sys.exc_info())
                pdb.post_mortem()
            else:
                if isinstance(e, SacredError):
                    print(format_sacred_error(e, short_usage), file=sys.stderr)
                else:
                    print_filtered_stacktrace()
                sys.exit(1)

    def open_resource(self, filename: PathType, mode: str='r'):
        if False:
            print('Hello World!')
        'Open a file and also save it as a resource.\n\n        Opens a file, reports it to the observers as a resource, and returns\n        the opened file.\n\n        In Sacred terminology a resource is a file that the experiment needed\n        to access during a run. In case of a MongoObserver that means making\n        sure the file is stored in the database (but avoiding duplicates) along\n        its path and md5 sum.\n\n        This function can only be called during a run, and just calls the\n        :py:meth:`sacred.run.Run.open_resource` method.\n\n        Parameters\n        ----------\n        filename\n            name of the file that should be opened\n        mode\n            mode that file will be open\n\n        Returns\n        -------\n        The opened file-object.\n        '
        assert self.current_run is not None, 'Can only be called during a run.'
        return self.current_run.open_resource(filename, mode)

    def add_resource(self, filename: PathType) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a file as a resource.\n\n        In Sacred terminology a resource is a file that the experiment needed\n        to access during a run. In case of a MongoObserver that means making\n        sure the file is stored in the database (but avoiding duplicates) along\n        its path and md5 sum.\n\n        This function can only be called during a run, and just calls the\n        :py:meth:`sacred.run.Run.add_resource` method.\n\n        Parameters\n        ----------\n        filename\n            name of the file to be stored as a resource\n        '
        assert self.current_run is not None, 'Can only be called during a run.'
        self.current_run.add_resource(filename)

    def add_artifact(self, filename: PathType, name: Optional[str]=None, metadata: Optional[dict]=None, content_type: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Add a file as an artifact.\n\n        In Sacred terminology an artifact is a file produced by the experiment\n        run. In case of a MongoObserver that means storing the file in the\n        database.\n\n        This function can only be called during a run, and just calls the\n        :py:meth:`sacred.run.Run.add_artifact` method.\n\n        Parameters\n        ----------\n        filename\n            name of the file to be stored as artifact\n        name\n            optionally set the name of the artifact.\n            Defaults to the relative file-path.\n        metadata\n            optionally attach metadata to the artifact.\n            This only has an effect when using the MongoObserver.\n        content_type\n            optionally attach a content-type to the artifact.\n            This only has an effect when using the MongoObserver.\n        '
        assert self.current_run is not None, 'Can only be called during a run.'
        self.current_run.add_artifact(filename, name, metadata, content_type)

    @property
    def info(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        'Access the info-dict for storing custom information.\n\n        Only works during a run and is essentially a shortcut to:\n\n        Example\n        -------\n        ::\n\n            @ex.capture\n            def my_captured_function(_run):\n                # [...]\n                _run.info   # == ex.info\n        '
        return self.current_run.info

    def log_scalar(self, name: str, value: float, step: Optional[int]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a new measurement.\n\n        The measurement will be processed by the MongoDB* observer\n        during a heartbeat event.\n        Other observers are not yet supported.\n\n\n        Parameters\n        ----------\n        name\n            The name of the metric, e.g. training.loss\n        value\n            The measured value\n        step\n            The step number (integer), e.g. the iteration number\n            If not specified, an internal counter for each metric\n            is used, incremented by one.\n        '
        self.current_run.log_scalar(name, value, step)

    def post_process_name(self, name, ingredient):
        if False:
            i = 10
            return i + 15
        if ingredient == self:
            return name[len(self.path) + 1:]
        return name

    def get_default_options(self) -> dict:
        if False:
            i = 10
            return i + 15
        "Get a dictionary of default options as used with run.\n\n        Returns\n        -------\n        A dictionary containing option keys of the form '--beat_interval'.\n        Their values are boolean if the option is a flag, otherwise None or\n        its default value.\n\n        "
        default_options = {}
        for option in self.all_cli_options:
            if isinstance(option, CLIOption):
                if option.is_flag:
                    default_value = False
                else:
                    default_value = None
            elif option.arg is None:
                default_value = False
            else:
                default_value = None
            default_options[option.get_flag()] = default_value
        return default_options

    def _create_run(self, command_name=None, config_updates=None, named_configs=(), info=None, meta_info=None, options=None):
        if False:
            i = 10
            return i + 15
        command_name = command_name or self.default_command
        if command_name is None:
            raise RuntimeError('No command found to be run. Specify a command or define a main function.')
        default_options = self.get_default_options()
        if options:
            default_options.update(options)
        options = default_options
        for oh in self.option_hooks:
            oh(options=options)
        run = create_run(self, command_name, config_updates, named_configs=named_configs, force=options.get(commandline_options.force_option.get_flag(), False), log_level=options.get(commandline_options.loglevel_option.get_flag(), None))
        if info is not None:
            run.info.update(info)
        run.meta_info['command'] = command_name
        run.meta_info['options'] = options
        run.meta_info['named_configs'] = list(named_configs)
        if config_updates is not None:
            run.meta_info['config_updates'] = config_updates
        if meta_info:
            run.meta_info.update(meta_info)
        options_list = gather_command_line_options() + self.additional_cli_options
        for option in options_list:
            option_value = options.get(option.get_flag(), False)
            if option_value:
                option.apply(option_value, run)
        self.current_run = run
        return run

    def _check_command(self, cmd_name):
        if False:
            print('Hello World!')
        commands = dict(self.gather_commands())
        if cmd_name is not None and cmd_name not in commands:
            return 'Error: Command "{}" not found. Available commands are: {}'.format(cmd_name, ', '.join(commands.keys()))
        if cmd_name is None:
            return 'Error: No command found to be run. Specify a command or define main function. Available commands are: {}'.format(', '.join(commands.keys()))

    def _handle_help(self, args, usage):
        if False:
            while True:
                i = 10
        if args['help'] or args['--help']:
            if args['COMMAND'] is None:
                print(usage)
                return True
            else:
                commands = dict(self.gather_commands())
                print(help_for_command(commands[args['COMMAND']]))
                return True
        return False

def gather_command_line_options(filter_disabled=None):
    if False:
        print('Hello World!')
    'Get a sorted list of all CommandLineOption subclasses.'
    if filter_disabled is None:
        filter_disabled = not SETTINGS.COMMAND_LINE.SHOW_DISABLED_OPTIONS
    options = []
    for opt in get_inheritors(commandline_options.CommandLineOption):
        warnings.warn('Subclassing `CommandLineOption` is deprecated. Please use the `sacred.cli_option` decorator and pass the function to the Experiment constructor.')
        if filter_disabled and (not opt._enabled):
            continue
        options.append(opt)
    options += DEFAULT_COMMAND_LINE_OPTIONS
    return sorted(options, key=commandline_options.get_name)
DEFAULT_COMMAND_LINE_OPTIONS = [s3_option, commandline_options.pdb_option, commandline_options.debug_option, file_storage_option, commandline_options.loglevel_option, mongo_db_option, sql_option, commandline_options.capture_option, commandline_options.help_option, commandline_options.print_config_option, commandline_options.name_option, commandline_options.id_option, commandline_options.priority_option, commandline_options.unobserved_option, commandline_options.beat_interval_option, commandline_options.queue_option, commandline_options.force_option, commandline_options.comment_option, commandline_options.enforce_clean_option, tiny_db_option]