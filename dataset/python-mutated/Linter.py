from contextlib import contextmanager
from functools import partial, partialmethod
import logging
import inspect
from itertools import chain
import re
import shutil
from subprocess import check_call, CalledProcessError, DEVNULL
from types import MappingProxyType
from cli_helpers.utils import strip_ansi
from coalib.bearlib.abstractions.LinterClass import LinterClass
from coalib.bears.LocalBear import LocalBear
from coalib.bears.GlobalBear import GlobalBear
from coala_utils.ContextManagers import make_temp
from coala_utils.decorators import assert_right_type, enforce_signature
from coalib.misc.Shell import run_shell_command
from coalib.results.Diff import Diff
from coalib.results.Result import Result
from coalib.results.SourceRange import SourceRange
from coalib.results.RESULT_SEVERITY import RESULT_SEVERITY
from coalib.settings.FunctionMetadata import FunctionMetadata

def _prepare_options(options, bear_class):
    if False:
        while True:
            i = 10
    '\n    Prepares options for ``linter`` for a given options dict in-place.\n\n    :param options:\n        The options dict that contains user/developer inputs.\n    :param bear_class:\n        The Bear ``class`` which is being decorated by ``linter``.\n    '
    allowed_options = {'executable', 'output_format', 'use_stdin', 'use_stdout', 'use_stderr', 'normalize_line_numbers', 'normalize_column_numbers', 'remove_zero_numbers', 'config_suffix', 'executable_check_fail_info', 'prerequisite_check_command', 'global_bear', 'strip_ansi'}
    if not options['use_stdout'] and (not options['use_stderr']):
        raise ValueError('No output streams provided at all.')
    if options['output_format'] == 'corrected' or options['output_format'] == 'unified-diff':
        if 'diff_severity' in options and options['diff_severity'] not in RESULT_SEVERITY.reverse:
            raise TypeError('Invalid value for `diff_severity`: ' + repr(options['diff_severity']))
        if 'result_message' in options:
            assert_right_type(options['result_message'], str, 'result_message')
        if 'diff_distance' in options:
            assert_right_type(options['diff_distance'], int, 'diff_distance')
        allowed_options |= {'diff_severity', 'result_message', 'diff_distance'}
    elif options['output_format'] == 'regex':
        if 'output_regex' not in options:
            raise ValueError("`output_regex` needed when specified output-format 'regex'.")
        options['output_regex'] = re.compile(options['output_regex'])
        supported_names = {'origin', 'message', 'severity', 'filename', 'line', 'column', 'end_line', 'end_column', 'additional_info'}
        no_of_non_named_groups = options['output_regex'].groups - len(options['output_regex'].groupindex)
        if no_of_non_named_groups:
            logging.warning("{}: Using unnecessary capturing groups affects the performance of coala. You should use '(?:<pattern>)' instead of '(<pattern>)' for your regex.".format(bear_class.__name__))
        for capture_group_name in options['output_regex'].groupindex:
            if capture_group_name not in supported_names:
                logging.warning("{}: Superfluous capturing group '{}' used. Is this a typo? If not, consider removing the capturing group to improve coala's performance.".format(bear_class.__name__, capture_group_name))
        if 'severity_map' in options:
            if 'severity' not in options['output_regex'].groupindex:
                raise ValueError('Provided `severity_map` but named group `severity` is not used in `output_regex`.')
            assert_right_type(options['severity_map'], dict, 'severity_map')
            for (key, value) in options['severity_map'].items():
                assert_right_type(key, str, 'severity_map key')
                try:
                    assert_right_type(value, int, '<severity_map dict-value>')
                except TypeError:
                    raise TypeError('The value {!r} for key {!r} inside given severity-map is no valid severity value.'.format(value, key))
                if value not in RESULT_SEVERITY.reverse:
                    raise TypeError('Invalid severity value {!r} for key {!r} inside given severity-map.'.format(value, key))
            options['severity_map'] = {key.lower(): value for (key, value) in options['severity_map'].items()}
        if 'result_message' in options:
            assert_right_type(options['result_message'], str, 'result_message')
        allowed_options |= {'output_regex', 'severity_map', 'result_message'}
    elif options['output_format'] is not None:
        raise ValueError('Invalid `output_format` specified.')
    if options['prerequisite_check_command']:
        if 'prerequisite_check_fail_message' in options:
            assert_right_type(options['prerequisite_check_fail_message'], str, 'prerequisite_check_fail_message')
        else:
            options['prerequisite_check_fail_message'] = 'Prerequisite check failed.'
        allowed_options.add('prerequisite_check_fail_message')
    if options['global_bear'] and options['use_stdin']:
        raise ValueError("Incompatible arguments provided:'use_stdin' and 'global_bear' can't both be True.")
    superfluous_options = options.keys() - allowed_options
    if superfluous_options:
        raise ValueError('Invalid keyword arguments provided: ' + ', '.join((repr(s) for s in sorted(superfluous_options))))

def _create_linter(klass, options):
    if False:
        i = 10
        return i + 15
    _prepare_options(options, klass)

    class LinterMeta(type):

        def __repr__(cls):
            if False:
                while True:
                    i = 10
            return '<{} linter class (wrapping {!r}) at ({})>'.format(cls.__name__, options['executable'], hex(id(cls)))

    class LinterBase(metaclass=LinterMeta):

        @staticmethod
        def generate_config(filename, file):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Generates the content of a config-file the linter-tool might need.\n\n            The contents generated from this function are written to a\n            temporary file and the path is provided inside\n            ``create_arguments()``.\n\n            By default no configuration is generated.\n\n            You can provide additional keyword arguments and defaults. These\n            will be interpreted as required settings that need to be provided\n            through a coafile-section.\n\n            :param filename:\n                The name of the file currently processed.\n            :param file:\n                The contents of the file currently processed.\n            :return:\n                The config-file-contents as a string or ``None``.\n            '
            return None

        @staticmethod
        def get_executable():
            if False:
                i = 10
                return i + 15
            '\n            Returns the executable of this class.\n\n            :return:\n                The executable name.\n            '
            return options['executable']

        @classmethod
        def check_prerequisites(cls):
            if False:
                print('Hello World!')
            '\n            Checks whether the linter-tool the bear uses is operational.\n\n            :return:\n                True if operational, otherwise a string containing more info.\n            '
            if shutil.which(cls.get_executable()) is None:
                return repr(cls.get_executable()) + ' is not installed.' + (' ' + options['executable_check_fail_info'] if options['executable_check_fail_info'] else '')
            else:
                if options['prerequisite_check_command']:
                    try:
                        check_call(options['prerequisite_check_command'], stdout=DEVNULL, stderr=DEVNULL)
                        return True
                    except (OSError, CalledProcessError):
                        return options['prerequisite_check_fail_message']
                return True

        @classmethod
        def _get_create_arguments_metadata(cls):
            if False:
                i = 10
                return i + 15
            return FunctionMetadata.from_function(cls.create_arguments, omit={'self', 'filename', 'file', 'config_file'})

        @classmethod
        def _get_generate_config_metadata(cls):
            if False:
                return 10
            return FunctionMetadata.from_function(cls.generate_config, omit={'filename', 'file'})

        @classmethod
        def _get_process_output_metadata(cls):
            if False:
                while True:
                    i = 10
            metadata = FunctionMetadata.from_function(cls.process_output)
            if options['output_format'] is None:
                omitted = {'self', 'output', 'filename', 'file'}
            else:
                omitted = set(chain(metadata.non_optional_params, metadata.optional_params))
            metadata.omit = omitted
            return metadata

        @classmethod
        def get_metadata(cls):
            if False:
                return 10
            merged_metadata = FunctionMetadata.merge(cls._get_process_output_metadata(), cls._get_generate_config_metadata(), cls._get_create_arguments_metadata())
            merged_metadata.desc = inspect.getdoc(cls)
            return merged_metadata

        def _convert_output_regex_match_to_result(self, match, filename, severity_map, result_message):
            if False:
                return 10
            '\n            Converts the matched named-groups of ``output_regex`` to an actual\n            ``Result``.\n\n            :param match:\n                The regex match object.\n            :param filename:\n                The name of the file this match belongs to or ``None`` for\n                project scope.\n            :param severity_map:\n                The dict to use to map the severity-match to an actual\n                ``RESULT_SEVERITY``.\n            :param result_message:\n                The static message to use for results instead of grabbing it\n                from the executable output via the ``message`` named regex\n                group.\n            '
            groups = match.groupdict()
            if 'severity' in groups:
                try:
                    groups['severity'] = severity_map[groups['severity'].lower()]
                except KeyError:
                    self.warn(repr(groups['severity']) + ' not found in severity-map. Assuming `RESULT_SEVERITY.NORMAL`.')
                    groups['severity'] = RESULT_SEVERITY.NORMAL
            else:
                groups['severity'] = RESULT_SEVERITY.NORMAL
            for variable in ('line', 'column', 'end_line', 'end_column'):
                groups[variable] = None if groups.get(variable, None) is None else int(groups[variable])

            def add_one(x):
                if False:
                    for i in range(10):
                        print('nop')
                return None if x is None else x + 1
            if options['normalize_line_numbers']:
                for variable in ('line', 'end_line'):
                    groups[variable] = add_one(groups[variable])
            if options['normalize_column_numbers']:
                for variable in ('column', 'end_column'):
                    groups[variable] = add_one(groups[variable])
            if 'origin' in groups:
                groups['origin'] = '{} ({})'.format(klass.__name__, groups['origin'].strip())
            if filename is None:
                filename = groups.get('filename', None)
            if options['remove_zero_numbers']:
                for variable in ('line', 'column', 'end_line', 'end_column'):
                    if groups[variable] == 0:
                        groups[variable] = None
            result_params = {'origin': groups.get('origin', self), 'message': groups.get('message', '').strip() if result_message is None else result_message, 'severity': groups['severity'], 'additional_info': groups.get('additional_info', '').strip()}
            if filename:
                source_range = SourceRange.from_values(filename, groups['line'], groups['column'], groups['end_line'], groups['end_column'])
                result_params['affected_code'] = (source_range,)
            return Result(**result_params)

        def process_diff(self, diff, filename, diff_severity, result_message, diff_distance):
            if False:
                print('Hello World!')
            '\n            Processes the given ``coalib.results.Diff`` object and yields\n            correction results.\n\n            :param diff:\n                A ``coalib.results.Diff`` object containing\n                differences of the file named ``filename``.\n            :param filename:\n                The name of the file currently being corrected.\n            :param diff_severity:\n                The severity to use for generating results.\n            :param result_message:\n                The message to use for generating results.\n            :param diff_distance:\n                Number of unchanged lines that are allowed in between two\n                changed lines so they get yielded as one diff. If a negative\n                distance is given, every change will be yielded as an own diff,\n                even if they are right beneath each other.\n            :return:\n                An iterator returning results containing patches for the\n                file to correct.\n            '
            for splitted_diff in diff.split_diff(distance=diff_distance):
                yield Result(self, result_message, affected_code=splitted_diff.affected_code(filename), diffs={filename: splitted_diff}, severity=diff_severity)

        def process_output_corrected(self, output, filename, file, diff_severity=RESULT_SEVERITY.NORMAL, result_message='Inconsistency found.', diff_distance=1):
            if False:
                while True:
                    i = 10
            "\n            Processes the executable's output as a corrected file.\n\n            :param output:\n                The output of the program as a string.\n            :param filename:\n                The filename of the file currently being corrected.\n            :param file:\n                The contents of the file currently being corrected.\n            :param diff_severity:\n                The severity to use for generating results.\n            :param result_message:\n                The message to use for generating results.\n            :param diff_distance:\n                Number of unchanged lines that are allowed in between two\n                changed lines so they get yielded as one diff. If a negative\n                distance is given, every change will be yielded as an own diff,\n                even if they are right beneath each other.\n            :return:\n                An iterator returning results containing patches for the\n                file to correct.\n            "
            return self.process_diff(Diff.from_string_arrays(file, output.splitlines(keepends=True)), filename, diff_severity, result_message, diff_distance)

        def process_output_unified_diff(self, output, filename, file, diff_severity=RESULT_SEVERITY.NORMAL, result_message='Inconsistency found.', diff_distance=1):
            if False:
                return 10
            "\n            Processes the executable's output as a unified diff.\n\n            :param output:\n                The output of the program as a string containing the\n                unified diff for correction.\n            :param filename:\n                The filename of the file currently being corrected.\n            :param file:\n                The contents of the file currently being corrected.\n            :param diff_severity:\n                The severity to use for generating results.\n            :param result_message:\n                The message-string to use for generating results.\n            :param diff_distance:\n                Number of unchanged lines that are allowed in between two\n                changed lines so they get yielded as one diff. If a negative\n                distance is given, every change will be yielded as an own diff,\n                even if they are right beneath each other.\n            :return:\n                An iterator returning results containing patches for the\n                file to correct.\n            "
            return self.process_diff(Diff.from_unified_diff(output, file), filename, diff_severity, result_message, diff_distance)

        def process_output_regex(self, output, filename, file, output_regex, severity_map=MappingProxyType({'critical': RESULT_SEVERITY.MAJOR, 'c': RESULT_SEVERITY.MAJOR, 'fatal': RESULT_SEVERITY.MAJOR, 'fail': RESULT_SEVERITY.MAJOR, 'f': RESULT_SEVERITY.MAJOR, 'error': RESULT_SEVERITY.MAJOR, 'err': RESULT_SEVERITY.MAJOR, 'e': RESULT_SEVERITY.MAJOR, 'warning': RESULT_SEVERITY.NORMAL, 'warn': RESULT_SEVERITY.NORMAL, 'w': RESULT_SEVERITY.NORMAL, 'information': RESULT_SEVERITY.INFO, 'info': RESULT_SEVERITY.INFO, 'i': RESULT_SEVERITY.INFO, 'note': RESULT_SEVERITY.INFO, 'suggestion': RESULT_SEVERITY.INFO}), result_message=None):
            if False:
                i = 10
                return i + 15
            "\n            Processes the executable's output using a regex.\n\n            :param output:\n                The output of the program as a string.\n            :param filename:\n                The filename of the file currently being corrected.\n            :param file:\n                The contents of the file currently being corrected.\n            :param output_regex:\n                The regex to parse the output with. It should use as many\n                of the following named groups (via ``(?P<name>...)``) to\n                provide a good result:\n\n                - filename - The name of the linted file. This is relevant for\n                    global bears only.\n                - line - The line where the issue starts.\n                - column - The column where the issue starts.\n                - end_line - The line where the issue ends.\n                - end_column - The column where the issue ends.\n                - severity - The severity of the issue.\n                - message - The message of the result.\n                - origin - The origin of the issue.\n                - additional_info - Additional info provided by the issue.\n\n                The groups ``line``, ``column``, ``end_line`` and\n                ``end_column`` don't have to match numbers only, they can\n                also match nothing, the generated ``Result`` is filled\n                automatically with ``None`` then for the appropriate\n                properties.\n            :param severity_map:\n                A dict used to map a severity string (captured from the\n                ``output_regex`` with the named group ``severity``) to an\n                actual ``coalib.results.RESULT_SEVERITY`` for a result.\n            :param result_message:\n                The static message to use for results instead of grabbing it\n                from the executable output via the ``message`` named regex\n                group.\n            :return:\n                An iterator returning results.\n            "
            for match in re.finditer(output_regex, output):
                yield self._convert_output_regex_match_to_result(match, filename, severity_map=severity_map, result_message=result_message)
        if options['output_format'] is None:
            if not callable(getattr(klass, 'process_output', None)):
                raise ValueError('`process_output` not provided by given class {!r}.'.format(klass.__name__))
        else:
            if hasattr(klass, 'process_output'):
                raise ValueError('Found `process_output` already defined by class {!r}, but {!r} output-format is specified.'.format(klass.__name__, options['output_format']))
            if options['output_format'] == 'corrected':
                _process_output_args = {key: options[key] for key in ('result_message', 'diff_severity', 'diff_distance') if key in options}
                _processing_function = partialmethod(process_output_corrected, **_process_output_args)
            elif options['output_format'] == 'unified-diff':
                _process_output_args = {key: options[key] for key in ('result_message', 'diff_severity', 'diff_distance') if key in options}
                _processing_function = partialmethod(process_output_unified_diff, **_process_output_args)
            else:
                assert options['output_format'] == 'regex'
                _process_output_args = {key: options[key] for key in ('output_regex', 'severity_map', 'result_message') if key in options}
                _processing_function = partialmethod(process_output_regex, **_process_output_args)

            def process_output(self, output, filename=None, file=None):
                if False:
                    print('Hello World!')
                '\n                Processes the output of the executable and yields results\n                accordingly.\n\n                :param output:\n                    The output of the executable. This can be either a string\n                    or a tuple depending on the usage of ``use_stdout`` and\n                    ``use_stderr`` parameters of ``@linter``. If only one of\n                    these arguments is ``True``, a string is placed (containing\n                    the selected output stream). If both are ``True``, a tuple\n                    is placed with ``(stdout, stderr)``.\n                :param filename:\n                    The name of the file currently processed or ``None`` for\n                    project scope.\n                :param file:\n                    The contents of the file (line-splitted) or ``None`` for\n                    project scope.\n                '
                if isinstance(output, str):
                    output = (output,)
                for string in output:
                    yield from self._processing_function(string, filename, file)

        @classmethod
        @contextmanager
        def _create_config(cls, filename=None, file=None, **kwargs):
            if False:
                return 10
            '\n            Provides a context-manager that creates the config file if the\n            user provides one and cleans it up when done with linting.\n\n            :param filename:\n                The filename of the file being linted. ``None`` for project\n                scope.\n            :param file:\n                The content of the file being linted. ``None`` for project\n                scope.\n            :param kwargs:\n                Section settings passed from ``run()``.\n            :return:\n                A context-manager handling the config-file.\n            '
            content = cls.generate_config(filename, file, **kwargs)
            if content is None:
                yield None
            else:
                with make_temp(suffix=options['config_suffix']) as config_file:
                    with open(config_file, mode='w') as fl:
                        fl.write(content)
                    yield config_file

        def run(self, filename=None, file=None, **kwargs):
            if False:
                return 10
            '\n            Runs the wrapped tool.\n\n            :param filename:\n                The filename of the file being linted. ``None`` for project\n                scope.\n            :param file:\n                The content of the file being linted. ``None`` for project\n                scope.\n            '
            generate_config_kwargs = FunctionMetadata.filter_parameters(self._get_generate_config_metadata(), kwargs)
            with self._create_config(filename, file, **generate_config_kwargs) as config_file:
                create_arguments_kwargs = FunctionMetadata.filter_parameters(self._get_create_arguments_metadata(), kwargs)
                if isinstance(self, LocalBear):
                    args = self.create_arguments(filename, file, config_file, **create_arguments_kwargs)
                else:
                    args = self.create_arguments(config_file, **create_arguments_kwargs)
                try:
                    args = tuple(args)
                except TypeError:
                    self.err('The given arguments {!r} are not iterable.'.format(args))
                    return
                arguments = (self.get_executable(),) + args
                self.debug("Running '{}'".format(' '.join((str(arg) for arg in arguments))))
                result = run_shell_command(arguments, stdin=''.join(file) if options['use_stdin'] else None, cwd=self.get_config_dir())
                (stdout, stderr) = result
                output = []
                if options['use_stdout']:
                    output.append(stdout)
                elif stdout:
                    logging.warning('{}: Discarded stdout: {}'.format(self.__class__.__name__, stdout))
                if options['use_stderr']:
                    output.append(stderr)
                elif stderr:
                    logging.warning('{}: Discarded stderr: {}'.format(self.__class__.__name__, stderr))
                if result.code:
                    logging.warning('{}: Exit code {}'.format(self.__class__.__name__, result.code))
                if not any(output):
                    logging.info('{}: No output; skipping processing'.format(self.__class__.__name__))
                    return
                if options['strip_ansi']:
                    output = tuple(map(strip_ansi, output))
                if len(output) == 1:
                    output = output[0]
                else:
                    output = tuple(output)
                process_output_kwargs = FunctionMetadata.filter_parameters(self._get_process_output_metadata(), kwargs)
                return self.process_output(output, filename, file, **process_output_kwargs)

        def __repr__(self):
            if False:
                i = 10
                return i + 15
            return '<{} linter object (wrapping {!r}) at {}>'.format(type(self).__name__, self.get_executable(), hex(id(self)))

    class LocalLinterMeta(type(LinterBase), type(LocalBear)):
        """
        Solving base metaclasses conflict for ``LocalLinterBase``.
        """

    class LocalLinterBase(LinterBase, LocalBear, metaclass=LocalLinterMeta):

        @staticmethod
        def create_arguments(filename, file, config_file):
            if False:
                return 10
            '\n            Creates the arguments for the linter.\n\n            You can provide additional keyword arguments and defaults. These\n            will be interpreted as required settings that need to be provided\n            through a coafile-section.\n\n            :param filename:\n                The name of the file the linter-tool shall process.\n            :param file:\n                The contents of the file.\n            :param config_file:\n                The path of the config-file if used. ``None`` if unused.\n            :return:\n                A sequence of arguments to feed the linter-tool with.\n            '
            raise NotImplementedError

    class GlobalLinterMeta(type(LinterBase), type(GlobalBear)):
        """
        Solving base metaclasses conflict for ``GlobalLinterBase``.
        """

    class GlobalLinterBase(LinterBase, GlobalBear, metaclass=GlobalLinterMeta):

        @staticmethod
        def create_arguments(config_file):
            if False:
                return 10
            '\n            Creates the arguments for the linter.\n\n            You can provide additional keyword arguments and defaults. These\n            will be interpreted as required settings that need to be provided\n            through a coafile-section. This is the file agnostic version for\n            global bears.\n\n            :param config_file:\n                The path of the config-file if used. ``None`` if unused.\n            :return:\n                A sequence of arguments to feed the linter-tool with.\n            '
            raise NotImplementedError
    LinterBaseClass = GlobalLinterBase if options['global_bear'] else LocalLinterBase
    result_klass = type(klass.__name__, (klass, LinterBaseClass), {'__module__': klass.__module__})
    result_klass.__doc__ = klass.__doc__ or ''
    LinterClass.register(result_klass)
    return result_klass

@enforce_signature
def linter(executable: str, global_bear: bool=False, use_stdin: bool=False, use_stdout: bool=True, use_stderr: bool=False, normalize_line_numbers: bool=False, normalize_column_numbers: bool=False, remove_zero_numbers: bool=False, config_suffix: str='', executable_check_fail_info: str='', prerequisite_check_command: tuple=(), output_format: (str, None)=None, strip_ansi: bool=False, **options):
    if False:
        i = 10
        return i + 15
    "\n    Decorator that creates a ``Bear`` that is able to process results from\n    an external linter tool. Depending on the value of ``global_bear`` this\n    can either be a ``LocalBear`` or a ``GlobalBear``.\n\n    The main functionality is achieved through the ``create_arguments()``\n    function that constructs the command-line-arguments that get passed to your\n    executable.\n\n    >>> @linter('xlint', output_format='regex', output_regex='...')\n    ... class XLintBear:\n    ...     @staticmethod\n    ...     def create_arguments(filename, file, config_file):\n    ...         return '--lint', filename\n\n    Or for a ``GlobalBear`` without the ``filename`` and ``file``:\n\n    >>> @linter('ylint',\n    ...         global_bear=True,\n    ...         output_format='regex',\n    ...         output_regex='...')\n    ... class YLintBear:\n    ...     def create_arguments(self, config_file):\n    ...         return '--lint', self.file_dict.keys()\n\n    Requiring settings is possible like in ``Bear.run()`` with supplying\n    additional keyword arguments (and if needed with defaults).\n\n    >>> @linter('xlint', output_format='regex', output_regex='...')\n    ... class XLintBear:\n    ...     @staticmethod\n    ...     def create_arguments(filename,\n    ...                          file,\n    ...                          config_file,\n    ...                          lintmode: str,\n    ...                          enable_aggressive_lints: bool=False):\n    ...         arguments = ('--lint', filename, '--mode=' + lintmode)\n    ...         if enable_aggressive_lints:\n    ...             arguments += ('--aggressive',)\n    ...         return arguments\n\n    Sometimes your tool requires an actual file that contains configuration.\n    ``linter`` allows you to just define the contents the configuration shall\n    contain via ``generate_config()`` and handles everything else for you.\n\n    >>> @linter('xlint', output_format='regex', output_regex='...')\n    ... class XLintBear:\n    ...     @staticmethod\n    ...     def generate_config(filename,\n    ...                         file,\n    ...                         lintmode,\n    ...                         enable_aggressive_lints):\n    ...         modestring = ('aggressive'\n    ...                       if enable_aggressive_lints else\n    ...                       'non-aggressive')\n    ...         contents = ('<xlint>',\n    ...                     '    <mode>' + lintmode + '</mode>',\n    ...                     '    <aggressive>' + modestring + '</aggressive>',\n    ...                     '</xlint>')\n    ...         return '\\n'.join(contents)\n    ...\n    ...     @staticmethod\n    ...     def create_arguments(filename,\n    ...                          file,\n    ...                          config_file):\n    ...         return '--lint', filename, '--config', config_file\n\n    As you can see you don't need to copy additional keyword-arguments you\n    introduced from ``create_arguments()`` to ``generate_config()`` and\n    vice-versa. ``linter`` takes care of forwarding the right arguments to the\n    right place, so you are able to avoid signature duplication.\n\n    If you override ``process_output``, you have the same feature like above\n    (auto-forwarding of the right arguments defined in your function\n    signature).\n\n    Note when overriding ``process_output``: Providing a single output stream\n    (via ``use_stdout`` or ``use_stderr``) puts the according string attained\n    from the stream into parameter ``output``, providing both output streams\n    inputs a tuple with ``(stdout, stderr)``. Providing ``use_stdout=False``\n    and ``use_stderr=False`` raises a ``ValueError``. By default ``use_stdout``\n    is ``True`` and ``use_stderr`` is ``False``.\n\n    Every ``linter`` is also a subclass of the ``LinterClass`` class.\n\n    >>> issubclass(XLintBear, LinterClass)\n    True\n\n    Documentation:\n    Bear description shall be provided at class level.\n    If you document your additional parameters inside ``create_arguments``,\n    ``generate_config`` and ``process_output``, beware that conflicting\n    documentation between them may be overridden. Document duplicated\n    parameters inside ``create_arguments`` first, then in ``generate_config``\n    and after that inside ``process_output``.\n\n    For the tutorial see:\n    http://api.coala.io/en/latest/Developers/Writing_Linter_Bears.html\n\n    :param executable:\n        The linter tool.\n    :param use_stdin:\n        Whether the input file is sent via stdin instead of passing it over the\n        command-line-interface.\n    :param use_stdout:\n        Whether to use the stdout output stream.\n        Incompatible with ``global_bear=True``.\n    :param use_stderr:\n        Whether to use the stderr output stream.\n    :param normalize_line_numbers:\n        Whether to normalize line numbers (increase by one) to fit\n        coala's one-based convention.\n    :param normalize_column_numbers:\n        Whether to normalize column numbers (increase by one) to fit\n        coala's one-based convention.\n    :param remove_zero_numbers:\n        Whether to remove 0 line or column number and use None instead.\n    :param config_suffix:\n        The suffix-string to append to the filename of the configuration file\n        created when ``generate_config`` is supplied. Useful if your executable\n        expects getting a specific file-type with specific file-ending for the\n        configuration file.\n    :param executable_check_fail_info:\n        Information that is provided together with the fail message from the\n        normal executable check. By default no additional info is printed.\n    :param prerequisite_check_command:\n        A custom command to check for when ``check_prerequisites`` gets\n        invoked (via ``subprocess.check_call()``). Must be an ``Iterable``.\n    :param prerequisite_check_fail_message:\n        A custom message that gets displayed when ``check_prerequisites``\n        fails while invoking ``prerequisite_check_command``. Can only be\n        provided together with ``prerequisite_check_command``.\n    :param global_bear:\n        Whether the created bear should be a ``GlobalBear`` or not. Global\n        bears will be run once on the whole project, instead of once per file.\n        Incompatible with ``use_stdin=True``.\n    :param output_format:\n        The output format of the underlying executable. Valid values are\n\n        - ``None``: Define your own format by overriding ``process_output``.\n          Overriding ``process_output`` is then mandatory, not specifying it\n          raises a ``ValueError``.\n        - ``'regex'``: Parse output using a regex. See parameter\n          ``output_regex``.\n        - ``'corrected'``: The output is the corrected of the given file. Diffs\n          are then generated to supply patches for results.\n        - ``'unified-diff'``: The output is the unified diff of the corrections.\n          Patches are then supplied for results using this output.\n\n        Passing something else raises a ``ValueError``.\n    :param output_regex:\n        The regex expression as a string that is used to parse the output\n        generated by the underlying executable. It should use as many of the\n        following named groups (via ``(?P<name>...)``) to provide a good\n        result:\n\n        - filename - The name of the linted file. This is relevant for\n            global bears only.\n        - line - The line where the issue starts.\n        - column - The column where the issue starts.\n        - end_line - The line where the issue ends.\n        - end_column - The column where the issue ends.\n        - severity - The severity of the issue.\n        - message - The message of the result.\n        - origin - The origin of the issue.\n        - additional_info - Additional info provided by the issue.\n\n        The groups ``line``, ``column``, ``end_line`` and ``end_column`` don't\n        have to match numbers only, they can also match nothing, the generated\n        ``Result`` is filled automatically with ``None`` then for the\n        appropriate properties.\n\n        Needs to be provided if ``output_format`` is ``'regex'``.\n    :param severity_map:\n        A dict used to map a severity string (captured from the\n        ``output_regex`` with the named group ``severity``) to an actual\n        ``coalib.results.RESULT_SEVERITY`` for a result. Severity strings are\n        mapped **case-insensitive**!\n\n        - ``RESULT_SEVERITY.MAJOR``: Mapped by ``critical``, ``c``,\n          ``fatal``, ``fail``, ``f``, ``error``, ``err`` or ``e``.\n        - ``RESULT_SEVERITY.NORMAL``: Mapped by ``warning``, ``warn`` or ``w``.\n        - ``RESULT_SEVERITY.INFO``: Mapped by ``information``, ``info``, ``i``,\n          ``note`` or ``suggestion``.\n\n        A ``ValueError`` is raised when the named group ``severity`` is not\n        used inside ``output_regex`` and this parameter is given.\n    :param diff_severity:\n        The severity to use for all results if ``output_format`` is\n        ``'corrected'`` or ``'unified-diff'``. By default this value is\n        ``coalib.results.RESULT_SEVERITY.NORMAL``. The given value needs to be\n        defined inside ``coalib.results.RESULT_SEVERITY``.\n    :param result_message:\n        The message-string to use for all results. Can be used only together\n        with ``corrected`` or ``unified-diff`` or ``regex`` output format.\n        When using ``corrected`` or ``unified-diff``, the default value is\n        ``'Inconsistency found.'``, while for ``regex`` this static message is\n        disabled and the message matched by ``output_regex`` is used instead.\n    :param diff_distance:\n        Number of unchanged lines that are allowed in between two changed lines\n        so they get yielded as one diff if ``corrected`` or ``unified-diff``\n        output-format is given. If a negative distance is given, every change\n        will be yielded as an own diff, even if they are right beneath each\n        other. By default this value is ``1``.\n    :param strip_ansi:\n        Supresses colored output from linters when enabled by stripping the\n        ascii characters around the text.\n    :raises ValueError:\n        Raised when invalid options are supplied.\n    :raises TypeError:\n        Raised when incompatible types are supplied.\n        See parameter documentations for allowed types.\n    :return:\n        A ``LocalBear`` derivation that lints code using an external tool.\n    "
    options['executable'] = executable
    options['output_format'] = output_format
    options['use_stdin'] = use_stdin
    options['use_stdout'] = use_stdout
    options['use_stderr'] = use_stderr
    options['normalize_line_numbers'] = normalize_line_numbers
    options['normalize_column_numbers'] = normalize_column_numbers
    options['remove_zero_numbers'] = remove_zero_numbers
    options['config_suffix'] = config_suffix
    options['executable_check_fail_info'] = executable_check_fail_info
    options['prerequisite_check_command'] = prerequisite_check_command
    options['global_bear'] = global_bear
    options['strip_ansi'] = strip_ansi
    return partial(_create_linter, options=options)