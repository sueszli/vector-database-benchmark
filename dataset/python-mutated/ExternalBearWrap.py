import json
import inspect
from functools import partial
from collections import OrderedDict
from coalib.bears.LocalBear import LocalBear
from coala_utils.decorators import enforce_signature
from coalib.misc.Shell import run_shell_command
from coalib.results.Result import Result
from coalib.results.SourceRange import SourceRange
from coalib.settings.FunctionMetadata import FunctionMetadata

def _prepare_options(options):
    if False:
        return 10
    '\n    Checks for illegal options and raises ValueError.\n\n    :param options:\n        The options dict that contains user/developer inputs.\n    :raises ValueError:\n        Raised when illegal options are specified.\n    '
    allowed_options = {'executable', 'settings'}
    superfluous_options = options.keys() - allowed_options
    if superfluous_options:
        raise ValueError('Invalid keyword arguments provided: ' + ', '.join((repr(s) for s in sorted(superfluous_options))))
    if 'settings' not in options:
        options['settings'] = {}

def _create_wrapper(klass, options):
    if False:
        print('Hello World!')
    NoDefaultValue = object()

    class ExternalBearWrapBase(LocalBear):

        @staticmethod
        def create_arguments():
            if False:
                i = 10
                return i + 15
            '\n            This method has to be implemented by the class that uses\n            the decorator in order to create the arguments needed for\n            the executable.\n            '
            return ()

        @classmethod
        def get_executable(cls):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Returns the executable of this class.\n\n            :return:\n                The executable name.\n            '
            return options['executable']

        @staticmethod
        def _normalize_desc(description, setting_type, default_value=NoDefaultValue):
            if False:
                while True:
                    i = 10
            '\n            Normalizes the description of the parameters only if there\n            is none provided.\n\n            :param description:\n                The parameter description to be modified in case it is empty.\n            :param setting_type:\n                The type of the setting. It is needed to create the final\n                tuple.\n            :param default_value:\n                The default value of the setting.\n            :return:\n                A value for the OrderedDict in the ``FunctionMetadata`` object.\n            '
            if description == '':
                description = FunctionMetadata.str_nodesc
            if default_value is NoDefaultValue:
                return (description, setting_type)
            else:
                return (description + ' ' + FunctionMetadata.str_optional.format(default_value), setting_type, default_value)

        @classmethod
        def get_non_optional_params(cls):
            if False:
                print('Hello World!')
            "\n            Fetches the non_optional_params from ``options['settings']``\n            and also normalizes their descriptions.\n\n            :return:\n                An OrderedDict that is used to create a\n                ``FunctionMetadata`` object.\n            "
            non_optional_params = {}
            for (setting_name, description) in options['settings'].items():
                if len(description) == 2:
                    non_optional_params[setting_name] = cls._normalize_desc(description[0], description[1])
            return OrderedDict(non_optional_params)

        @classmethod
        def get_optional_params(cls):
            if False:
                i = 10
                return i + 15
            "\n            Fetches the optional_params from ``options['settings']``\n            and also normalizes their descriptions.\n\n            :return:\n                An OrderedDict that is used to create a\n                ``FunctionMetadata`` object.\n            "
            optional_params = {}
            for (setting_name, description) in options['settings'].items():
                if len(description) == 3:
                    optional_params[setting_name] = cls._normalize_desc(description[0], description[1], description[2])
            return OrderedDict(optional_params)

        @classmethod
        def get_metadata(cls):
            if False:
                while True:
                    i = 10
            metadata = FunctionMetadata('run', optional_params=cls.get_optional_params(), non_optional_params=cls.get_non_optional_params())
            metadata.desc = inspect.getdoc(cls)
            return metadata

        @classmethod
        def _prepare_settings(cls, settings):
            if False:
                i = 10
                return i + 15
            '\n            Adds the optional settings to the settings dict in-place.\n\n            :param settings:\n                The settings dict.\n            '
            opt_params = cls.get_optional_params()
            for (setting_name, description) in opt_params.items():
                if setting_name not in settings:
                    settings[setting_name] = description[2]

        def parse_output(self, out, filename):
            if False:
                print('Hello World!')
            '\n            Parses the output JSON into Result objects.\n\n            :param out:\n                Raw output from the given executable (should be JSON).\n            :param filename:\n                The filename of the analyzed file. Needed to\n                create the Result objects.\n            :return:\n                An iterator yielding ``Result`` objects.\n            '
            output = json.loads(out)
            for result in output['results']:
                affected_code = tuple((SourceRange.from_values(code_range['file'], code_range['start']['line'], code_range['start'].get('column'), code_range.get('end', {}).get('line'), code_range.get('end', {}).get('column')) for code_range in result['affected_code']))
                yield Result(origin=result['origin'], message=result['message'], affected_code=affected_code, severity=result.get('severity', 1), debug_msg=result.get('debug_msg', ''), additional_info=result.get('additional_info', ''))

        def run(self, filename, file, **settings):
            if False:
                i = 10
                return i + 15
            self._prepare_settings(settings)
            json_string = json.dumps({'filename': filename, 'file': file, 'settings': settings})
            args = self.create_arguments()
            try:
                args = tuple(args)
            except TypeError:
                self.err('The given arguments {!r} are not iterable.'.format(args))
                return
            shell_command = (self.get_executable(),) + args
            (out, err) = run_shell_command(shell_command, json_string)
            return self.parse_output(out, filename)
    result_klass = type(klass.__name__, (klass, ExternalBearWrapBase), {})
    result_klass.__doc__ = klass.__doc__ or ''
    return result_klass

@enforce_signature
def external_bear_wrap(executable: str, **options):
    if False:
        while True:
            i = 10
    options['executable'] = executable
    _prepare_options(options)
    return partial(_create_wrapper, options=options)