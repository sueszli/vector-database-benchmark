import logging
import math
import re
import shlex
from .exceptions import PapermillException
from .models import Parameter
logger = logging.getLogger(__name__)

class PapermillTranslators:
    """
    The holder which houses any translator registered with the system.
    This object is used in a singleton manner to save and load particular
    named Translator objects for reference externally.
    """

    def __init__(self):
        if False:
            return 10
        self._translators = {}

    def register(self, language, translator):
        if False:
            print('Hello World!')
        self._translators[language] = translator

    def find_translator(self, kernel_name, language):
        if False:
            return 10
        if kernel_name in self._translators:
            return self._translators[kernel_name]
        elif language in self._translators:
            return self._translators[language]
        raise PapermillException("No parameter translator functions specified for kernel '{}' or language '{}'".format(kernel_name, language))

class Translator:

    @classmethod
    def translate_raw_str(cls, val):
        if False:
            i = 10
            return i + 15
        'Reusable by most interpreters'
        return f'{val}'

    @classmethod
    def translate_escaped_str(cls, str_val):
        if False:
            print('Hello World!')
        'Reusable by most interpreters'
        if isinstance(str_val, str):
            str_val = str_val.encode('unicode_escape')
            str_val = str_val.decode('utf-8')
            str_val = str_val.replace('"', '\\"')
        return f'"{str_val}"'

    @classmethod
    def translate_str(cls, val):
        if False:
            for i in range(10):
                print('nop')
        'Default behavior for translation'
        return cls.translate_escaped_str(val)

    @classmethod
    def translate_none(cls, val):
        if False:
            print('Hello World!')
        'Default behavior for translation'
        return cls.translate_raw_str(val)

    @classmethod
    def translate_int(cls, val):
        if False:
            return 10
        'Default behavior for translation'
        return cls.translate_raw_str(val)

    @classmethod
    def translate_float(cls, val):
        if False:
            return 10
        'Default behavior for translation'
        return cls.translate_raw_str(val)

    @classmethod
    def translate_bool(cls, val):
        if False:
            for i in range(10):
                print('nop')
        'Default behavior for translation'
        return 'true' if val else 'false'

    @classmethod
    def translate_dict(cls, val):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'dict type translation not implemented for {cls}')

    @classmethod
    def translate_list(cls, val):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'list type translation not implemented for {cls}')

    @classmethod
    def translate(cls, val):
        if False:
            return 10
        'Translate each of the standard json/yaml types to appropiate objects.'
        if val is None:
            return cls.translate_none(val)
        elif isinstance(val, str):
            return cls.translate_str(val)
        elif isinstance(val, bool):
            return cls.translate_bool(val)
        elif isinstance(val, int):
            return cls.translate_int(val)
        elif isinstance(val, float):
            return cls.translate_float(val)
        elif isinstance(val, dict):
            return cls.translate_dict(val)
        elif isinstance(val, list):
            return cls.translate_list(val)
        return cls.translate_escaped_str(val)

    @classmethod
    def comment(cls, cmt_str):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'comment translation not implemented for {cls}')

    @classmethod
    def assign(cls, name, str_val):
        if False:
            for i in range(10):
                print('nop')
        return f'{name} = {str_val}'

    @classmethod
    def codify(cls, parameters, comment='Parameters'):
        if False:
            print('Hello World!')
        content = f'{cls.comment(comment)}\n'
        for (name, val) in parameters.items():
            content += f'{cls.assign(name, cls.translate(val))}\n'
        return content

    @classmethod
    def inspect(cls, parameters_cell):
        if False:
            while True:
                i = 10
        'Inspect the parameters cell to get a Parameter list\n\n        It must return an empty list if no parameters are found and\n        it should ignore inspection errors.\n\n        .. note::\n            ``inferred_type_name`` should be "None" if unknown (set it\n            to "NoneType" for null value)\n\n        Parameters\n        ----------\n        parameters_cell : NotebookNode\n            Cell tagged _parameters_\n\n        Returns\n        -------\n        List[Parameter]\n            A list of all parameters\n        '
        raise NotImplementedError(f'parameters introspection not implemented for {cls}')

class PythonTranslator(Translator):
    PARAMETER_PATTERN = re.compile('^(?P<target>\\w[\\w_]*)\\s*(:\\s*[\\"\']?(?P<annotation>\\w[\\w_\\[\\],\\s]*)[\\"\']?\\s*)?=\\s*(?P<value>.*?)(\\s*#\\s*(type:\\s*(?P<type_comment>[^\\s]*)\\s*)?(?P<help>.*))?$')

    @classmethod
    def translate_float(cls, val):
        if False:
            print('Hello World!')
        if math.isfinite(val):
            return cls.translate_raw_str(val)
        elif math.isnan(val):
            return "float('nan')"
        elif val < 0:
            return "float('-inf')"
        else:
            return "float('inf')"

    @classmethod
    def translate_bool(cls, val):
        if False:
            for i in range(10):
                print('nop')
        return cls.translate_raw_str(val)

    @classmethod
    def translate_dict(cls, val):
        if False:
            i = 10
            return i + 15
        escaped = ', '.join([f'{cls.translate_str(k)}: {cls.translate(v)}' for (k, v) in val.items()])
        return f'{{{escaped}}}'

    @classmethod
    def translate_list(cls, val):
        if False:
            i = 10
            return i + 15
        escaped = ', '.join([cls.translate(v) for v in val])
        return f'[{escaped}]'

    @classmethod
    def comment(cls, cmt_str):
        if False:
            print('Hello World!')
        return f'# {cmt_str}'.strip()

    @classmethod
    def codify(cls, parameters, comment='Parameters'):
        if False:
            while True:
                i = 10
        content = super().codify(parameters, comment)
        try:
            import black
            fm = black.FileMode(string_normalization=False)
            content = black.format_str(content, mode=fm)
        except ImportError:
            logger.debug("Black is not installed, parameters won't be formatted")
        except AttributeError as aerr:
            logger.warning(f'Black encountered an error, skipping formatting ({aerr})')
        return content

    @classmethod
    def inspect(cls, parameters_cell):
        if False:
            while True:
                i = 10
        'Inspect the parameters cell to get a Parameter list\n\n        It must return an empty list if no parameters are found and\n        it should ignore inspection errors.\n\n        Parameters\n        ----------\n        parameters_cell : NotebookNode\n            Cell tagged _parameters_\n\n        Returns\n        -------\n        List[Parameter]\n            A list of all parameters\n        '
        params = []
        src = parameters_cell['source']

        def flatten_accumulator(accumulator):
            if False:
                print('Hello World!')
            'Flatten a multilines variable definition.\n\n            Remove all comments except on the latest line - will be interpreted as help.\n\n            Args:\n                accumulator (List[str]): Line composing the variable definition\n            Returns:\n                Flatten definition\n            '
            flat_string = ''
            for line in accumulator[:-1]:
                if '#' in line:
                    comment_pos = line.index('#')
                    flat_string += line[:comment_pos].strip()
                else:
                    flat_string += line.strip()
            if len(accumulator):
                flat_string += accumulator[-1].strip()
            return flat_string
        grouped_variable = []
        accumulator = []
        for (iline, line) in enumerate(src.splitlines()):
            if len(line.strip()) == 0 or line.strip().startswith('#'):
                continue
            nequal = line.count('=')
            if nequal > 0:
                grouped_variable.append(flatten_accumulator(accumulator))
                accumulator = []
                if nequal > 1:
                    logger.warning(f"Unable to parse line {iline + 1} '{line}'.")
                    continue
            accumulator.append(line)
        grouped_variable.append(flatten_accumulator(accumulator))
        for definition in grouped_variable:
            if len(definition) == 0:
                continue
            match = re.match(cls.PARAMETER_PATTERN, definition)
            if match is not None:
                attr = match.groupdict()
                if attr['target'] is None:
                    continue
                type_name = str(attr['annotation'] or attr['type_comment'] or None)
                params.append(Parameter(name=attr['target'].strip(), inferred_type_name=type_name.strip(), default=str(attr['value']).strip(), help=str(attr['help'] or '').strip()))
        return params

class RTranslator(Translator):

    @classmethod
    def translate_none(cls, val):
        if False:
            for i in range(10):
                print('nop')
        return 'NULL'

    @classmethod
    def translate_bool(cls, val):
        if False:
            return 10
        return 'TRUE' if val else 'FALSE'

    @classmethod
    def translate_dict(cls, val):
        if False:
            while True:
                i = 10
        escaped = ', '.join([f'{cls.translate_str(k)} = {cls.translate(v)}' for (k, v) in val.items()])
        return f'list({escaped})'

    @classmethod
    def translate_list(cls, val):
        if False:
            i = 10
            return i + 15
        escaped = ', '.join([cls.translate(v) for v in val])
        return f'list({escaped})'

    @classmethod
    def comment(cls, cmt_str):
        if False:
            return 10
        return f'# {cmt_str}'.strip()

    @classmethod
    def assign(cls, name, str_val):
        if False:
            return 10
        while name.startswith('_'):
            name = name[1:]
        return f'{name} = {str_val}'

class ScalaTranslator(Translator):

    @classmethod
    def translate_int(cls, val):
        if False:
            for i in range(10):
                print('nop')
        strval = cls.translate_raw_str(val)
        return strval + 'L' if val > 2147483647 or val < -2147483648 else strval

    @classmethod
    def translate_dict(cls, val):
        if False:
            for i in range(10):
                print('nop')
        'Translate dicts to scala Maps'
        escaped = ', '.join([f'{cls.translate_str(k)} -> {cls.translate(v)}' for (k, v) in val.items()])
        return f'Map({escaped})'

    @classmethod
    def translate_list(cls, val):
        if False:
            for i in range(10):
                print('nop')
        'Translate list to scala Seq'
        escaped = ', '.join([cls.translate(v) for v in val])
        return f'Seq({escaped})'

    @classmethod
    def comment(cls, cmt_str):
        if False:
            while True:
                i = 10
        return f'// {cmt_str}'.strip()

    @classmethod
    def assign(cls, name, str_val):
        if False:
            return 10
        return f'val {name} = {str_val}'

class JuliaTranslator(Translator):

    @classmethod
    def translate_none(cls, val):
        if False:
            for i in range(10):
                print('nop')
        return 'nothing'

    @classmethod
    def translate_dict(cls, val):
        if False:
            i = 10
            return i + 15
        escaped = ', '.join([f'{cls.translate_str(k)} => {cls.translate(v)}' for (k, v) in val.items()])
        return f'Dict({escaped})'

    @classmethod
    def translate_list(cls, val):
        if False:
            while True:
                i = 10
        escaped = ', '.join([cls.translate(v) for v in val])
        return f'[{escaped}]'

    @classmethod
    def comment(cls, cmt_str):
        if False:
            for i in range(10):
                print('nop')
        return f'# {cmt_str}'.strip()

class MatlabTranslator(Translator):

    @classmethod
    def translate_escaped_str(cls, str_val):
        if False:
            print('Hello World!')
        'Translate a string to an escaped Matlab string'
        if isinstance(str_val, str):
            str_val = str_val.encode('unicode_escape')
            str_val = str_val.decode('utf-8')
            str_val = str_val.replace('"', '""')
        return f'"{str_val}"'

    @staticmethod
    def __translate_char_array(str_val):
        if False:
            i = 10
            return i + 15
        'Translates a string to a Matlab char array'
        if isinstance(str_val, str):
            str_val = str_val.encode('unicode_escape')
            str_val = str_val.decode('utf-8')
            str_val = str_val.replace("'", "''")
        return f"'{str_val}'"

    @classmethod
    def translate_none(cls, val):
        if False:
            for i in range(10):
                print('nop')
        return 'NaN'

    @classmethod
    def translate_dict(cls, val):
        if False:
            return 10
        keys = ', '.join([f'{cls.__translate_char_array(k)}' for (k, v) in val.items()])
        vals = ', '.join([f'{cls.translate(v)}' for (k, v) in val.items()])
        return f'containers.Map({{{keys}}}, {{{vals}}})'

    @classmethod
    def translate_list(cls, val):
        if False:
            print('Hello World!')
        escaped = ', '.join([cls.translate(v) for v in val])
        return f'{{{escaped}}}'

    @classmethod
    def comment(cls, cmt_str):
        if False:
            for i in range(10):
                print('nop')
        return f'% {cmt_str}'.strip()

    @classmethod
    def codify(cls, parameters, comment='Parameters'):
        if False:
            return 10
        content = f'{cls.comment(comment)}\n'
        for (name, val) in parameters.items():
            content += f'{cls.assign(name, cls.translate(val))};\n'
        return content

class CSharpTranslator(Translator):

    @classmethod
    def translate_none(cls, val):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Option type not implemented for C#.')

    @classmethod
    def translate_bool(cls, val):
        if False:
            print('Hello World!')
        return 'true' if val else 'false'

    @classmethod
    def translate_int(cls, val):
        if False:
            while True:
                i = 10
        strval = cls.translate_raw_str(val)
        return strval + 'L' if val > 2147483647 or val < -2147483648 else strval

    @classmethod
    def translate_dict(cls, val):
        if False:
            return 10
        'Translate dicts to nontyped dictionary'
        kvps = ', '.join([f'{{ {cls.translate_str(k)} , {cls.translate(v)} }}' for (k, v) in val.items()])
        return f'new Dictionary<string,Object>{{ {kvps} }}'

    @classmethod
    def translate_list(cls, val):
        if False:
            i = 10
            return i + 15
        'Translate list to array'
        escaped = ', '.join([cls.translate(v) for v in val])
        return f'new [] {{ {escaped} }}'

    @classmethod
    def comment(cls, cmt_str):
        if False:
            return 10
        return f'// {cmt_str}'.strip()

    @classmethod
    def assign(cls, name, str_val):
        if False:
            return 10
        return f'var {name} = {str_val};'

class FSharpTranslator(Translator):

    @classmethod
    def translate_none(cls, val):
        if False:
            for i in range(10):
                print('nop')
        return 'None'

    @classmethod
    def translate_bool(cls, val):
        if False:
            for i in range(10):
                print('nop')
        return 'true' if val else 'false'

    @classmethod
    def translate_int(cls, val):
        if False:
            for i in range(10):
                print('nop')
        strval = cls.translate_raw_str(val)
        return strval + 'L' if val > 2147483647 or val < -2147483648 else strval

    @classmethod
    def translate_dict(cls, val):
        if False:
            for i in range(10):
                print('nop')
        tuples = '; '.join([f'({cls.translate_str(k)}, {cls.translate(v)} :> IComparable)' for (k, v) in val.items()])
        return f'[ {tuples} ] |> Map.ofList'

    @classmethod
    def translate_list(cls, val):
        if False:
            print('Hello World!')
        escaped = '; '.join([cls.translate(v) for v in val])
        return f'[ {escaped} ]'

    @classmethod
    def comment(cls, cmt_str):
        if False:
            return 10
        return f'(* {cmt_str} *)'.strip()

    @classmethod
    def assign(cls, name, str_val):
        if False:
            return 10
        return f'let {name} = {str_val}'

class PowershellTranslator(Translator):

    @classmethod
    def translate_escaped_str(cls, str_val):
        if False:
            return 10
        'Translate a string to an escaped Matlab string'
        if isinstance(str_val, str):
            str_val = str_val.encode('unicode_escape')
            str_val = str_val.decode('utf-8')
            str_val = str_val.replace('"', '`"')
        return f'"{str_val}"'

    @classmethod
    def translate_float(cls, val):
        if False:
            return 10
        if math.isfinite(val):
            return cls.translate_raw_str(val)
        elif math.isnan(val):
            return '[double]::NaN'
        elif val < 0:
            return '[double]::NegativeInfinity'
        else:
            return '[double]::PositiveInfinity'

    @classmethod
    def translate_none(cls, val):
        if False:
            for i in range(10):
                print('nop')
        return '$Null'

    @classmethod
    def translate_bool(cls, val):
        if False:
            return 10
        return '$True' if val else '$False'

    @classmethod
    def translate_dict(cls, val):
        if False:
            while True:
                i = 10
        kvps = '\n '.join([f'{cls.translate_str(k)} = {cls.translate(v)}' for (k, v) in val.items()])
        return f'@{{{kvps}}}'

    @classmethod
    def translate_list(cls, val):
        if False:
            print('Hello World!')
        escaped = ', '.join([cls.translate(v) for v in val])
        return f'@({escaped})'

    @classmethod
    def comment(cls, cmt_str):
        if False:
            print('Hello World!')
        return f'# {cmt_str}'.strip()

    @classmethod
    def assign(cls, name, str_val):
        if False:
            i = 10
            return i + 15
        return f'${name} = {str_val}'

class BashTranslator(Translator):

    @classmethod
    def translate_none(cls, val):
        if False:
            print('Hello World!')
        return ''

    @classmethod
    def translate_bool(cls, val):
        if False:
            while True:
                i = 10
        return 'true' if val else 'false'

    @classmethod
    def translate_escaped_str(cls, str_val):
        if False:
            for i in range(10):
                print('nop')
        return shlex.quote(str(str_val))

    @classmethod
    def translate_list(cls, val):
        if False:
            for i in range(10):
                print('nop')
        escaped = ' '.join([cls.translate(v) for v in val])
        return f'({escaped})'

    @classmethod
    def comment(cls, cmt_str):
        if False:
            for i in range(10):
                print('nop')
        return f'# {cmt_str}'.strip()

    @classmethod
    def assign(cls, name, str_val):
        if False:
            return 10
        return f'{name}={str_val}'
papermill_translators = PapermillTranslators()
papermill_translators.register('python', PythonTranslator)
papermill_translators.register('R', RTranslator)
papermill_translators.register('scala', ScalaTranslator)
papermill_translators.register('julia', JuliaTranslator)
papermill_translators.register('matlab', MatlabTranslator)
papermill_translators.register('.net-csharp', CSharpTranslator)
papermill_translators.register('.net-fsharp', FSharpTranslator)
papermill_translators.register('.net-powershell', PowershellTranslator)
papermill_translators.register('pysparkkernel', PythonTranslator)
papermill_translators.register('sparkkernel', ScalaTranslator)
papermill_translators.register('sparkrkernel', RTranslator)
papermill_translators.register('bash', BashTranslator)

def translate_parameters(kernel_name, language, parameters, comment='Parameters'):
    if False:
        i = 10
        return i + 15
    return papermill_translators.find_translator(kernel_name, language).codify(parameters, comment)