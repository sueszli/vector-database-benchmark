import inspect
import logging
import re
from collections import namedtuple
from typing import Pattern
from faker import Faker
from faker.config import AVAILABLE_LOCALES, DEFAULT_LOCALE
from faker.sphinx.validator import SampleCodeValidator
logger = logging.getLogger(__name__)
_fake = Faker(AVAILABLE_LOCALES)
_base_provider_method_pattern: Pattern = re.compile('^faker\\.providers\\.BaseProvider\\.(?P<method>\\w+)$')
_standard_provider_method_pattern: Pattern = re.compile('^faker\\.providers\\.\\w+\\.Provider\\.(?P<method>\\w+)$')
_locale_provider_method_pattern: Pattern = re.compile('^faker\\.providers\\.\\w+\\.(?P<locale>[a-z]{2,3}_[A-Z]{2})\\.Provider\\.(?P<method>\\w+)$')
_sample_line_pattern: Pattern = re.compile('^:sample(?: size=(?P<size>[1-9][0-9]*))?(?: seed=(?P<seed>[0-9]+))?:(?: ?(?P<kwargs>.*))?$')
_command_template = 'generator.{method}({kwargs})'
_sample_output_template = '>>> Faker.seed({seed})\n>>> for _ in range({size}):\n...     fake.{method}({kwargs})\n...\n{results}\n\n'
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_SEED = 0
Sample = namedtuple('Sample', ['size', 'seed', 'kwargs'])

class ProviderMethodDocstring:
    """
    Class that preprocesses provider method docstrings to generate sample usage and output

    Notes on how samples are generated:
    - If the docstring belongs to a standard provider method, sample usage and output will be
      generated using a `Faker` object in the `DEFAULT_LOCALE`.
    - If the docstring belongs to a localized provider method, the correct locale will be used.
    - If the docstring does not belong to any provider method, docstring preprocessing will be skipped.
    - Docstring lines will be parsed for potential sample sections, and the generation details of each
      sample section will internally be represented as a ``Sample`` namedtuple.
    - Each ``Sample`` will have info on the keyword arguments to pass to the provider method, how many
      times the provider method will be called, and the initial seed value to ``Faker.seed()``.
    """

    def __init__(self, app, what, name, obj, options, lines):
        if False:
            return 10
        self._line_iter = iter(lines)
        self._parsed_lines = []
        self._samples = []
        self._skipped = True
        self._log_prefix = f'{inspect.getfile(obj)}:docstring of {name}: WARNING:'
        if what != 'method':
            return
        base_provider_method_match = _base_provider_method_pattern.match(name)
        locale_provider_method_match = _locale_provider_method_pattern.match(name)
        standard_provider_method_match = _standard_provider_method_pattern.match(name)
        if base_provider_method_match:
            groupdict = base_provider_method_match.groupdict()
            self._method = groupdict['method']
            self._locale = DEFAULT_LOCALE
        elif standard_provider_method_match:
            groupdict = standard_provider_method_match.groupdict()
            self._method = groupdict['method']
            self._locale = DEFAULT_LOCALE
        elif locale_provider_method_match:
            groupdict = locale_provider_method_match.groupdict()
            self._method = groupdict['method']
            self._locale = groupdict['locale']
        else:
            return
        self._skipped = False
        self._parse()
        self._generate_samples()

    def _log_warning(self, warning):
        if False:
            print('Hello World!')
        logger.warning(f'{self._log_prefix} {warning}')

    def _parse(self):
        if False:
            i = 10
            return i + 15
        while True:
            try:
                line = next(self._line_iter)
            except StopIteration:
                break
            else:
                self._parse_section(line)

    def _parse_section(self, section):
        if False:
            for i in range(10):
                print('nop')
        if not section.startswith(':sample'):
            self._parsed_lines.append(section)
            return
        try:
            next_line = next(self._line_iter)
        except StopIteration:
            self._process_sample_section(section)
            return
        if next_line.startswith(':sample'):
            self._process_sample_section(section)
            self._parse_section(next_line)
        elif next_line == '':
            self._process_sample_section(section)
        else:
            section = section + next_line
            self._parse_section(section)

    def _process_sample_section(self, section):
        if False:
            for i in range(10):
                print('nop')
        match = _sample_line_pattern.match(section)
        if not match:
            msg = f'The section `{section}` is malformed and will be discarded.'
            self._log_warning(msg)
            return
        groupdict = match.groupdict()
        size = groupdict.get('size')
        seed = groupdict.get('seed')
        kwargs = groupdict.get('kwargs')
        size = max(int(size), DEFAULT_SAMPLE_SIZE) if size else DEFAULT_SAMPLE_SIZE
        seed = int(seed) if seed else DEFAULT_SEED
        kwargs = self._beautify_kwargs(kwargs) if kwargs else ''
        sample = Sample(size, seed, kwargs)
        self._samples.append(sample)

    def _beautify_kwargs(self, kwargs):
        if False:
            for i in range(10):
                print('nop')

        def _repl_whitespace(match):
            if False:
                while True:
                    i = 10
            quoted = match.group(1) or match.group(2)
            return quoted if quoted else ''

        def _repl_comma(match):
            if False:
                i = 10
                return i + 15
            quoted = match.group(1) or match.group(2)
            return quoted if quoted else ', '
        result = re.sub('("[^"]*")|(\\\'[^\\\']*\\\')|[ \\t]+', _repl_whitespace, kwargs)
        result = re.sub('("[^"]*")|(\\\'[^\\\']*\\\')|,', _repl_comma, result)
        return result.strip()

    def _stringify_result(self, value):
        if False:
            print('Hello World!')
        return repr(value)

    def _generate_eval_scope(self):
        if False:
            for i in range(10):
                print('nop')
        from collections import OrderedDict
        return {'generator': _fake[self._locale], 'OrderedDict': OrderedDict}

    def _inject_default_sample_section(self):
        if False:
            print('Hello World!')
        default_sample = Sample(DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, '')
        self._samples.append(default_sample)

    def _generate_samples(self):
        if False:
            i = 10
            return i + 15
        if not self._samples:
            self._inject_default_sample_section()
        output = ''
        eval_scope = self._generate_eval_scope()
        for sample in self._samples:
            command = _command_template.format(method=self._method, kwargs=sample.kwargs)
            validator = SampleCodeValidator(command)
            if validator.errors:
                msg = f'Invalid code elements detected. Sample generation will be skipped for method `{self._method}` with arguments `{sample.kwargs}`.'
                self._log_warning(msg)
                continue
            try:
                Faker.seed(sample.seed)
                results = '\n'.join([self._stringify_result(eval(command, eval_scope)) for _ in range(sample.size)])
            except Exception:
                msg = f'Sample generation failed for method `{self._method}` with arguments `{sample.kwargs}`.'
                self._log_warning(msg)
                continue
            else:
                output += _sample_output_template.format(seed=sample.seed, method=self._method, kwargs=sample.kwargs, size=sample.size, results=results)
        if output:
            output = ':examples:\n\n' + output
            self._parsed_lines.extend(output.split('\n'))

    @property
    def skipped(self):
        if False:
            print('Hello World!')
        return self._skipped

    @property
    def lines(self):
        if False:
            for i in range(10):
                print('nop')
        return self._parsed_lines