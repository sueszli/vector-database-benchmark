"""
See https://zulip.readthedocs.io/en/latest/translating/internationalization.html
for background.

The contents of this file are taken from
https://github.com/niwinz/django-jinja/blob/master/django_jinja/management/commands/makemessages.py

Jinja2's i18n functionality is not exactly the same as Django's.
In particular, the tags names and their syntax are different:

  1. The Django ``trans`` tag is replaced by a _() global.
  2. The Django ``blocktrans`` tag is called ``trans``.

(1) isn't an issue, since the whole ``makemessages`` process is based on
converting the template tags to ``_()`` calls. However, (2) means that
those Jinja2 ``trans`` tags will not be picked up by Django's
``makemessages`` command.

There aren't any nice solutions here. While Jinja2's i18n extension does
come with extraction capabilities built in, the code behind ``makemessages``
unfortunately isn't extensible, so we can:

  * Duplicate the command + code behind it.
  * Offer a separate command for Jinja2 extraction.
  * Try to get Django to offer hooks into makemessages().
  * Monkey-patch.

We are currently doing that last thing. It turns out there we are lucky
for once: It's simply a matter of extending two regular expressions.
Credit for the approach goes to:
https://stackoverflow.com/questions/2090717

"""
import glob
import itertools
import json
import os
import re
import subprocess
from typing import Any, Collection, Dict, Iterator, List, Mapping
from django.core.management.base import CommandParser
from django.core.management.commands import makemessages
from django.template.base import BLOCK_TAG_END, BLOCK_TAG_START
from django.utils.translation import template
from typing_extensions import override
strip_whitespace_right = re.compile(f'({BLOCK_TAG_START}-?\\s*(trans|pluralize).*?-{BLOCK_TAG_END})\\s+')
strip_whitespace_left = re.compile(f'\\s+({BLOCK_TAG_START}-\\s*(endtrans|pluralize).*?-?{BLOCK_TAG_END})')
regexes = ['{{~?#tr}}([\\s\\S]*?)(?:~?{{/tr}}|{{#\\*inline )', '{{~?\\s*t "([\\s\\S]*?)"\\W*~?}}', "{{~?\\s*t '([\\s\\S]*?)'\\W*~?}}", '\\(t "([\\s\\S]*?)"\\)', '=\\(t "([\\s\\S]*?)"\\)(?=[^{]*}})', "=\\(t '([\\s\\S]*?)'\\)(?=[^{]*}})"]
tags = [('err_', 'error')]
frontend_compiled_regexes = [re.compile(regex) for regex in regexes]
multiline_js_comment = re.compile('/\\*.*?\\*/', re.DOTALL)
singleline_js_comment = re.compile('//.*?\n')

def strip_whitespaces(src: str) -> str:
    if False:
        print('Hello World!')
    src = strip_whitespace_left.sub('\\1', src)
    src = strip_whitespace_right.sub('\\1', src)
    return src

class Command(makemessages.Command):
    xgettext_options = makemessages.Command.xgettext_options
    for (func, tag) in tags:
        xgettext_options += [f'--keyword={func}:1,"{tag}"']

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        if False:
            i = 10
            return i + 15
        super().add_arguments(parser)
        parser.add_argument('--frontend-source', default='web/templates', help='Name of the Handlebars template directory')
        parser.add_argument('--frontend-output', default='locale', help='Name of the frontend messages output directory')
        parser.add_argument('--frontend-namespace', default='translations.json', help='Namespace of the frontend locale file')

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            print('Hello World!')
        self.handle_django_locales(*args, **options)
        self.handle_frontend_locales(**options)

    def handle_frontend_locales(self, *, frontend_source: str, frontend_output: str, frontend_namespace: str, locale: List[str], exclude: List[str], all: bool, **options: Any) -> None:
        if False:
            i = 10
            return i + 15
        self.frontend_source = frontend_source
        self.frontend_output = frontend_output
        self.frontend_namespace = frontend_namespace
        self.frontend_locale = locale
        self.frontend_exclude = exclude
        self.frontend_all = all
        translation_strings = self.get_translation_strings()
        self.write_translation_strings(translation_strings)

    def handle_django_locales(self, *args: Any, **options: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        old_endblock_re = template.endblock_re
        old_block_re = template.block_re
        old_constant_re = template.constant_re
        old_templatize = template.templatize
        template.endblock_re = re.compile(template.endblock_re.pattern + '|' + '^-?\\s*endtrans\\s*-?$')
        template.block_re = re.compile(template.block_re.pattern + '|' + '^-?\\s*trans(?:\\s+(?!\'|")(?=.*?=.*?)|\\s*-?$)')
        template.plural_re = re.compile(template.plural_re.pattern + '|' + '^-?\\s*pluralize(?:\\s+.+|-?$)')
        template.constant_re = re.compile('_\\(((?:".*?")|(?:\'.*?\')).*\\)')

        def my_templatize(src: str, *args: Any, **kwargs: Any) -> str:
            if False:
                while True:
                    i = 10
            new_src = strip_whitespaces(src)
            return old_templatize(new_src, *args, **kwargs)
        template.templatize = my_templatize
        try:
            ignore_patterns = options.get('ignore_patterns', [])
            ignore_patterns.append('docs/*')
            ignore_patterns.append('templates/zerver/emails/custom/*')
            ignore_patterns.append('var/*')
            options['ignore_patterns'] = ignore_patterns
            super().handle(*args, **options)
        finally:
            template.endblock_re = old_endblock_re
            template.block_re = old_block_re
            template.templatize = old_templatize
            template.constant_re = old_constant_re

    def extract_strings(self, data: str) -> List[str]:
        if False:
            return 10
        translation_strings: List[str] = []
        for regex in frontend_compiled_regexes:
            for match in regex.findall(data):
                match = match.strip()
                match = ' '.join((line.strip() for line in match.splitlines()))
                translation_strings.append(match)
        return translation_strings

    def ignore_javascript_comments(self, data: str) -> str:
        if False:
            print('Hello World!')
        data = multiline_js_comment.sub('', data)
        data = singleline_js_comment.sub('', data)
        return data

    def get_translation_strings(self) -> List[str]:
        if False:
            return 10
        translation_strings: List[str] = []
        dirname = self.get_template_dir()
        for (dirpath, dirnames, filenames) in os.walk(dirname):
            for filename in [f for f in filenames if f.endswith('.hbs')]:
                if filename.startswith('.'):
                    continue
                with open(os.path.join(dirpath, filename)) as reader:
                    data = reader.read()
                    translation_strings.extend(self.extract_strings(data))
        for (dirpath, dirnames, filenames) in itertools.chain(os.walk('web/src'), os.walk('web/shared/src')):
            for filename in [f for f in filenames if f.endswith(('.js', '.ts'))]:
                if filename.startswith('.'):
                    continue
                with open(os.path.join(dirpath, filename)) as reader:
                    data = reader.read()
                    data = self.ignore_javascript_comments(data)
                    translation_strings.extend(self.extract_strings(data))
        extracted = subprocess.check_output(['node_modules/.bin/formatjs', 'extract', '--additional-function-names=$t,$t_html', '--format=simple', '--ignore=**/*.d.ts', 'web/src/**/*.js', 'web/src/**/*.ts'])
        translation_strings.extend(json.loads(extracted).values())
        return list(set(translation_strings))

    def get_template_dir(self) -> str:
        if False:
            while True:
                i = 10
        return self.frontend_source

    def get_namespace(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.frontend_namespace

    def get_locales(self) -> Collection[str]:
        if False:
            i = 10
            return i + 15
        locale = self.frontend_locale
        exclude = self.frontend_exclude
        process_all = self.frontend_all
        default_locale_path = self.default_locale_path
        paths = glob.glob(f'{default_locale_path}/*')
        all_locales = [os.path.basename(path) for path in paths if os.path.isdir(path)]
        if process_all:
            return all_locales
        else:
            locales = locale or all_locales
            return set(locales) - set(exclude)

    def get_base_path(self) -> str:
        if False:
            print('Hello World!')
        return self.frontend_output

    def get_output_paths(self) -> Iterator[str]:
        if False:
            return 10
        base_path = self.get_base_path()
        locales = self.get_locales()
        for path in [os.path.join(base_path, locale) for locale in locales]:
            if not os.path.exists(path):
                os.makedirs(path)
            yield os.path.join(path, self.get_namespace())

    def get_new_strings(self, old_strings: Mapping[str, str], translation_strings: List[str], locale: str) -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        '\n        Missing strings are removed, new strings are added and already\n        translated strings are not touched.\n        '
        new_strings = {}
        for k in translation_strings:
            if locale == 'en':
                new_strings[k] = old_strings.get(k, k)
            else:
                new_strings[k] = old_strings.get(k, '')
        return new_strings

    def write_translation_strings(self, translation_strings: List[str]) -> None:
        if False:
            print('Hello World!')
        for (locale, output_path) in zip(self.get_locales(), self.get_output_paths()):
            self.stdout.write(f'[frontend] processing locale {locale}')
            try:
                with open(output_path) as reader:
                    old_strings = json.load(reader)
            except (OSError, ValueError):
                old_strings = {}
            new_strings = self.get_new_strings(old_strings, translation_strings, locale)
            with open(output_path, 'w') as writer:
                json.dump(new_strings, writer, indent=2, sort_keys=True)