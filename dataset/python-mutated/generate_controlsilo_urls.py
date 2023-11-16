from __future__ import annotations
import functools
import importlib
import re
from dataclasses import dataclass
from typing import Callable, List
from django.conf import settings
from django.core.exceptions import ViewDoesNotExist
from django.core.management import CommandError
from django.core.management.base import BaseCommand
from django.urls import URLPattern, URLResolver
from sentry.silo.base import SiloMode
IGNORED_PATTERN_NAMES = {'sentry-api-index', 'sentry-api-catchall'}

@dataclass
class PatternInfo:
    callable: Callable
    pattern: str
    name: str | None

    @property
    def url_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.name or ''

def describe_pattern(pattern):
    if False:
        i = 10
        return i + 15
    return str(pattern.pattern)
named_group_matcher = re.compile('\\(\\?P(<\\w+>)([^\\)]+)\\)')

def simplify_regex(pattern: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Convert python regex named capture groups into\n    simple patterns that will work with our javascript\n    code.\n    '
    pattern = pattern.replace('/^', '/')
    named_groups = [(m.start(0), m.end(0)) for m in named_group_matcher.finditer(pattern)]
    updated = pattern
    for (start, end) in reversed(named_groups):
        updated = updated[0:start] + '[^/]+' + updated[end:]
    return updated.replace('\\', '\\\\')

class Command(BaseCommand):
    help = 'Generate a list of URL patterns served by control silo endpoints'

    def add_arguments(self, parser):
        if False:
            return 10
        parser.add_argument('--format', choices=['text', 'js'], dest='format', default='text', help='The format of the file write to --output. Can either write data or JS code.')
        parser.add_argument('--output', dest='output', action='store', default=None, help='The file to write the generated pattern list to.')
        parser.add_argument('--urlconf', dest='urlconf', default='ROOT_URLCONF', help='The settings attribute with URL configuration.')

    def handle(self, **options):
        if False:
            i = 10
            return i + 15
        try:
            urlconf = importlib.import_module(getattr(settings, options['urlconf']))
        except ImportError as e:
            self.stdout.write(f'Failed to load URL configuration {e}')
            raise CommandError('Could not load URL configuration')
        view_functions = self.extract_views_from_urlpatterns(urlconf.urlpatterns)
        url_patterns = []
        for info in view_functions:
            func = info.callable
            if isinstance(func, functools.partial):
                func = func.func
            if hasattr(func, 'view_class'):
                func = func.view_class
            if hasattr(func, '__name__'):
                func_name = func.__name__
            elif hasattr(func, '__class__'):
                func_name = func.__class__.__name__
            else:
                func_name = re.sub(' at 0x[0-9a-f]+', '', repr(func))
            module = f'{func.__module__}.{func_name}'
            try:
                import_module = importlib.import_module(func.__module__)
                view_func = getattr(import_module, func_name)
            except AttributeError:
                continue
            except ImportError as err:
                raise CommandError(f'Could not load view in {module}: {err}')
            if not hasattr(view_func, 'silo_limit'):
                continue
            silo_limit = view_func.silo_limit
            if SiloMode.CONTROL not in silo_limit.modes:
                continue
            simple_pattern = simplify_regex(info.pattern)
            url_patterns.append(simple_pattern)
        contents = self.render(url_patterns, options['format'])
        if not options['output']:
            self.stdout.write(contents)
        else:
            self.stdout.write(f"Writing {options['output']} file..")
            with open(options['output'], 'w') as fh:
                fh.write(contents)
            self.stdout.write('All done')

    def render(self, url_patterns: List[str], format: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        if format == 'text':
            return '\n'.join(url_patterns)
        if format == 'js':
            js_regex = [f"new RegExp('{pattern}')," for pattern in url_patterns]
            pattern_code = '\n  '.join(js_regex)
            ts_code = f'// This is generated code.\n// To update it run `getsentry django generate_controlsilo_urls --format=js --output=/path/to/thisfile.ts`\nconst patterns: RegExp[] = [\n  {pattern_code}\n];\n\nexport default patterns;\n'
            return ts_code
        raise TypeError(f'Invalid format chosen {format}')

    def extract_views_from_urlpatterns(self, urlpatterns, base: str='', namespace: str | None=None) -> List[PatternInfo]:
        if False:
            i = 10
            return i + 15
        views = []
        for pat in urlpatterns:
            if isinstance(pat, URLPattern):
                try:
                    if not pat.name:
                        name = pat.name
                    elif namespace:
                        name = f'{namespace}:{pat.name}'
                    else:
                        name = pat.name
                    if name in IGNORED_PATTERN_NAMES:
                        continue
                    pattern = describe_pattern(pat)
                    views.append(PatternInfo(callable=pat.callback, pattern=base + pattern, name=name))
                except ViewDoesNotExist:
                    continue
            elif isinstance(pat, URLResolver):
                try:
                    patterns = pat.url_patterns
                except ImportError:
                    continue
                if namespace and pat.namespace:
                    resolved_namespace = f'{namespace}:{pat.namespace}'
                else:
                    resolved_namespace = pat.namespace or namespace or ''
                pattern = describe_pattern(pat)
                views.extend(self.extract_views_from_urlpatterns(patterns, base + pattern, namespace=resolved_namespace))
            elif hasattr(pat, '_get_callback'):
                try:
                    views.append(PatternInfo(callable=pat._get_callback(), pattern=base + describe_pattern(pat), name=pat.name))
                except ViewDoesNotExist:
                    continue
            elif hasattr(pat, 'url_patterns') or hasattr(pat, '_get_url_patterns'):
                try:
                    patterns = pat.url_patterns
                except ImportError:
                    continue
                views.extend(self.extract_views_from_urlpatterns(patterns, base=base + describe_pattern(pat), namespace=namespace))
            else:
                raise TypeError(f'{pat} is not a urlpattern object.')
        return views