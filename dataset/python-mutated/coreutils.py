import functools
import inspect
import logging
import re
import unicodedata
from hashlib import md5
from typing import TYPE_CHECKING, Any, Dict, Iterable, Union
from warnings import warn
from anyascii import anyascii
from django.apps import apps
from django.conf import settings
from django.conf.locale import LANG_INFO
from django.core.cache import cache
from django.core.cache.utils import make_template_fragment_key
from django.core.exceptions import ImproperlyConfigured, SuspiciousOperation
from django.core.signals import setting_changed
from django.db.models import Model
from django.db.models.base import ModelBase
from django.dispatch import receiver
from django.http import HttpRequest
from django.test import RequestFactory
from django.utils.encoding import force_str
from django.utils.text import capfirst, slugify
from django.utils.translation import check_for_language, get_supported_language_variant
from django.utils.translation import gettext_lazy as _
from wagtail.utils.deprecation import RemovedInWagtail70Warning
if TYPE_CHECKING:
    from wagtail.models import Site
logger = logging.getLogger(__name__)
WAGTAIL_APPEND_SLASH = getattr(settings, 'WAGTAIL_APPEND_SLASH', True)

def camelcase_to_underscore(str):
    if False:
        while True:
            i = 10
    return re.sub('(((?<=[a-z])[A-Z])|([A-Z](?![A-Z]|$)))', '_\\1', str).lower().strip('_')

def string_to_ascii(value):
    if False:
        print('Hello World!')
    '\n    Convert a string to ascii.\n    '
    return str(anyascii(value))

def get_model_string(model):
    if False:
        i = 10
        return i + 15
    '\n    Returns a string that can be used to identify the specified model.\n\n    The format is: `app_label.ModelName`\n\n    This an be reversed with the `resolve_model_string` function\n    '
    return model._meta.app_label + '.' + model.__name__

def resolve_model_string(model_string, default_app=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Resolve an 'app_label.model_name' string into an actual model class.\n    If a model class is passed in, just return that.\n\n    Raises a LookupError if a model can not be found, or ValueError if passed\n    something that is neither a model or a string.\n    "
    if isinstance(model_string, str):
        try:
            (app_label, model_name) = model_string.split('.')
        except ValueError:
            if default_app is not None:
                app_label = default_app
                model_name = model_string
            else:
                raise ValueError('Can not resolve {!r} into a model. Model names should be in the form app_label.model_name'.format(model_string), model_string)
        return apps.get_model(app_label, model_name)
    elif isinstance(model_string, type):
        return model_string
    else:
        raise ValueError(f'Can not resolve {model_string!r} into a model', model_string)
SCRIPT_RE = re.compile('<(-*)/script>')

def escape_script(text):
    if False:
        return 10
    "\n    Escape `</script>` tags in 'text' so that it can be placed within a `<script>` block without\n    accidentally closing it. A '-' character will be inserted for each time it is escaped:\n    `<-/script>`, `<--/script>` etc.\n    "
    warn('The `escape_script` hook is deprecated - use `template` elements instead.', category=RemovedInWagtail70Warning)
    return SCRIPT_RE.sub('<-\\1/script>', text)
SLUGIFY_RE = re.compile('[^\\w\\s-]', re.UNICODE)

def cautious_slugify(value):
    if False:
        i = 10
        return i + 15
    "\n    Convert a string to ASCII exactly as Django's slugify does, with the exception\n    that any non-ASCII alphanumeric characters (that cannot be ASCIIfied under Unicode\n    normalisation) are escaped into codes like 'u0421' instead of being deleted entirely.\n\n    This ensures that the result of slugifying (for example - Cyrillic) text will not be an empty\n    string, and can thus be safely used as an identifier (albeit not a human-readable one).\n    "
    value = force_str(value)
    value = unicodedata.normalize('NFKD', value)
    value = SLUGIFY_RE.sub('', value)
    value = value.encode('ascii', 'backslashreplace').decode('ascii')
    return slugify(value)

def safe_snake_case(value):
    if False:
        return 10
    "\n    Convert a string to ASCII similar to Django's slugify, with catious handling of\n    non-ASCII alphanumeric characters. See `cautious_slugify`.\n\n    Any inner whitespace, hyphens or dashes will be converted to underscores and\n    will be safe for Django template or filename usage.\n    "
    slugified_ascii_string = cautious_slugify(value)
    snake_case_string = slugified_ascii_string.replace('-', '_')
    return snake_case_string

def get_content_type_label(content_type):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a human-readable label for a content type object, suitable for display in the admin\n    in place of the default 'wagtailcore | page' representation\n    "
    if content_type is None:
        return _('Unknown content type')
    model = content_type.model_class()
    if model:
        return str(capfirst(model._meta.verbose_name))
    else:
        return capfirst(content_type.model)

def accepts_kwarg(func, kwarg):
    if False:
        print('Hello World!')
    '\n    Determine whether the callable `func` has a signature that accepts the keyword argument `kwarg`\n    '
    signature = inspect.signature(func)
    try:
        signature.bind_partial(**{kwarg: None})
        return True
    except TypeError:
        return False

class InvokeViaAttributeShortcut:
    """
    Used to create a shortcut that allows an object's named
    single-argument method to be invoked using a simple
    attribute reference syntax. For example, adding the
    following to an object:

    obj.page_url = InvokeViaAttributeShortcut(obj, 'get_page_url')

    Would allow you to invoke get_page_url() like so:

    obj.page_url.terms_and_conditions

    As well as the usual:

    obj.get_page_url('terms_and_conditions')
    """
    __slots__ = ('obj', 'method_name')

    def __init__(self, obj, method_name):
        if False:
            return 10
        self.obj = obj
        self.method_name = method_name

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        method = getattr(self.obj, self.method_name)
        return method(name)

    def __getstate__(self):
        if False:
            print('Hello World!')
        return {'obj': self.obj, 'method_name': self.method_name}

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self.obj = state['obj']
        self.method_name = state['method_name']

def find_available_slug(parent, requested_slug, ignore_page_id=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Finds an available slug within the specified parent.\n\n    If the requested slug is not available, this adds a number on the end, for example:\n\n     - 'requested-slug'\n     - 'requested-slug-1'\n     - 'requested-slug-2'\n\n    And so on, until an available slug is found.\n\n    The `ignore_page_id` keyword argument is useful for when you are updating a page,\n    you can pass the page being updated here so the page's current slug is not\n    treated as in use by another page.\n    "
    pages = parent.get_children().filter(slug__startswith=requested_slug)
    if ignore_page_id:
        pages = pages.exclude(id=ignore_page_id)
    existing_slugs = set(pages.values_list('slug', flat=True))
    slug = requested_slug
    number = 1
    while slug in existing_slugs:
        slug = requested_slug + '-' + str(number)
        number += 1
    return slug

@functools.lru_cache(maxsize=None)
def get_content_languages():
    if False:
        print('Hello World!')
    '\n    Cache of settings.WAGTAIL_CONTENT_LANGUAGES in a dictionary for easy lookups by key.\n    '
    content_languages = getattr(settings, 'WAGTAIL_CONTENT_LANGUAGES', None)
    languages = dict(settings.LANGUAGES)
    if content_languages is None:
        default_language_code = get_supported_language_variant(settings.LANGUAGE_CODE)
        try:
            language_name = languages[default_language_code]
        except KeyError:
            default_language_code = default_language_code.split('-')[0]
            try:
                language_name = languages[default_language_code]
            except KeyError:
                language_name = settings.LANGUAGE_CODE
                languages[default_language_code] = settings.LANGUAGE_CODE
        content_languages = [(default_language_code, language_name)]
    for (language_code, name) in content_languages:
        if language_code not in languages:
            raise ImproperlyConfigured('The language {} is specified in WAGTAIL_CONTENT_LANGUAGES but not LANGUAGES. WAGTAIL_CONTENT_LANGUAGES must be a subset of LANGUAGES.'.format(language_code))
    return dict(content_languages)

@functools.lru_cache(maxsize=1000)
def get_supported_content_language_variant(lang_code, strict=False):
    if False:
        i = 10
        return i + 15
    "\n    Return the language code that's listed in supported languages, possibly\n    selecting a more generic variant. Raise LookupError if nothing is found.\n    If `strict` is False (the default), look for a country-specific variant\n    when neither the language code nor its generic variant is found.\n    lru_cache should have a maxsize to prevent from memory exhaustion attacks,\n    as the provided language codes are taken from the HTTP request. See also\n    <https://www.djangoproject.com/weblog/2007/oct/26/security-fix/>.\n\n    This is equvilant to Django's `django.utils.translation.get_supported_content_language_variant`\n    but reads the `WAGTAIL_CONTENT_LANGUAGES` setting instead.\n    "
    if lang_code:
        possible_lang_codes = [lang_code]
        try:
            possible_lang_codes.extend(LANG_INFO[lang_code]['fallback'])
        except KeyError:
            pass
        generic_lang_code = lang_code.split('-')[0]
        possible_lang_codes.append(generic_lang_code)
        supported_lang_codes = get_content_languages()
        for code in possible_lang_codes:
            if code in supported_lang_codes and check_for_language(code):
                return code
        if not strict:
            for supported_code in supported_lang_codes:
                if supported_code.startswith(generic_lang_code + '-'):
                    return supported_code
    raise LookupError(lang_code)

def get_locales_display_names() -> dict:
    if False:
        while True:
            i = 10
    '\n    Cache of the locale id -> locale display name mapping\n    '
    from wagtail.models import Locale
    cached_map = cache.get('wagtail_locales_display_name')
    if cached_map is None:
        cached_map = {locale.pk: locale.get_display_name() for locale in Locale.objects.all()}
        cache.set('wagtail_locales_display_name', cached_map)
    return cached_map

@receiver(setting_changed)
def reset_cache(**kwargs):
    if False:
        while True:
            i = 10
    '\n    Clear cache when global WAGTAIL_CONTENT_LANGUAGES/LANGUAGES/LANGUAGE_CODE settings are changed\n    '
    if kwargs['setting'] in ('WAGTAIL_CONTENT_LANGUAGES', 'LANGUAGES', 'LANGUAGE_CODE'):
        get_content_languages.cache_clear()
        get_supported_content_language_variant.cache_clear()

def multigetattr(item, accessor):
    if False:
        while True:
            i = 10
    "\n    Like getattr, but accepts a dotted path as the accessor to be followed to any depth.\n    At each step, the lookup on the object can be a dictionary lookup (foo['bar']) or an attribute\n    lookup (foo.bar), and if it results in a callable, will be called (provided we can do so with\n    no arguments, and it does not have an 'alters_data' property).\n\n    Modelled on the variable resolution logic in Django templates:\n    https://github.com/django/django/blob/f331eba6d576752dd79c4b37c41d981daa537fe6/django/template/base.py#L838\n    "
    current = item
    for bit in accessor.split('.'):
        try:
            current = current[bit]
        except (TypeError, AttributeError, KeyError, ValueError, IndexError):
            try:
                current = getattr(current, bit)
            except (TypeError, AttributeError):
                if bit in dir(current):
                    raise
                try:
                    current = current[int(bit)]
                except (IndexError, ValueError, KeyError, TypeError):
                    raise AttributeError(f'Failed lookup for key [{bit}] in {current!r}')
        if callable(current):
            if getattr(current, 'alters_data', False):
                raise SuspiciousOperation(f'Cannot call {current!r} from multigetattr')
            current = current()
    return current

def get_dummy_request(*, path: str='/', site: 'Site'=None) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    '\n    Return a simple ``HttpRequest`` instance that can be passed to\n    ``Page.get_url()`` and other methods to benefit from improved performance\n    when no real ``HttpRequest`` instance is available.\n\n    If ``site`` is provided, the ``HttpRequest`` is made to look like it came\n    from that Wagtail ``Site``.\n    '
    server_port = 80
    if site:
        server_name = site.hostname
        server_port = site.port
    elif settings.ALLOWED_HOSTS == ['*']:
        server_name = 'example.com'
    else:
        server_name = settings.ALLOWED_HOSTS[0]
    return RequestFactory(SERVER_NAME=server_name).get(path, SERVER_PORT=server_port)

def safe_md5(data=b'', usedforsecurity=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Safely use the MD5 hash algorithm with the given ``data`` and a flag\n    indicating if the purpose of the digest is for security or not.\n\n    On security-restricted systems (such as FIPS systems), insecure hashes\n    like MD5 are disabled by default. But passing ``usedforsecurity`` as\n    ``False`` tells the underlying security implementation we're not trying\n    to use the digest for secure purposes and to please just go ahead and\n    allow it to happen.\n    "
    try:
        return md5(data, usedforsecurity=usedforsecurity)
    except TypeError:
        return md5(data)

class BatchProcessor:
    """
    A class to help with processing of an unknown (and potentially very
    high) number of objects.

    Just set ``max_size`` to the maximum number of instances you want
    to be held in memory at any one time, and batches will be sent to the
    ``process()`` method as that number is reached, without you having to
    invoke ``process()`` regularly yourself. Just remember to invoke
    ``process()`` when you're done adding items, otherwise the final batch
    of objects will not be processed.
    """

    def __init__(self, max_size: int):
        if False:
            while True:
                i = 10
        self.max_size = max_size
        self.items = []
        self.added_count = 0

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.added_count

    def add(self, item: Any) -> None:
        if False:
            return 10
        self.items.append(item)
        self.added_count += 1
        if self.max_size and len(self.items) == self.max_size:
            self.process()

    def extend(self, iterable: Iterable[Any]) -> None:
        if False:
            while True:
                i = 10
        for item in iterable:
            self.add(item)

    def process(self):
        if False:
            print('Hello World!')
        self.pre_process()
        self._do_processing()
        self.post_process()
        self.items.clear()

    def pre_process(self):
        if False:
            return 10
        '\n        A hook to allow subclasses to do any pre-processing of the data\n        before the ``process()`` method is called.\n        '
        pass

    def _do_processing(self):
        if False:
            print('Hello World!')
        '\n        To be overridden by subclasses to do whatever it is\n        that needs to be done to the items in ``self.items``.\n        '
        raise NotImplementedError

    def post_process(self):
        if False:
            i = 10
            return i + 15
        '\n        A hook to allow subclasses to do any post-processing\n        after the ``process()`` method is called, and before\n        ``self.items`` is cleared\n        '
        pass

class BatchCreator(BatchProcessor):
    """
    A class to help with bulk creation of an unknown (and potentially very
    high) number of model instances.

    Just set ``max_size`` to the maximum number of instances you want
    to be held in memory at any one time, and batches of objects will
    be created as that number is reached, without you having to invoke
    the ``process()`` method regularly yourself. Just remember to
    invoke ``process()`` when you're done adding items, to ensure
    that the final batch items is saved.

    ``BatchSaver`` is migration-friendly! Just use the ``model``
    keyword argument when initializing to override the hardcoded model
    class with the version from your migration.
    """
    model: ModelBase = None

    def __init__(self, max_size: int, *, model: ModelBase=None, ignore_conflicts=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(max_size)
        self.ignore_conflicts = ignore_conflicts
        self.created_count = 0
        if model is not None:
            self.model = model

    def initialize_instance(self, kwargs):
        if False:
            print('Hello World!')
        return self.model(**kwargs)

    def add(self, *, instance: Model=None, **kwargs) -> None:
        if False:
            return 10
        if instance is None:
            instance = self.initialize_instance(kwargs)
        self.items.append(instance)
        self.added_count += 1
        if self.max_size and len(self.items) == self.max_size:
            self.process()

    def extend(self, iterable: Iterable[Union[Model, Dict[str, Any]]]) -> None:
        if False:
            while True:
                i = 10
        for value in iterable:
            if isinstance(value, self.model):
                self.add(instance=value)
            else:
                self.add(**value)

    def _do_processing(self):
        if False:
            i = 10
            return i + 15
        '\n        Use bulk_create() to save ``self.items``.\n        '
        if not self.items:
            return None
        self.created_count += len(self.model.objects.bulk_create(self.items, ignore_conflicts=self.ignore_conflicts))

    def get_summary(self):
        if False:
            while True:
                i = 10
        opts = self.model._meta
        return f'{self.created_count}/{self.added_count} {opts.verbose_name_plural} were created successfully.'

def make_wagtail_template_fragment_key(fragment_name, page, site, vary_on=None):
    if False:
        print('Hello World!')
    '\n    A modified version of `make_template_fragment_key` which varies on page and\n    site for use with `{% wagtailpagecache %}`.\n    '
    if vary_on is None:
        vary_on = []
    vary_on.extend([page.cache_key, site.id])
    return make_template_fragment_key(fragment_name, vary_on)