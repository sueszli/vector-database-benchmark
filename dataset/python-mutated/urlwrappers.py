import functools
import logging
import os
import pathlib
from pelican.utils import slugify
logger = logging.getLogger(__name__)

@functools.total_ordering
class URLWrapper:

    def __init__(self, name, settings):
        if False:
            for i in range(10):
                print('nop')
        self.settings = settings
        self._name = name
        self._slug = None
        self._slug_from_name = True

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    @name.setter
    def name(self, name):
        if False:
            i = 10
            return i + 15
        self._name = name
        if self._slug_from_name:
            self._slug = None

    @property
    def slug(self):
        if False:
            i = 10
            return i + 15
        if self._slug is None:
            class_key = f'{self.__class__.__name__.upper()}_REGEX_SUBSTITUTIONS'
            regex_subs = self.settings.get(class_key, self.settings.get('SLUG_REGEX_SUBSTITUTIONS', []))
            preserve_case = self.settings.get('SLUGIFY_PRESERVE_CASE', False)
            self._slug = slugify(self.name, regex_subs=regex_subs, preserve_case=preserve_case, use_unicode=self.settings.get('SLUGIFY_USE_UNICODE', False))
        return self._slug

    @slug.setter
    def slug(self, slug):
        if False:
            while True:
                i = 10
        self._slug_from_name = False
        self._slug = slug

    def as_dict(self):
        if False:
            while True:
                i = 10
        d = self.__dict__
        d['name'] = self.name
        d['slug'] = self.slug
        return d

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.slug)

    def _normalize_key(self, key):
        if False:
            for i in range(10):
                print('nop')
        class_key = f'{self.__class__.__name__.upper()}_REGEX_SUBSTITUTIONS'
        regex_subs = self.settings.get(class_key, self.settings.get('SLUG_REGEX_SUBSTITUTIONS', []))
        use_unicode = self.settings.get('SLUGIFY_USE_UNICODE', False)
        preserve_case = self.settings.get('SLUGIFY_PRESERVE_CASE', False)
        return slugify(key, regex_subs=regex_subs, preserve_case=preserve_case, use_unicode=use_unicode)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, self.__class__):
            return self.slug == other.slug
        if isinstance(other, str):
            return self.slug == self._normalize_key(other)
        return False

    def __ne__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, self.__class__):
            return self.slug != other.slug
        if isinstance(other, str):
            return self.slug != self._normalize_key(other)
        return True

    def __lt__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, self.__class__):
            return self.slug < other.slug
        if isinstance(other, str):
            return self.slug < self._normalize_key(other)
        return False

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.name

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'<{type(self).__name__} {repr(self._name)}>'

    def _from_settings(self, key, get_page_name=False):
        if False:
            return 10
        'Returns URL information as defined in settings.\n\n        When get_page_name=True returns URL without anything after {slug} e.g.\n        if in settings: CATEGORY_URL="cat/{slug}.html" this returns\n        "cat/{slug}" Useful for pagination.\n\n        '
        setting = f'{self.__class__.__name__.upper()}_{key}'
        value = self.settings[setting]
        if isinstance(value, pathlib.Path):
            value = str(value)
        if not isinstance(value, str):
            logger.warning('%s is set to %s', setting, value)
            return value
        elif get_page_name:
            return os.path.splitext(value)[0].format(**self.as_dict())
        else:
            return value.format(**self.as_dict())
    page_name = property(functools.partial(_from_settings, key='URL', get_page_name=True))
    url = property(functools.partial(_from_settings, key='URL'))
    save_as = property(functools.partial(_from_settings, key='SAVE_AS'))

class Category(URLWrapper):
    pass

class Tag(URLWrapper):

    def __init__(self, name, *args, **kwargs):
        if False:
            return 10
        super().__init__(name.strip(), *args, **kwargs)

class Author(URLWrapper):
    pass