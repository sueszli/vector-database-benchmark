import copy
import datetime
import locale
import logging
import os
import re
from datetime import timezone
from html import unescape
from urllib.parse import unquote, urljoin, urlparse, urlunparse
try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    from backports.zoneinfo import ZoneInfo
from pelican.plugins import signals
from pelican.settings import DEFAULT_CONFIG
from pelican.utils import deprecated_attribute, memoized, path_to_url, posixize_path, sanitised_join, set_date_tzinfo, slugify, truncate_html_words
from pelican.urlwrappers import Author, Category, Tag, URLWrapper
logger = logging.getLogger(__name__)

class Content:
    """Represents a content.

    :param content: the string to parse, containing the original content.
    :param metadata: the metadata associated to this page (optional).
    :param settings: the settings dictionary (optional).
    :param source_path: The location of the source of this content (if any).
    :param context: The shared context between generators.

    """

    @deprecated_attribute(old='filename', new='source_path', since=(3, 2, 0))
    def filename():
        if False:
            while True:
                i = 10
        return None

    def __init__(self, content, metadata=None, settings=None, source_path=None, context=None):
        if False:
            i = 10
            return i + 15
        if metadata is None:
            metadata = {}
        if settings is None:
            settings = copy.deepcopy(DEFAULT_CONFIG)
        self.settings = settings
        self._content = content
        if context is None:
            context = {}
        self._context = context
        self.translations = []
        local_metadata = dict()
        local_metadata.update(metadata)
        for (key, value) in local_metadata.items():
            if key in ('save_as', 'url'):
                key = 'override_' + key
            setattr(self, key.lower(), value)
        self.metadata = local_metadata
        self.template = self._get_template()
        if not hasattr(self, 'author'):
            if hasattr(self, 'authors'):
                self.author = self.authors[0]
            elif 'AUTHOR' in settings:
                self.author = Author(settings['AUTHOR'], settings)
        if not hasattr(self, 'authors') and hasattr(self, 'author'):
            self.authors = [self.author]
        self.in_default_lang = True
        if 'DEFAULT_LANG' in settings:
            default_lang = settings['DEFAULT_LANG'].lower()
            if not hasattr(self, 'lang'):
                self.lang = default_lang
            self.in_default_lang = self.lang == default_lang
        if not hasattr(self, 'slug'):
            if settings['SLUGIFY_SOURCE'] == 'title' and hasattr(self, 'title'):
                value = self.title
            elif settings['SLUGIFY_SOURCE'] == 'basename' and source_path is not None:
                value = os.path.basename(os.path.splitext(source_path)[0])
            else:
                value = None
            if value is not None:
                self.slug = slugify(value, regex_subs=settings.get('SLUG_REGEX_SUBSTITUTIONS', []), preserve_case=settings.get('SLUGIFY_PRESERVE_CASE', False), use_unicode=settings.get('SLUGIFY_USE_UNICODE', False))
        self.source_path = source_path
        self.relative_source_path = self.get_relative_source_path()
        if not hasattr(self, 'date_format'):
            if hasattr(self, 'lang') and self.lang in settings['DATE_FORMATS']:
                self.date_format = settings['DATE_FORMATS'][self.lang]
            else:
                self.date_format = settings['DEFAULT_DATE_FORMAT']
        if isinstance(self.date_format, tuple):
            locale_string = self.date_format[0]
            locale.setlocale(locale.LC_ALL, locale_string)
            self.date_format = self.date_format[1]
        default_timezone = settings.get('TIMEZONE', 'UTC')
        timezone = getattr(self, 'timezone', default_timezone)
        self.timezone = ZoneInfo(timezone)
        if hasattr(self, 'date'):
            self.date = set_date_tzinfo(self.date, timezone)
            self.locale_date = self.date.strftime(self.date_format)
        if hasattr(self, 'modified'):
            self.modified = set_date_tzinfo(self.modified, timezone)
            self.locale_modified = self.modified.strftime(self.date_format)
        if not hasattr(self, 'status'):
            self.status = getattr(self, 'default_status', '')
        if 'summary' in metadata:
            self._summary = metadata['summary']
        signals.content_object_init.send(self)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.source_path or repr(self)

    def _has_valid_mandatory_properties(self):
        if False:
            while True:
                i = 10
        'Test mandatory properties are set.'
        for prop in self.mandatory_properties:
            if not hasattr(self, prop):
                logger.error("Skipping %s: could not find information about '%s'", self, prop)
                return False
        return True

    def _has_valid_save_as(self):
        if False:
            i = 10
            return i + 15
        "Return true if save_as doesn't write outside output path, false\n        otherwise."
        try:
            output_path = self.settings['OUTPUT_PATH']
        except KeyError:
            return True
        try:
            sanitised_join(output_path, self.save_as)
        except RuntimeError:
            logger.error('Skipping %s: file %r would be written outside output path', self, self.save_as)
            return False
        return True

    def _has_valid_status(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'allowed_statuses'):
            if self.status not in self.allowed_statuses:
                logger.error("Unknown status '%s' for file %s, skipping it. (Not in %s)", self.status, self, self.allowed_statuses)
                return False
        return True

    def is_valid(self):
        if False:
            print('Hello World!')
        'Validate Content'
        return all([self._has_valid_mandatory_properties(), self._has_valid_save_as(), self._has_valid_status()])

    @property
    def url_format(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the URL, formatted with the proper values'
        metadata = copy.copy(self.metadata)
        path = self.metadata.get('path', self.get_relative_source_path())
        metadata.update({'path': path_to_url(path), 'slug': getattr(self, 'slug', ''), 'lang': getattr(self, 'lang', 'en'), 'date': getattr(self, 'date', datetime.datetime.now()), 'author': self.author.slug if hasattr(self, 'author') else '', 'category': self.category.slug if hasattr(self, 'category') else ''})
        return metadata

    def _expand_settings(self, key, klass=None):
        if False:
            while True:
                i = 10
        if not klass:
            klass = self.__class__.__name__
        fq_key = f'{klass}_{key}'.upper()
        return str(self.settings[fq_key]).format(**self.url_format)

    def get_url_setting(self, key):
        if False:
            print('Hello World!')
        if hasattr(self, 'override_' + key):
            return getattr(self, 'override_' + key)
        key = key if self.in_default_lang else 'lang_%s' % key
        return self._expand_settings(key)

    def _link_replacer(self, siteurl, m):
        if False:
            for i in range(10):
                print('nop')
        what = m.group('what')
        value = urlparse(m.group('value'))
        path = value.path
        origin = m.group('path')
        if self.settings['RELATIVE_URLS']:
            joiner = os.path.join
        else:
            joiner = urljoin
            if not siteurl.endswith('/'):
                siteurl += '/'
        if what in {'filename', 'static', 'attach'}:

            def _get_linked_content(key, url):
                if False:
                    i = 10
                    return i + 15
                nonlocal value

                def _find_path(path):
                    if False:
                        i = 10
                        return i + 15
                    if path.startswith('/'):
                        path = path[1:]
                    else:
                        path = self.get_relative_source_path(os.path.join(self.relative_dir, path))
                    return self._context[key].get(path, None)
                result = _find_path(url.path)
                if result is not None:
                    return result
                result = _find_path(unquote(url.path))
                if result is not None:
                    return result
                unescaped_url = urlparse(unescape(url.geturl()))
                result = _find_path(unescaped_url.path)
                if result is not None:
                    value = unescaped_url
                    return result
                if what == 'filename' and key == 'generated_content':
                    linked_content = _get_linked_content('static_content', value)
                    if linked_content:
                        logger.warning('{filename} used for linking to static content %s in %s. Use {static} instead', value.path, self.get_relative_source_path())
                        return linked_content
                return None
            if what == 'filename':
                key = 'generated_content'
            else:
                key = 'static_content'
            linked_content = _get_linked_content(key, value)
            if linked_content:
                if what == 'attach':
                    linked_content.attach_to(self)
                origin = joiner(siteurl, linked_content.url)
                origin = origin.replace('\\', '/')
            else:
                logger.warning("Unable to find '%s', skipping url replacement.", value.geturl(), extra={'limit_msg': 'Other resources were not found and their urls not replaced'})
        elif what == 'category':
            origin = joiner(siteurl, Category(path, self.settings).url)
        elif what == 'tag':
            origin = joiner(siteurl, Tag(path, self.settings).url)
        elif what == 'index':
            origin = joiner(siteurl, self.settings['INDEX_SAVE_AS'])
        elif what == 'author':
            origin = joiner(siteurl, Author(path, self.settings).url)
        else:
            logger.warning("Replacement Indicator '%s' not recognized, skipping replacement", what)
        parts = list(value)
        parts[2] = origin
        origin = urlunparse(parts)
        return ''.join((m.group('markup'), m.group('quote'), origin, m.group('quote')))

    def _get_intrasite_link_regex(self):
        if False:
            i = 10
            return i + 15
        intrasite_link_regex = self.settings['INTRASITE_LINK_REGEX']
        regex = '\n            (?P<markup><[^\\>]+  # match tag with all url-value attributes\n                (?:href|src|poster|data|cite|formaction|action|content)\\s*=\\s*)\n\n            (?P<quote>["\\\'])      # require value to be quoted\n            (?P<path>{}(?P<value>.*?))  # the url value\n            (?P=quote)'.format(intrasite_link_regex)
        return re.compile(regex, re.X)

    def _update_content(self, content, siteurl):
        if False:
            for i in range(10):
                print('nop')
        'Update the content attribute.\n\n        Change all the relative paths of the content to relative paths\n        suitable for the output content.\n\n        :param content: content resource that will be passed to the templates.\n        :param siteurl: siteurl which is locally generated by the writer in\n                        case of RELATIVE_URLS.\n        '
        if not content:
            return content
        hrefs = self._get_intrasite_link_regex()
        return hrefs.sub(lambda m: self._link_replacer(siteurl, m), content)

    def get_static_links(self):
        if False:
            for i in range(10):
                print('nop')
        static_links = set()
        hrefs = self._get_intrasite_link_regex()
        for m in hrefs.finditer(self._content):
            what = m.group('what')
            value = urlparse(m.group('value'))
            path = value.path
            if what not in {'static', 'attach'}:
                continue
            if path.startswith('/'):
                path = path[1:]
            else:
                path = self.get_relative_source_path(os.path.join(self.relative_dir, path))
            path = path.replace('%20', ' ')
            static_links.add(path)
        return static_links

    def get_siteurl(self):
        if False:
            for i in range(10):
                print('nop')
        return self._context.get('localsiteurl', '')

    @memoized
    def get_content(self, siteurl):
        if False:
            while True:
                i = 10
        if hasattr(self, '_get_content'):
            content = self._get_content()
        else:
            content = self._content
        return self._update_content(content, siteurl)

    @property
    def content(self):
        if False:
            while True:
                i = 10
        return self.get_content(self.get_siteurl())

    @memoized
    def get_summary(self, siteurl):
        if False:
            while True:
                i = 10
        'Returns the summary of an article.\n\n        This is based on the summary metadata if set, otherwise truncate the\n        content.\n        '
        if 'summary' in self.metadata:
            return self.metadata['summary']
        if self.settings['SUMMARY_MAX_LENGTH'] is None:
            return self.content
        return truncate_html_words(self.content, self.settings['SUMMARY_MAX_LENGTH'], self.settings['SUMMARY_END_SUFFIX'])

    @property
    def summary(self):
        if False:
            return 10
        return self.get_summary(self.get_siteurl())

    def _get_summary(self):
        if False:
            while True:
                i = 10
        'deprecated function to access summary'
        logger.warning('_get_summary() has been deprecated since 3.6.4. Use the summary decorator instead')
        return self.summary

    @summary.setter
    def summary(self, value):
        if False:
            print('Hello World!')
        'Dummy function'
        pass

    @property
    def status(self):
        if False:
            i = 10
            return i + 15
        return self._status

    @status.setter
    def status(self, value):
        if False:
            i = 10
            return i + 15
        self._status = value.lower()

    @property
    def url(self):
        if False:
            while True:
                i = 10
        return self.get_url_setting('url')

    @property
    def save_as(self):
        if False:
            return 10
        return self.get_url_setting('save_as')

    def _get_template(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, 'template') and self.template is not None:
            return self.template
        else:
            return self.default_template

    def get_relative_source_path(self, source_path=None):
        if False:
            i = 10
            return i + 15
        'Return the relative path (from the content path) to the given\n        source_path.\n\n        If no source path is specified, use the source path of this\n        content object.\n        '
        if not source_path:
            source_path = self.source_path
        if source_path is None:
            return None
        return posixize_path(os.path.relpath(os.path.abspath(os.path.join(self.settings['PATH'], source_path)), os.path.abspath(self.settings['PATH'])))

    @property
    def relative_dir(self):
        if False:
            return 10
        return posixize_path(os.path.dirname(os.path.relpath(os.path.abspath(self.source_path), os.path.abspath(self.settings['PATH']))))

    def refresh_metadata_intersite_links(self):
        if False:
            for i in range(10):
                print('nop')
        for key in self.settings['FORMATTED_FIELDS']:
            if key in self.metadata and key != 'summary':
                value = self._update_content(self.metadata[key], self.get_siteurl())
                self.metadata[key] = value
                setattr(self, key.lower(), value)
        if 'summary' in self.settings['FORMATTED_FIELDS'] and 'summary' in self.metadata:
            self._summary = self._update_content(self._summary, self.get_siteurl())
            self.metadata['summary'] = self._summary

class Page(Content):
    mandatory_properties = ('title',)
    allowed_statuses = ('published', 'hidden', 'draft')
    default_status = 'published'
    default_template = 'page'

    def _expand_settings(self, key):
        if False:
            for i in range(10):
                print('nop')
        klass = 'draft_page' if self.status == 'draft' else None
        return super()._expand_settings(key, klass)

class Article(Content):
    mandatory_properties = ('title', 'date', 'category')
    allowed_statuses = ('published', 'hidden', 'draft')
    default_status = 'published'
    default_template = 'article'

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        if not self.settings['WITH_FUTURE_DATES'] and hasattr(self, 'date'):
            if self.date.tzinfo is None:
                now = datetime.datetime.now()
            else:
                now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
            if self.date > now:
                self.status = 'draft'
        if not hasattr(self, 'date') and self.status == 'draft':
            self.date = datetime.datetime.max.replace(tzinfo=self.timezone)

    def _expand_settings(self, key):
        if False:
            while True:
                i = 10
        klass = 'draft' if self.status == 'draft' else 'article'
        return super()._expand_settings(key, klass)

class Static(Content):
    mandatory_properties = ('title',)
    default_status = 'published'
    default_template = None

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self._output_location_referenced = False

    @deprecated_attribute(old='filepath', new='source_path', since=(3, 2, 0))
    def filepath():
        if False:
            print('Hello World!')
        return None

    @deprecated_attribute(old='src', new='source_path', since=(3, 2, 0))
    def src():
        if False:
            i = 10
            return i + 15
        return None

    @deprecated_attribute(old='dst', new='save_as', since=(3, 2, 0))
    def dst():
        if False:
            while True:
                i = 10
        return None

    @property
    def url(self):
        if False:
            for i in range(10):
                print('nop')
        self._output_location_referenced = True
        return super().url

    @property
    def save_as(self):
        if False:
            print('Hello World!')
        self._output_location_referenced = True
        return super().save_as

    def attach_to(self, content):
        if False:
            return 10
        'Override our output directory with that of the given content object.'
        linking_source_dir = os.path.dirname(content.source_path)
        tail_path = os.path.relpath(self.source_path, linking_source_dir)
        if tail_path.startswith(os.pardir + os.sep):
            tail_path = os.path.basename(tail_path)
        new_save_as = os.path.join(os.path.dirname(content.save_as), tail_path)
        new_url = path_to_url(new_save_as)

        def _log_reason(reason):
            if False:
                for i in range(10):
                    print('nop')
            logger.warning('The {attach} link in %s cannot relocate %s because %s. Falling back to {filename} link behavior instead.', content.get_relative_source_path(), self.get_relative_source_path(), reason, extra={'limit_msg': 'More {attach} warnings silenced.'})
        if hasattr(self, 'override_save_as') or hasattr(self, 'override_url'):
            if new_save_as != self.save_as or new_url != self.url:
                _log_reason('its output location was already overridden')
            return
        if self._output_location_referenced:
            if new_save_as != self.save_as or new_url != self.url:
                _log_reason('another link already referenced its location')
            return
        self.override_save_as = new_save_as
        self.override_url = new_url