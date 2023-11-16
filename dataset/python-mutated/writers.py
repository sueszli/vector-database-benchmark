import logging
import os
from posixpath import join as posix_join
from urllib.parse import urljoin
from feedgenerator import Atom1Feed, Rss201rev2Feed, get_tag_uri
from markupsafe import Markup
from pelican.paginator import Paginator
from pelican.plugins import signals
from pelican.utils import get_relative_path, path_to_url, sanitised_join, set_date_tzinfo
logger = logging.getLogger(__name__)

class Writer:

    def __init__(self, output_path, settings=None):
        if False:
            for i in range(10):
                print('nop')
        self.output_path = output_path
        self.reminder = dict()
        self.settings = settings or {}
        self._written_files = set()
        self._overridden_files = set()
        if 'RELATIVE_URLS' in self.settings and self.settings['RELATIVE_URLS']:
            self.urljoiner = posix_join
        else:
            self.urljoiner = lambda base, url: urljoin(base if base.endswith('/') else base + '/', str(url))

    def _create_new_feed(self, feed_type, feed_title, context):
        if False:
            for i in range(10):
                print('nop')
        feed_class = Rss201rev2Feed if feed_type == 'rss' else Atom1Feed
        if feed_title:
            feed_title = context['SITENAME'] + ' - ' + feed_title
        else:
            feed_title = context['SITENAME']
        return feed_class(title=Markup(feed_title).striptags(), link=self.site_url + '/', feed_url=self.feed_url, description=context.get('SITESUBTITLE', ''), subtitle=context.get('SITESUBTITLE', None))

    def _add_item_to_the_feed(self, feed, item):
        if False:
            while True:
                i = 10
        title = Markup(item.title).striptags()
        link = self.urljoiner(self.site_url, item.url)
        if isinstance(feed, Rss201rev2Feed):
            content = None
            if self.settings.get('RSS_FEED_SUMMARY_ONLY'):
                description = item.summary
            else:
                description = item.get_content(self.site_url)
        else:
            content = item.get_content(self.site_url)
            description = item.summary
            if description == content:
                description = None
        categories = []
        if hasattr(item, 'category'):
            categories.append(item.category)
        if hasattr(item, 'tags'):
            categories.extend(item.tags)
        feed.add_item(title=title, link=link, unique_id=get_tag_uri(link, item.date), description=description, content=content, categories=categories or None, author_name=getattr(item, 'author', ''), pubdate=set_date_tzinfo(item.date, self.settings.get('TIMEZONE', None)), updateddate=set_date_tzinfo(item.modified, self.settings.get('TIMEZONE', None)) if hasattr(item, 'modified') else None)

    def _open_w(self, filename, encoding, override=False):
        if False:
            i = 10
            return i + 15
        'Open a file to write some content to it.\n\n        Exit if we have already written to that file, unless one (and no more\n        than one) of the writes has the override parameter set to True.\n        '
        if filename in self._overridden_files:
            if override:
                raise RuntimeError('File %s is set to be overridden twice' % filename)
            logger.info('Skipping %s', filename)
            filename = os.devnull
        elif filename in self._written_files:
            if override:
                logger.info('Overwriting %s', filename)
            else:
                raise RuntimeError('File %s is to be overwritten' % filename)
        if override:
            self._overridden_files.add(filename)
        self._written_files.add(filename)
        return open(filename, 'w', encoding=encoding)

    def write_feed(self, elements, context, path=None, url=None, feed_type='atom', override_output=False, feed_title=None):
        if False:
            print('Hello World!')
        'Generate a feed with the list of articles provided\n\n        Return the feed. If no path or output_path is specified, just\n        return the feed object.\n\n        :param elements: the articles to put on the feed.\n        :param context: the context to get the feed metadata.\n        :param path: the path to output.\n        :param url: the publicly visible feed URL; if None, path is used\n            instead\n        :param feed_type: the feed type to use (atom or rss)\n        :param override_output: boolean telling if we can override previous\n            output with the same name (and if next files written with the same\n            name should be skipped to keep that one)\n        :param feed_title: the title of the feed.o\n        '
        self.site_url = context.get('SITEURL', path_to_url(get_relative_path(path)))
        self.feed_domain = context.get('FEED_DOMAIN')
        self.feed_url = self.urljoiner(self.feed_domain, url or path)
        feed = self._create_new_feed(feed_type, feed_title, context)
        for element in elements[:self.settings['FEED_MAX_ITEMS']]:
            self._add_item_to_the_feed(feed, element)
        signals.feed_generated.send(context, feed=feed)
        if path:
            complete_path = sanitised_join(self.output_path, path)
            try:
                os.makedirs(os.path.dirname(complete_path))
            except Exception:
                pass
            with self._open_w(complete_path, 'utf-8', override_output) as fp:
                feed.write(fp, 'utf-8')
                logger.info('Writing %s', complete_path)
            signals.feed_written.send(complete_path, context=context, feed=feed)
        return feed

    def write_file(self, name, template, context, relative_urls=False, paginated=None, template_name=None, override_output=False, url=None, **kwargs):
        if False:
            print('Hello World!')
        'Render the template and write the file.\n\n        :param name: name of the file to output\n        :param template: template to use to generate the content\n        :param context: dict to pass to the templates.\n        :param relative_urls: use relative urls or absolutes ones\n        :param paginated: dict of article list to paginate - must have the\n            same length (same list in different orders)\n        :param template_name: the template name, for pagination\n        :param override_output: boolean telling if we can override previous\n            output with the same name (and if next files written with the same\n            name should be skipped to keep that one)\n        :param url: url of the file (needed by the paginator)\n        :param **kwargs: additional variables to pass to the templates\n        '
        if name is False or name == '':
            return
        elif not name:
            return

        def _write_file(template, localcontext, output_path, name, override):
            if False:
                i = 10
                return i + 15
            'Render the template write the file.'
            if localcontext['localsiteurl']:
                context['localsiteurl'] = localcontext['localsiteurl']
            output = template.render(localcontext)
            path = sanitised_join(output_path, name)
            try:
                os.makedirs(os.path.dirname(path))
            except Exception:
                pass
            with self._open_w(path, 'utf-8', override=override) as f:
                f.write(output)
            logger.info('Writing %s', path)
            signals.content_written.send(path, context=localcontext)

        def _get_localcontext(context, name, kwargs, relative_urls):
            if False:
                print('Hello World!')
            localcontext = context.copy()
            localcontext['localsiteurl'] = localcontext.get('localsiteurl', None)
            if relative_urls:
                relative_url = path_to_url(get_relative_path(name))
                localcontext['SITEURL'] = relative_url
                localcontext['localsiteurl'] = relative_url
            localcontext['output_file'] = name
            localcontext.update(kwargs)
            return localcontext
        if paginated is None:
            paginated = {key: val for (key, val) in kwargs.items() if key in {'articles', 'dates'}}
        if paginated and template_name in self.settings['PAGINATED_TEMPLATES']:
            per_page = self.settings['PAGINATED_TEMPLATES'][template_name] or self.settings['DEFAULT_PAGINATION']
            paginators = {key: Paginator(name, url, val, self.settings, per_page) for (key, val) in paginated.items()}
            for page_num in range(list(paginators.values())[0].num_pages):
                paginated_kwargs = kwargs.copy()
                for key in paginators.keys():
                    paginator = paginators[key]
                    previous_page = paginator.page(page_num) if page_num > 0 else None
                    page = paginator.page(page_num + 1)
                    next_page = paginator.page(page_num + 2) if page_num + 1 < paginator.num_pages else None
                    paginated_kwargs.update({'%s_paginator' % key: paginator, '%s_page' % key: page, '%s_previous_page' % key: previous_page, '%s_next_page' % key: next_page})
                localcontext = _get_localcontext(context, page.save_as, paginated_kwargs, relative_urls)
                _write_file(template, localcontext, self.output_path, page.save_as, override_output)
        else:
            localcontext = _get_localcontext(context, name, kwargs, relative_urls)
            _write_file(template, localcontext, self.output_path, name, override_output)