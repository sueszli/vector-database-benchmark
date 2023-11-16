__title__ = 'newspaper'
__author__ = 'Lucas Ou-Yang'
__license__ = 'MIT'
__copyright__ = 'Copyright 2014, Lucas Ou-Yang'
import logging
import copy
import os
import glob
from urllib.parse import urlparse
import requests
from . import images
from . import network
from . import nlp
from . import settings
from . import urls
from .cleaners import DocumentCleaner
from .configuration import Configuration
from .extractors import ContentExtractor
from .outputformatters import OutputFormatter
from .utils import URLHelper, RawHelper, extend_config, get_available_languages, extract_meta_refresh
from .videos.extractors import VideoExtractor
log = logging.getLogger(__name__)

class ArticleDownloadState(object):
    NOT_STARTED = 0
    FAILED_RESPONSE = 1
    SUCCESS = 2

class ArticleException(Exception):
    pass

class Article(object):
    """Article objects abstract an online news article page
    """

    def __init__(self, url, title='', source_url='', config=None, **kwargs):
        if False:
            return 10
        'The **kwargs argument may be filled with config values, which\n        is added into the config object\n        '
        if isinstance(title, Configuration) or isinstance(source_url, Configuration):
            raise ArticleException('Configuration object being passed incorrectly as title or source_url! Please verify `Article`s __init__() fn.')
        self.config = config or Configuration()
        self.config = extend_config(self.config, kwargs)
        self.extractor = ContentExtractor(self.config)
        if source_url == '':
            scheme = urls.get_scheme(url)
            if scheme is None:
                scheme = 'http'
            source_url = scheme + '://' + urls.get_domain(url)
        if source_url is None or source_url == '':
            raise ArticleException('input url bad format')
        self.source_url = source_url
        self.url = urls.prepare_url(url, self.source_url)
        self.title = title
        self.top_img = self.top_image = ''
        self.meta_img = ''
        self.imgs = self.images = []
        self.movies = []
        self.text = ''
        self.keywords = []
        self.meta_keywords = []
        self.tags = set()
        self.authors = []
        self.publish_date = ''
        self.summary = ''
        self.html = ''
        self.article_html = ''
        self.is_parsed = False
        self.download_state = ArticleDownloadState.NOT_STARTED
        self.download_exception_msg = None
        self.meta_description = ''
        self.meta_lang = ''
        self.meta_favicon = ''
        self.meta_site_name = ''
        self.meta_data = {}
        self.canonical_link = ''
        self.top_node = None
        self.clean_top_node = None
        self.doc = None
        self.clean_doc = None
        self.additional_data = {}

    def build(self):
        if False:
            while True:
                i = 10
        "Build a lone article from a URL independent of the source (newspaper).\n        Don't normally call this method b/c it's good to multithread articles\n        on a source (newspaper) level.\n        "
        self.download()
        self.parse()
        self.nlp()

    def _parse_scheme_file(self, path):
        if False:
            i = 10
            return i + 15
        try:
            with open(path, 'r') as fin:
                return fin.read()
        except OSError as e:
            self.download_state = ArticleDownloadState.FAILED_RESPONSE
            self.download_exception_msg = e.strerror
            return None

    def _parse_scheme_http(self):
        if False:
            while True:
                i = 10
        try:
            return network.get_html_2XX_only(self.url, self.config)
        except requests.exceptions.RequestException as e:
            self.download_state = ArticleDownloadState.FAILED_RESPONSE
            self.download_exception_msg = str(e)
            return None

    def download(self, input_html=None, title=None, recursion_counter=0):
        if False:
            for i in range(10):
                print('nop')
        "Downloads the link's HTML content, don't use if you are batch async\n        downloading articles\n\n        recursion_counter (currently 1) stops refreshes that are potentially\n        infinite\n        "
        if input_html is None:
            parsed_url = urlparse(self.url)
            if parsed_url.scheme == 'file':
                html = self._parse_scheme_file(parsed_url.path)
            else:
                html = self._parse_scheme_http()
            if html is None:
                log.debug('Download failed on URL %s because of %s' % (self.url, self.download_exception_msg))
                return
        else:
            html = input_html
        if self.config.follow_meta_refresh:
            meta_refresh_url = extract_meta_refresh(html)
            if meta_refresh_url and recursion_counter < 1:
                return self.download(input_html=network.get_html(meta_refresh_url), recursion_counter=recursion_counter + 1)
        self.set_html(html)
        self.set_title(title)

    def parse(self):
        if False:
            for i in range(10):
                print('nop')
        self.throw_if_not_downloaded_verbose()
        self.doc = self.config.get_parser().fromstring(self.html)
        self.clean_doc = copy.deepcopy(self.doc)
        if self.doc is None:
            return
        parse_candidate = self.get_parse_candidate()
        self.link_hash = parse_candidate.link_hash
        document_cleaner = DocumentCleaner(self.config)
        output_formatter = OutputFormatter(self.config)
        title = self.extractor.get_title(self.clean_doc)
        self.set_title(title)
        authors = self.extractor.get_authors(self.clean_doc)
        self.set_authors(authors)
        meta_lang = self.extractor.get_meta_lang(self.clean_doc)
        self.set_meta_language(meta_lang)
        if self.config.use_meta_language:
            self.extractor.update_language(self.meta_lang)
            output_formatter.update_language(self.meta_lang)
        meta_favicon = self.extractor.get_favicon(self.clean_doc)
        self.set_meta_favicon(meta_favicon)
        meta_site_name = self.extractor.get_meta_site_name(self.clean_doc)
        self.set_meta_site_name(meta_site_name)
        meta_description = self.extractor.get_meta_description(self.clean_doc)
        self.set_meta_description(meta_description)
        canonical_link = self.extractor.get_canonical_link(self.url, self.clean_doc)
        self.set_canonical_link(canonical_link)
        tags = self.extractor.extract_tags(self.clean_doc)
        self.set_tags(tags)
        meta_keywords = self.extractor.get_meta_keywords(self.clean_doc)
        self.set_meta_keywords(meta_keywords)
        meta_data = self.extractor.get_meta_data(self.clean_doc)
        self.set_meta_data(meta_data)
        self.publish_date = self.extractor.get_publishing_date(self.url, self.clean_doc)
        self.doc = document_cleaner.clean(self.doc)
        self.top_node = self.extractor.calculate_best_node(self.doc)
        if self.top_node is not None:
            video_extractor = VideoExtractor(self.config, self.top_node)
            self.set_movies(video_extractor.get_videos())
            self.top_node = self.extractor.post_cleanup(self.top_node)
            self.clean_top_node = copy.deepcopy(self.top_node)
            (text, article_html) = output_formatter.get_formatted(self.top_node)
            self.set_article_html(article_html)
            self.set_text(text)
        self.fetch_images()
        self.is_parsed = True
        self.release_resources()

    def fetch_images(self):
        if False:
            return 10
        if self.clean_doc is not None:
            meta_img_url = self.extractor.get_meta_img_url(self.url, self.clean_doc)
            self.set_meta_img(meta_img_url)
            imgs = self.extractor.get_img_urls(self.url, self.clean_doc)
            if self.meta_img:
                imgs.add(self.meta_img)
            self.set_imgs(imgs)
        if self.clean_top_node is not None and (not self.has_top_image()):
            first_img = self.extractor.get_first_img_url(self.url, self.clean_top_node)
            if self.config.fetch_images:
                self.set_top_img(first_img)
            else:
                self.set_top_img_no_check(first_img)
        if not self.has_top_image() and self.config.fetch_images:
            self.set_reddit_top_img()

    def has_top_image(self):
        if False:
            print('Hello World!')
        return self.top_img is not None and self.top_img != ''

    def is_valid_url(self):
        if False:
            i = 10
            return i + 15
        'Performs a check on the url of this link to determine if article\n        is a real news article or not\n        '
        return urls.valid_url(self.url)

    def is_valid_body(self):
        if False:
            while True:
                i = 10
        "If the article's body text is long enough to meet\n        standard article requirements, keep the article\n        "
        if not self.is_parsed:
            raise ArticleException("must parse article before checking                                     if it's body is valid!")
        meta_type = self.extractor.get_meta_type(self.clean_doc)
        wordcount = self.text.split(' ')
        sentcount = self.text.split('.')
        if meta_type == 'article' and len(wordcount) > self.config.MIN_WORD_COUNT:
            log.debug('%s verified for article and wc' % self.url)
            return True
        if not self.is_media_news() and (not self.text):
            log.debug('%s caught for no media no text' % self.url)
            return False
        if self.title is None or len(self.title.split(' ')) < 2:
            log.debug('%s caught for bad title' % self.url)
            return False
        if len(wordcount) < self.config.MIN_WORD_COUNT:
            log.debug('%s caught for word cnt' % self.url)
            return False
        if len(sentcount) < self.config.MIN_SENT_COUNT:
            log.debug('%s caught for sent cnt' % self.url)
            return False
        if self.html is None or self.html == '':
            log.debug('%s caught for no html' % self.url)
            return False
        log.debug('%s verified for default true' % self.url)
        return True

    def is_media_news(self):
        if False:
            return 10
        'If the article is related heavily to media:\n        gallery, video, big pictures, etc\n        '
        safe_urls = ['/video', '/slide', '/gallery', '/powerpoint', '/fashion', '/glamour', '/cloth']
        for s in safe_urls:
            if s in self.url:
                return True
        return False

    def nlp(self):
        if False:
            return 10
        'Keyword extraction wrapper\n        '
        self.throw_if_not_downloaded_verbose()
        self.throw_if_not_parsed_verbose()
        nlp.load_stopwords(self.config.get_language())
        text_keyws = list(nlp.keywords(self.text).keys())
        title_keyws = list(nlp.keywords(self.title).keys())
        keyws = list(set(title_keyws + text_keyws))
        self.set_keywords(keyws)
        max_sents = self.config.MAX_SUMMARY_SENT
        summary_sents = nlp.summarize(title=self.title, text=self.text, max_sents=max_sents)
        summary = '\n'.join(summary_sents)
        self.set_summary(summary)

    def get_parse_candidate(self):
        if False:
            while True:
                i = 10
        'A parse candidate is a wrapper object holding a link hash of this\n        article and a final_url of the article\n        '
        if self.html:
            return RawHelper.get_parsing_candidate(self.url, self.html)
        return URLHelper.get_parsing_candidate(self.url)

    def build_resource_path(self):
        if False:
            i = 10
            return i + 15
        'Must be called after computing HTML/final URL\n        '
        res_path = self.get_resource_path()
        if not os.path.exists(res_path):
            os.mkdir(res_path)

    def get_resource_path(self):
        if False:
            return 10
        'Every article object has a special directory to store data in from\n        initialization to garbage collection\n        '
        res_dir_fn = 'article_resources'
        resource_directory = os.path.join(settings.TOP_DIRECTORY, res_dir_fn)
        if not os.path.exists(resource_directory):
            os.mkdir(resource_directory)
        dir_path = os.path.join(resource_directory, '%s_' % self.link_hash)
        return dir_path

    def release_resources(self):
        if False:
            i = 10
            return i + 15
        path = self.get_resource_path()
        for fname in glob.glob(path):
            try:
                os.remove(fname)
            except OSError:
                pass

    def set_reddit_top_img(self):
        if False:
            i = 10
            return i + 15
        "Wrapper for setting images. Queries known image attributes\n        first, then uses Reddit's image algorithm as a fallback.\n        "
        try:
            s = images.Scraper(self)
            self.set_top_img(s.largest_image_url())
        except TypeError as e:
            if "Can't convert 'NoneType' object to str implicitly" in e.args[0]:
                log.debug('No pictures found. Top image not set, %s' % e)
            elif 'timed out' in e.args[0]:
                log.debug('Download of picture timed out. Top image not set, %s' % e)
            else:
                log.critical('TypeError other than None type error. Cannot set top image using the Reddit algorithm. Possible error with PIL., %s' % e)
        except Exception as e:
            log.critical('Other error with setting top image using the Reddit algorithm. Possible error with PIL, %s' % e)

    def set_title(self, input_title):
        if False:
            print('Hello World!')
        if input_title:
            self.title = input_title[:self.config.MAX_TITLE]

    def set_text(self, text):
        if False:
            while True:
                i = 10
        text = text[:self.config.MAX_TEXT]
        if text:
            self.text = text

    def set_html(self, html):
        if False:
            return 10
        'Encode HTML before setting it\n        '
        if html:
            if isinstance(html, bytes):
                html = self.config.get_parser().get_unicode_html(html)
            self.html = html
            self.download_state = ArticleDownloadState.SUCCESS

    def set_article_html(self, article_html):
        if False:
            for i in range(10):
                print('nop')
        "Sets the HTML of just the article's `top_node`\n        "
        if article_html:
            self.article_html = article_html

    def set_meta_img(self, src_url):
        if False:
            print('Hello World!')
        self.meta_img = src_url
        self.set_top_img_no_check(src_url)

    def set_top_img(self, src_url):
        if False:
            return 10
        if src_url is not None:
            s = images.Scraper(self)
            if s.satisfies_requirements(src_url):
                self.set_top_img_no_check(src_url)

    def set_top_img_no_check(self, src_url):
        if False:
            for i in range(10):
                print('nop')
        'Provide 2 APIs for images. One at "top_img", "imgs"\n        and one at "top_image", "images"\n        '
        self.top_img = src_url
        self.top_image = src_url

    def set_imgs(self, imgs):
        if False:
            return 10
        'The motive for this method is the same as above, provide APIs\n        for both `article.imgs` and `article.images`\n        '
        self.images = imgs
        self.imgs = imgs

    def set_keywords(self, keywords):
        if False:
            while True:
                i = 10
        'Keys are stored in list format\n        '
        if not isinstance(keywords, list):
            raise Exception('Keyword input must be list!')
        if keywords:
            self.keywords = keywords[:self.config.MAX_KEYWORDS]

    def set_authors(self, authors):
        if False:
            i = 10
            return i + 15
        'Authors are in ["firstName lastName", "firstName lastName"] format\n        '
        if not isinstance(authors, list):
            raise Exception('authors input must be list!')
        if authors:
            self.authors = authors[:self.config.MAX_AUTHORS]

    def set_summary(self, summary):
        if False:
            i = 10
            return i + 15
        'Summary here refers to a paragraph of text from the\n        title text and body text\n        '
        self.summary = summary[:self.config.MAX_SUMMARY]

    def set_meta_language(self, meta_lang):
        if False:
            print('Hello World!')
        'Save langauges in their ISO 2-character form\n        '
        if meta_lang and len(meta_lang) >= 2 and (meta_lang in get_available_languages()):
            self.meta_lang = meta_lang[:2]

    def set_meta_keywords(self, meta_keywords):
        if False:
            for i in range(10):
                print('nop')
        'Store the keys in list form\n        '
        self.meta_keywords = [k.strip() for k in meta_keywords.split(',')]

    def set_meta_favicon(self, meta_favicon):
        if False:
            print('Hello World!')
        self.meta_favicon = meta_favicon

    def set_meta_site_name(self, meta_site_name):
        if False:
            print('Hello World!')
        self.meta_site_name = meta_site_name

    def set_meta_description(self, meta_description):
        if False:
            for i in range(10):
                print('nop')
        self.meta_description = meta_description

    def set_meta_data(self, meta_data):
        if False:
            return 10
        self.meta_data = meta_data

    def set_canonical_link(self, canonical_link):
        if False:
            for i in range(10):
                print('nop')
        self.canonical_link = canonical_link

    def set_tags(self, tags):
        if False:
            for i in range(10):
                print('nop')
        self.tags = tags

    def set_movies(self, movie_objects):
        if False:
            print('Hello World!')
        'Trim video objects into just urls\n        '
        movie_urls = [o.src for o in movie_objects if o and o.src]
        self.movies = movie_urls

    def throw_if_not_downloaded_verbose(self):
        if False:
            return 10
        'Parse ArticleDownloadState -> log readable status\n        -> maybe throw ArticleException\n        '
        if self.download_state == ArticleDownloadState.NOT_STARTED:
            raise ArticleException('You must `download()` an article first!')
        elif self.download_state == ArticleDownloadState.FAILED_RESPONSE:
            raise ArticleException('Article `download()` failed with %s on URL %s' % (self.download_exception_msg, self.url))

    def throw_if_not_parsed_verbose(self):
        if False:
            return 10
        'Parse `is_parsed` status -> log readable status\n        -> maybe throw ArticleException\n        '
        if not self.is_parsed:
            raise ArticleException('You must `parse()` an article first!')