import gzip
from io import BytesIO
import json
import logging
import os
import posixpath
import re
try:
    import threading
except ImportError:
    import dummy_threading as threading
import zlib
from . import DistlibException
from .compat import urljoin, urlparse, urlunparse, url2pathname, pathname2url, queue, quote, unescape, build_opener, HTTPRedirectHandler as BaseRedirectHandler, text_type, Request, HTTPError, URLError
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import cached_property, ensure_slash, split_filename, get_project_data, parse_requirement, parse_name_and_version, ServerProxy, normalize_name
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible
logger = logging.getLogger(__name__)
HASHER_HASH = re.compile('^(\\w+)=([a-f0-9]+)')
CHARSET = re.compile(';\\s*charset\\s*=\\s*(.*)\\s*$', re.I)
HTML_CONTENT_TYPE = re.compile('text/html|application/x(ht)?ml')
DEFAULT_INDEX = 'https://pypi.org/pypi'

def get_all_distribution_names(url=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return all distribution names known by an index.\n    :param url: The URL of the index.\n    :return: A list of all known distribution names.\n    '
    if url is None:
        url = DEFAULT_INDEX
    client = ServerProxy(url, timeout=3.0)
    try:
        return client.list_packages()
    finally:
        client('close')()

class RedirectHandler(BaseRedirectHandler):
    """
    A class to work around a bug in some Python 3.2.x releases.
    """

    def http_error_302(self, req, fp, code, msg, headers):
        if False:
            print('Hello World!')
        newurl = None
        for key in ('location', 'uri'):
            if key in headers:
                newurl = headers[key]
                break
        if newurl is None:
            return
        urlparts = urlparse(newurl)
        if urlparts.scheme == '':
            newurl = urljoin(req.get_full_url(), newurl)
            if hasattr(headers, 'replace_header'):
                headers.replace_header(key, newurl)
            else:
                headers[key] = newurl
        return BaseRedirectHandler.http_error_302(self, req, fp, code, msg, headers)
    http_error_301 = http_error_303 = http_error_307 = http_error_302

class Locator(object):
    """
    A base class for locators - things that locate distributions.
    """
    source_extensions = ('.tar.gz', '.tar.bz2', '.tar', '.zip', '.tgz', '.tbz')
    binary_extensions = ('.egg', '.exe', '.whl')
    excluded_extensions = ('.pdf',)
    wheel_tags = None
    downloadable_extensions = source_extensions + ('.whl',)

    def __init__(self, scheme='default'):
        if False:
            while True:
                i = 10
        "\n        Initialise an instance.\n        :param scheme: Because locators look for most recent versions, they\n                       need to know the version scheme to use. This specifies\n                       the current PEP-recommended scheme - use ``'legacy'``\n                       if you need to support existing distributions on PyPI.\n        "
        self._cache = {}
        self.scheme = scheme
        self.opener = build_opener(RedirectHandler())
        self.matcher = None
        self.errors = queue.Queue()

    def get_errors(self):
        if False:
            while True:
                i = 10
        '\n        Return any errors which have occurred.\n        '
        result = []
        while not self.errors.empty():
            try:
                e = self.errors.get(False)
                result.append(e)
            except self.errors.Empty:
                continue
            self.errors.task_done()
        return result

    def clear_errors(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clear any errors which may have been logged.\n        '
        self.get_errors()

    def clear_cache(self):
        if False:
            i = 10
            return i + 15
        self._cache.clear()

    def _get_scheme(self):
        if False:
            i = 10
            return i + 15
        return self._scheme

    def _set_scheme(self, value):
        if False:
            while True:
                i = 10
        self._scheme = value
    scheme = property(_get_scheme, _set_scheme)

    def _get_project(self, name):
        if False:
            print('Hello World!')
        '\n        For a given project, get a dictionary mapping available versions to Distribution\n        instances.\n\n        This should be implemented in subclasses.\n\n        If called from a locate() request, self.matcher will be set to a\n        matcher for the requirement to satisfy, otherwise it will be None.\n        '
        raise NotImplementedError('Please implement in the subclass')

    def get_distribution_names(self):
        if False:
            return 10
        '\n        Return all the distribution names known to this locator.\n        '
        raise NotImplementedError('Please implement in the subclass')

    def get_project(self, name):
        if False:
            return 10
        '\n        For a given project, get a dictionary mapping available versions to Distribution\n        instances.\n\n        This calls _get_project to do all the work, and just implements a caching layer on top.\n        '
        if self._cache is None:
            result = self._get_project(name)
        elif name in self._cache:
            result = self._cache[name]
        else:
            self.clear_errors()
            result = self._get_project(name)
            self._cache[name] = result
        return result

    def score_url(self, url):
        if False:
            while True:
                i = 10
        '\n        Give an url a score which can be used to choose preferred URLs\n        for a given project release.\n        '
        t = urlparse(url)
        basename = posixpath.basename(t.path)
        compatible = True
        is_wheel = basename.endswith('.whl')
        is_downloadable = basename.endswith(self.downloadable_extensions)
        if is_wheel:
            compatible = is_compatible(Wheel(basename), self.wheel_tags)
        return (t.scheme == 'https', 'pypi.org' in t.netloc, is_downloadable, is_wheel, compatible, basename)

    def prefer_url(self, url1, url2):
        if False:
            for i in range(10):
                print('nop')
        '\n        Choose one of two URLs where both are candidates for distribution\n        archives for the same version of a distribution (for example,\n        .tar.gz vs. zip).\n\n        The current implementation favours https:// URLs over http://, archives\n        from PyPI over those from other locations, wheel compatibility (if a\n        wheel) and then the archive name.\n        '
        result = url2
        if url1:
            s1 = self.score_url(url1)
            s2 = self.score_url(url2)
            if s1 > s2:
                result = url1
            if result != url2:
                logger.debug('Not replacing %r with %r', url1, url2)
            else:
                logger.debug('Replacing %r with %r', url1, url2)
        return result

    def split_filename(self, filename, project_name):
        if False:
            while True:
                i = 10
        '\n        Attempt to split a filename in project name, version and Python version.\n        '
        return split_filename(filename, project_name)

    def convert_url_to_download_info(self, url, project_name):
        if False:
            return 10
        '\n        See if a URL is a candidate for a download URL for a project (the URL\n        has typically been scraped from an HTML page).\n\n        If it is, a dictionary is returned with keys "name", "version",\n        "filename" and "url"; otherwise, None is returned.\n        '

        def same_project(name1, name2):
            if False:
                return 10
            return normalize_name(name1) == normalize_name(name2)
        result = None
        (scheme, netloc, path, params, query, frag) = urlparse(url)
        if frag.lower().startswith('egg='):
            logger.debug('%s: version hint in fragment: %r', project_name, frag)
        m = HASHER_HASH.match(frag)
        if m:
            (algo, digest) = m.groups()
        else:
            (algo, digest) = (None, None)
        origpath = path
        if path and path[-1] == '/':
            path = path[:-1]
        if path.endswith('.whl'):
            try:
                wheel = Wheel(path)
                if not is_compatible(wheel, self.wheel_tags):
                    logger.debug('Wheel not compatible: %s', path)
                else:
                    if project_name is None:
                        include = True
                    else:
                        include = same_project(wheel.name, project_name)
                    if include:
                        result = {'name': wheel.name, 'version': wheel.version, 'filename': wheel.filename, 'url': urlunparse((scheme, netloc, origpath, params, query, '')), 'python-version': ', '.join(['.'.join(list(v[2:])) for v in wheel.pyver])}
            except Exception as e:
                logger.warning('invalid path for wheel: %s', path)
        elif not path.endswith(self.downloadable_extensions):
            logger.debug('Not downloadable: %s', path)
        else:
            path = filename = posixpath.basename(path)
            for ext in self.downloadable_extensions:
                if path.endswith(ext):
                    path = path[:-len(ext)]
                    t = self.split_filename(path, project_name)
                    if not t:
                        logger.debug('No match for project/version: %s', path)
                    else:
                        (name, version, pyver) = t
                        if not project_name or same_project(project_name, name):
                            result = {'name': name, 'version': version, 'filename': filename, 'url': urlunparse((scheme, netloc, origpath, params, query, ''))}
                            if pyver:
                                result['python-version'] = pyver
                    break
        if result and algo:
            result['%s_digest' % algo] = digest
        return result

    def _get_digest(self, info):
        if False:
            return 10
        '\n        Get a digest from a dictionary by looking at a "digests" dictionary\n        or keys of the form \'algo_digest\'.\n\n        Returns a 2-tuple (algo, digest) if found, else None. Currently\n        looks only for SHA256, then MD5.\n        '
        result = None
        if 'digests' in info:
            digests = info['digests']
            for algo in ('sha256', 'md5'):
                if algo in digests:
                    result = (algo, digests[algo])
                    break
        if not result:
            for algo in ('sha256', 'md5'):
                key = '%s_digest' % algo
                if key in info:
                    result = (algo, info[key])
                    break
        return result

    def _update_version_data(self, result, info):
        if False:
            return 10
        '\n        Update a result dictionary (the final result from _get_project) with a\n        dictionary for a specific version, which typically holds information\n        gleaned from a filename or URL for an archive for the distribution.\n        '
        name = info.pop('name')
        version = info.pop('version')
        if version in result:
            dist = result[version]
            md = dist.metadata
        else:
            dist = make_dist(name, version, scheme=self.scheme)
            md = dist.metadata
        dist.digest = digest = self._get_digest(info)
        url = info['url']
        result['digests'][url] = digest
        if md.source_url != info['url']:
            md.source_url = self.prefer_url(md.source_url, url)
            result['urls'].setdefault(version, set()).add(url)
        dist.locator = self
        result[version] = dist

    def locate(self, requirement, prereleases=False):
        if False:
            while True:
                i = 10
        "\n        Find the most recent distribution which matches the given\n        requirement.\n\n        :param requirement: A requirement of the form 'foo (1.0)' or perhaps\n                            'foo (>= 1.0, < 2.0, != 1.3)'\n        :param prereleases: If ``True``, allow pre-release versions\n                            to be located. Otherwise, pre-release versions\n                            are not returned.\n        :return: A :class:`Distribution` instance, or ``None`` if no such\n                 distribution could be located.\n        "
        result = None
        r = parse_requirement(requirement)
        if r is None:
            raise DistlibException('Not a valid requirement: %r' % requirement)
        scheme = get_scheme(self.scheme)
        self.matcher = matcher = scheme.matcher(r.requirement)
        logger.debug('matcher: %s (%s)', matcher, type(matcher).__name__)
        versions = self.get_project(r.name)
        if len(versions) > 2:
            slist = []
            vcls = matcher.version_class
            for k in versions:
                if k in ('urls', 'digests'):
                    continue
                try:
                    if not matcher.match(k):
                        pass
                    elif prereleases or not vcls(k).is_prerelease:
                        slist.append(k)
                except Exception:
                    logger.warning('error matching %s with %r', matcher, k)
                    pass
            if len(slist) > 1:
                slist = sorted(slist, key=scheme.key)
            if slist:
                logger.debug('sorted list: %s', slist)
                version = slist[-1]
                result = versions[version]
        if result:
            if r.extras:
                result.extras = r.extras
            result.download_urls = versions.get('urls', {}).get(version, set())
            d = {}
            sd = versions.get('digests', {})
            for url in result.download_urls:
                if url in sd:
                    d[url] = sd[url]
            result.digests = d
        self.matcher = None
        return result

class PyPIRPCLocator(Locator):
    """
    This locator uses XML-RPC to locate distributions. It therefore
    cannot be used with simple mirrors (that only mirror file content).
    """

    def __init__(self, url, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialise an instance.\n\n        :param url: The URL to use for XML-RPC.\n        :param kwargs: Passed to the superclass constructor.\n        '
        super(PyPIRPCLocator, self).__init__(**kwargs)
        self.base_url = url
        self.client = ServerProxy(url, timeout=3.0)

    def get_distribution_names(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all the distribution names known to this locator.\n        '
        return set(self.client.list_packages())

    def _get_project(self, name):
        if False:
            for i in range(10):
                print('nop')
        result = {'urls': {}, 'digests': {}}
        versions = self.client.package_releases(name, True)
        for v in versions:
            urls = self.client.release_urls(name, v)
            data = self.client.release_data(name, v)
            metadata = Metadata(scheme=self.scheme)
            metadata.name = data['name']
            metadata.version = data['version']
            metadata.license = data.get('license')
            metadata.keywords = data.get('keywords', [])
            metadata.summary = data.get('summary')
            dist = Distribution(metadata)
            if urls:
                info = urls[0]
                metadata.source_url = info['url']
                dist.digest = self._get_digest(info)
                dist.locator = self
                result[v] = dist
                for info in urls:
                    url = info['url']
                    digest = self._get_digest(info)
                    result['urls'].setdefault(v, set()).add(url)
                    result['digests'][url] = digest
        return result

class PyPIJSONLocator(Locator):
    """
    This locator uses PyPI's JSON interface. It's very limited in functionality
    and probably not worth using.
    """

    def __init__(self, url, **kwargs):
        if False:
            while True:
                i = 10
        super(PyPIJSONLocator, self).__init__(**kwargs)
        self.base_url = ensure_slash(url)

    def get_distribution_names(self):
        if False:
            print('Hello World!')
        '\n        Return all the distribution names known to this locator.\n        '
        raise NotImplementedError('Not available from this locator')

    def _get_project(self, name):
        if False:
            while True:
                i = 10
        result = {'urls': {}, 'digests': {}}
        url = urljoin(self.base_url, '%s/json' % quote(name))
        try:
            resp = self.opener.open(url)
            data = resp.read().decode()
            d = json.loads(data)
            md = Metadata(scheme=self.scheme)
            data = d['info']
            md.name = data['name']
            md.version = data['version']
            md.license = data.get('license')
            md.keywords = data.get('keywords', [])
            md.summary = data.get('summary')
            dist = Distribution(md)
            dist.locator = self
            urls = d['urls']
            result[md.version] = dist
            for info in d['urls']:
                url = info['url']
                dist.download_urls.add(url)
                dist.digests[url] = self._get_digest(info)
                result['urls'].setdefault(md.version, set()).add(url)
                result['digests'][url] = self._get_digest(info)
            for (version, infos) in d['releases'].items():
                if version == md.version:
                    continue
                omd = Metadata(scheme=self.scheme)
                omd.name = md.name
                omd.version = version
                odist = Distribution(omd)
                odist.locator = self
                result[version] = odist
                for info in infos:
                    url = info['url']
                    odist.download_urls.add(url)
                    odist.digests[url] = self._get_digest(info)
                    result['urls'].setdefault(version, set()).add(url)
                    result['digests'][url] = self._get_digest(info)
        except Exception as e:
            self.errors.put(text_type(e))
            logger.exception('JSON fetch failed: %s', e)
        return result

class Page(object):
    """
    This class represents a scraped HTML page.
    """
    _href = re.compile('\n(rel\\s*=\\s*(?:"(?P<rel1>[^"]*)"|\'(?P<rel2>[^\']*)\'|(?P<rel3>[^>\\s\n]*))\\s+)?\nhref\\s*=\\s*(?:"(?P<url1>[^"]*)"|\'(?P<url2>[^\']*)\'|(?P<url3>[^>\\s\n]*))\n(\\s+rel\\s*=\\s*(?:"(?P<rel4>[^"]*)"|\'(?P<rel5>[^\']*)\'|(?P<rel6>[^>\\s\n]*)))?\n', re.I | re.S | re.X)
    _base = re.compile('<base\\s+href\\s*=\\s*[\'"]?([^\'">]+)', re.I | re.S)

    def __init__(self, data, url):
        if False:
            while True:
                i = 10
        '\n        Initialise an instance with the Unicode page contents and the URL they\n        came from.\n        '
        self.data = data
        self.base_url = self.url = url
        m = self._base.search(self.data)
        if m:
            self.base_url = m.group(1)
    _clean_re = re.compile('[^a-z0-9$&+,/:;=?@.#%_\\\\|-]', re.I)

    @cached_property
    def links(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the URLs of all the links on a page together with information\n        about their "rel" attribute, for determining which ones to treat as\n        downloads and which ones to queue for further scraping.\n        '

        def clean(url):
            if False:
                while True:
                    i = 10
            'Tidy up an URL.'
            (scheme, netloc, path, params, query, frag) = urlparse(url)
            return urlunparse((scheme, netloc, quote(path), params, query, frag))
        result = set()
        for match in self._href.finditer(self.data):
            d = match.groupdict('')
            rel = d['rel1'] or d['rel2'] or d['rel3'] or d['rel4'] or d['rel5'] or d['rel6']
            url = d['url1'] or d['url2'] or d['url3']
            url = urljoin(self.base_url, url)
            url = unescape(url)
            url = self._clean_re.sub(lambda m: '%%%2x' % ord(m.group(0)), url)
            result.add((url, rel))
        result = sorted(result, key=lambda t: t[0], reverse=True)
        return result

class SimpleScrapingLocator(Locator):
    """
    A locator which scrapes HTML pages to locate downloads for a distribution.
    This runs multiple threads to do the I/O; performance is at least as good
    as pip's PackageFinder, which works in an analogous fashion.
    """
    decoders = {'deflate': zlib.decompress, 'gzip': lambda b: gzip.GzipFile(fileobj=BytesIO(b)).read(), 'none': lambda b: b}

    def __init__(self, url, timeout=None, num_workers=10, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialise an instance.\n        :param url: The root URL to use for scraping.\n        :param timeout: The timeout, in seconds, to be applied to requests.\n                        This defaults to ``None`` (no timeout specified).\n        :param num_workers: The number of worker threads you want to do I/O,\n                            This defaults to 10.\n        :param kwargs: Passed to the superclass.\n        '
        super(SimpleScrapingLocator, self).__init__(**kwargs)
        self.base_url = ensure_slash(url)
        self.timeout = timeout
        self._page_cache = {}
        self._seen = set()
        self._to_fetch = queue.Queue()
        self._bad_hosts = set()
        self.skip_externals = False
        self.num_workers = num_workers
        self._lock = threading.RLock()
        self._gplock = threading.RLock()
        self.platform_check = False

    def _prepare_threads(self):
        if False:
            return 10
        '\n        Threads are created only when get_project is called, and terminate\n        before it returns. They are there primarily to parallelise I/O (i.e.\n        fetching web pages).\n        '
        self._threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._fetch)
            t.daemon = True
            t.start()
            self._threads.append(t)

    def _wait_threads(self):
        if False:
            print('Hello World!')
        '\n        Tell all the threads to terminate (by sending a sentinel value) and\n        wait for them to do so.\n        '
        for t in self._threads:
            self._to_fetch.put(None)
        for t in self._threads:
            t.join()
        self._threads = []

    def _get_project(self, name):
        if False:
            i = 10
            return i + 15
        result = {'urls': {}, 'digests': {}}
        with self._gplock:
            self.result = result
            self.project_name = name
            url = urljoin(self.base_url, '%s/' % quote(name))
            self._seen.clear()
            self._page_cache.clear()
            self._prepare_threads()
            try:
                logger.debug('Queueing %s', url)
                self._to_fetch.put(url)
                self._to_fetch.join()
            finally:
                self._wait_threads()
            del self.result
        return result
    platform_dependent = re.compile('\\b(linux_(i\\d86|x86_64|arm\\w+)|win(32|_amd64)|macosx_?\\d+)\\b', re.I)

    def _is_platform_dependent(self, url):
        if False:
            i = 10
            return i + 15
        '\n        Does an URL refer to a platform-specific download?\n        '
        return self.platform_dependent.search(url)

    def _process_download(self, url):
        if False:
            for i in range(10):
                print('nop')
        "\n        See if an URL is a suitable download for a project.\n\n        If it is, register information in the result dictionary (for\n        _get_project) about the specific version it's for.\n\n        Note that the return value isn't actually used other than as a boolean\n        value.\n        "
        if self.platform_check and self._is_platform_dependent(url):
            info = None
        else:
            info = self.convert_url_to_download_info(url, self.project_name)
        logger.debug('process_download: %s -> %s', url, info)
        if info:
            with self._lock:
                self._update_version_data(self.result, info)
        return info

    def _should_queue(self, link, referrer, rel):
        if False:
            i = 10
            return i + 15
        '\n        Determine whether a link URL from a referring page and with a\n        particular "rel" attribute should be queued for scraping.\n        '
        (scheme, netloc, path, _, _, _) = urlparse(link)
        if path.endswith(self.source_extensions + self.binary_extensions + self.excluded_extensions):
            result = False
        elif self.skip_externals and (not link.startswith(self.base_url)):
            result = False
        elif not referrer.startswith(self.base_url):
            result = False
        elif rel not in ('homepage', 'download'):
            result = False
        elif scheme not in ('http', 'https', 'ftp'):
            result = False
        elif self._is_platform_dependent(link):
            result = False
        else:
            host = netloc.split(':', 1)[0]
            if host.lower() == 'localhost':
                result = False
            else:
                result = True
        logger.debug('should_queue: %s (%s) from %s -> %s', link, rel, referrer, result)
        return result

    def _fetch(self):
        if False:
            return 10
        '\n        Get a URL to fetch from the work queue, get the HTML page, examine its\n        links for download candidates and candidates for further scraping.\n\n        This is a handy method to run in a thread.\n        '
        while True:
            url = self._to_fetch.get()
            try:
                if url:
                    page = self.get_page(url)
                    if page is None:
                        continue
                    for (link, rel) in page.links:
                        if link not in self._seen:
                            try:
                                self._seen.add(link)
                                if not self._process_download(link) and self._should_queue(link, url, rel):
                                    logger.debug('Queueing %s from %s', link, url)
                                    self._to_fetch.put(link)
                            except MetadataInvalidError:
                                pass
            except Exception as e:
                self.errors.put(text_type(e))
            finally:
                self._to_fetch.task_done()
            if not url:
                break

    def get_page(self, url):
        if False:
            while True:
                i = 10
        "\n        Get the HTML for an URL, possibly from an in-memory cache.\n\n        XXX TODO Note: this cache is never actually cleared. It's assumed that\n        the data won't get stale over the lifetime of a locator instance (not\n        necessarily true for the default_locator).\n        "
        (scheme, netloc, path, _, _, _) = urlparse(url)
        if scheme == 'file' and os.path.isdir(url2pathname(path)):
            url = urljoin(ensure_slash(url), 'index.html')
        if url in self._page_cache:
            result = self._page_cache[url]
            logger.debug('Returning %s from cache: %s', url, result)
        else:
            host = netloc.split(':', 1)[0]
            result = None
            if host in self._bad_hosts:
                logger.debug('Skipping %s due to bad host %s', url, host)
            else:
                req = Request(url, headers={'Accept-encoding': 'identity'})
                try:
                    logger.debug('Fetching %s', url)
                    resp = self.opener.open(req, timeout=self.timeout)
                    logger.debug('Fetched %s', url)
                    headers = resp.info()
                    content_type = headers.get('Content-Type', '')
                    if HTML_CONTENT_TYPE.match(content_type):
                        final_url = resp.geturl()
                        data = resp.read()
                        encoding = headers.get('Content-Encoding')
                        if encoding:
                            decoder = self.decoders[encoding]
                            data = decoder(data)
                        encoding = 'utf-8'
                        m = CHARSET.search(content_type)
                        if m:
                            encoding = m.group(1)
                        try:
                            data = data.decode(encoding)
                        except UnicodeError:
                            data = data.decode('latin-1')
                        result = Page(data, final_url)
                        self._page_cache[final_url] = result
                except HTTPError as e:
                    if e.code != 404:
                        logger.exception('Fetch failed: %s: %s', url, e)
                except URLError as e:
                    logger.exception('Fetch failed: %s: %s', url, e)
                    with self._lock:
                        self._bad_hosts.add(host)
                except Exception as e:
                    logger.exception('Fetch failed: %s: %s', url, e)
                finally:
                    self._page_cache[url] = result
        return result
    _distname_re = re.compile('<a href=[^>]*>([^<]+)<')

    def get_distribution_names(self):
        if False:
            while True:
                i = 10
        '\n        Return all the distribution names known to this locator.\n        '
        result = set()
        page = self.get_page(self.base_url)
        if not page:
            raise DistlibException('Unable to get %s' % self.base_url)
        for match in self._distname_re.finditer(page.data):
            result.add(match.group(1))
        return result

class DirectoryLocator(Locator):
    """
    This class locates distributions in a directory tree.
    """

    def __init__(self, path, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialise an instance.\n        :param path: The root of the directory tree to search.\n        :param kwargs: Passed to the superclass constructor,\n                       except for:\n                       * recursive - if True (the default), subdirectories are\n                         recursed into. If False, only the top-level directory\n                         is searched,\n        '
        self.recursive = kwargs.pop('recursive', True)
        super(DirectoryLocator, self).__init__(**kwargs)
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise DistlibException('Not a directory: %r' % path)
        self.base_dir = path

    def should_include(self, filename, parent):
        if False:
            return 10
        '\n        Should a filename be considered as a candidate for a distribution\n        archive? As well as the filename, the directory which contains it\n        is provided, though not used by the current implementation.\n        '
        return filename.endswith(self.downloadable_extensions)

    def _get_project(self, name):
        if False:
            for i in range(10):
                print('nop')
        result = {'urls': {}, 'digests': {}}
        for (root, dirs, files) in os.walk(self.base_dir):
            for fn in files:
                if self.should_include(fn, root):
                    fn = os.path.join(root, fn)
                    url = urlunparse(('file', '', pathname2url(os.path.abspath(fn)), '', '', ''))
                    info = self.convert_url_to_download_info(url, name)
                    if info:
                        self._update_version_data(result, info)
            if not self.recursive:
                break
        return result

    def get_distribution_names(self):
        if False:
            return 10
        '\n        Return all the distribution names known to this locator.\n        '
        result = set()
        for (root, dirs, files) in os.walk(self.base_dir):
            for fn in files:
                if self.should_include(fn, root):
                    fn = os.path.join(root, fn)
                    url = urlunparse(('file', '', pathname2url(os.path.abspath(fn)), '', '', ''))
                    info = self.convert_url_to_download_info(url, None)
                    if info:
                        result.add(info['name'])
            if not self.recursive:
                break
        return result

class JSONLocator(Locator):
    """
    This locator uses special extended metadata (not available on PyPI) and is
    the basis of performant dependency resolution in distlib. Other locators
    require archive downloads before dependencies can be determined! As you
    might imagine, that can be slow.
    """

    def get_distribution_names(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all the distribution names known to this locator.\n        '
        raise NotImplementedError('Not available from this locator')

    def _get_project(self, name):
        if False:
            i = 10
            return i + 15
        result = {'urls': {}, 'digests': {}}
        data = get_project_data(name)
        if data:
            for info in data.get('files', []):
                if info['ptype'] != 'sdist' or info['pyversion'] != 'source':
                    continue
                dist = make_dist(data['name'], info['version'], summary=data.get('summary', 'Placeholder for summary'), scheme=self.scheme)
                md = dist.metadata
                md.source_url = info['url']
                if 'digest' in info and info['digest']:
                    dist.digest = ('md5', info['digest'])
                md.dependencies = info.get('requirements', {})
                dist.exports = info.get('exports', {})
                result[dist.version] = dist
                result['urls'].setdefault(dist.version, set()).add(info['url'])
        return result

class DistPathLocator(Locator):
    """
    This locator finds installed distributions in a path. It can be useful for
    adding to an :class:`AggregatingLocator`.
    """

    def __init__(self, distpath, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialise an instance.\n\n        :param distpath: A :class:`DistributionPath` instance to search.\n        '
        super(DistPathLocator, self).__init__(**kwargs)
        assert isinstance(distpath, DistributionPath)
        self.distpath = distpath

    def _get_project(self, name):
        if False:
            while True:
                i = 10
        dist = self.distpath.get_distribution(name)
        if dist is None:
            result = {'urls': {}, 'digests': {}}
        else:
            result = {dist.version: dist, 'urls': {dist.version: set([dist.source_url])}, 'digests': {dist.version: set([None])}}
        return result

class AggregatingLocator(Locator):
    """
    This class allows you to chain and/or merge a list of locators.
    """

    def __init__(self, *locators, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialise an instance.\n\n        :param locators: The list of locators to search.\n        :param kwargs: Passed to the superclass constructor,\n                       except for:\n                       * merge - if False (the default), the first successful\n                         search from any of the locators is returned. If True,\n                         the results from all locators are merged (this can be\n                         slow).\n        '
        self.merge = kwargs.pop('merge', False)
        self.locators = locators
        super(AggregatingLocator, self).__init__(**kwargs)

    def clear_cache(self):
        if False:
            print('Hello World!')
        super(AggregatingLocator, self).clear_cache()
        for locator in self.locators:
            locator.clear_cache()

    def _set_scheme(self, value):
        if False:
            while True:
                i = 10
        self._scheme = value
        for locator in self.locators:
            locator.scheme = value
    scheme = property(Locator.scheme.fget, _set_scheme)

    def _get_project(self, name):
        if False:
            print('Hello World!')
        result = {}
        for locator in self.locators:
            d = locator.get_project(name)
            if d:
                if self.merge:
                    files = result.get('urls', {})
                    digests = result.get('digests', {})
                    result.update(d)
                    df = result.get('urls')
                    if files and df:
                        for (k, v) in files.items():
                            if k in df:
                                df[k] |= v
                            else:
                                df[k] = v
                    dd = result.get('digests')
                    if digests and dd:
                        dd.update(digests)
                else:
                    if self.matcher is None:
                        found = True
                    else:
                        found = False
                        for k in d:
                            if self.matcher.match(k):
                                found = True
                                break
                    if found:
                        result = d
                        break
        return result

    def get_distribution_names(self):
        if False:
            i = 10
            return i + 15
        '\n        Return all the distribution names known to this locator.\n        '
        result = set()
        for locator in self.locators:
            try:
                result |= locator.get_distribution_names()
            except NotImplementedError:
                pass
        return result
default_locator = AggregatingLocator(SimpleScrapingLocator('https://pypi.org/simple/', timeout=3.0), scheme='legacy')
locate = default_locator.locate

class DependencyFinder(object):
    """
    Locate dependencies for distributions.
    """

    def __init__(self, locator=None):
        if False:
            return 10
        '\n        Initialise an instance, using the specified locator\n        to locate distributions.\n        '
        self.locator = locator or default_locator
        self.scheme = get_scheme(self.locator.scheme)

    def add_distribution(self, dist):
        if False:
            return 10
        '\n        Add a distribution to the finder. This will update internal information\n        about who provides what.\n        :param dist: The distribution to add.\n        '
        logger.debug('adding distribution %s', dist)
        name = dist.key
        self.dists_by_name[name] = dist
        self.dists[name, dist.version] = dist
        for p in dist.provides:
            (name, version) = parse_name_and_version(p)
            logger.debug('Add to provided: %s, %s, %s', name, version, dist)
            self.provided.setdefault(name, set()).add((version, dist))

    def remove_distribution(self, dist):
        if False:
            print('Hello World!')
        '\n        Remove a distribution from the finder. This will update internal\n        information about who provides what.\n        :param dist: The distribution to remove.\n        '
        logger.debug('removing distribution %s', dist)
        name = dist.key
        del self.dists_by_name[name]
        del self.dists[name, dist.version]
        for p in dist.provides:
            (name, version) = parse_name_and_version(p)
            logger.debug('Remove from provided: %s, %s, %s', name, version, dist)
            s = self.provided[name]
            s.remove((version, dist))
            if not s:
                del self.provided[name]

    def get_matcher(self, reqt):
        if False:
            return 10
        '\n        Get a version matcher for a requirement.\n        :param reqt: The requirement\n        :type reqt: str\n        :return: A version matcher (an instance of\n                 :class:`distlib.version.Matcher`).\n        '
        try:
            matcher = self.scheme.matcher(reqt)
        except UnsupportedVersionError:
            name = reqt.split()[0]
            matcher = self.scheme.matcher(name)
        return matcher

    def find_providers(self, reqt):
        if False:
            return 10
        '\n        Find the distributions which can fulfill a requirement.\n\n        :param reqt: The requirement.\n         :type reqt: str\n        :return: A set of distribution which can fulfill the requirement.\n        '
        matcher = self.get_matcher(reqt)
        name = matcher.key
        result = set()
        provided = self.provided
        if name in provided:
            for (version, provider) in provided[name]:
                try:
                    match = matcher.match(version)
                except UnsupportedVersionError:
                    match = False
                if match:
                    result.add(provider)
                    break
        return result

    def try_to_replace(self, provider, other, problems):
        if False:
            while True:
                i = 10
        "\n        Attempt to replace one provider with another. This is typically used\n        when resolving dependencies from multiple sources, e.g. A requires\n        (B >= 1.0) while C requires (B >= 1.1).\n\n        For successful replacement, ``provider`` must meet all the requirements\n        which ``other`` fulfills.\n\n        :param provider: The provider we are trying to replace with.\n        :param other: The provider we're trying to replace.\n        :param problems: If False is returned, this will contain what\n                         problems prevented replacement. This is currently\n                         a tuple of the literal string 'cantreplace',\n                         ``provider``, ``other``  and the set of requirements\n                         that ``provider`` couldn't fulfill.\n        :return: True if we can replace ``other`` with ``provider``, else\n                 False.\n        "
        rlist = self.reqts[other]
        unmatched = set()
        for s in rlist:
            matcher = self.get_matcher(s)
            if not matcher.match(provider.version):
                unmatched.add(s)
        if unmatched:
            problems.add(('cantreplace', provider, other, frozenset(unmatched)))
            result = False
        else:
            self.remove_distribution(other)
            del self.reqts[other]
            for s in rlist:
                self.reqts.setdefault(provider, set()).add(s)
            self.add_distribution(provider)
            result = True
        return result

    def find(self, requirement, meta_extras=None, prereleases=False):
        if False:
            while True:
                i = 10
        "\n        Find a distribution and all distributions it depends on.\n\n        :param requirement: The requirement specifying the distribution to\n                            find, or a Distribution instance.\n        :param meta_extras: A list of meta extras such as :test:, :build: and\n                            so on.\n        :param prereleases: If ``True``, allow pre-release versions to be\n                            returned - otherwise, don't return prereleases\n                            unless they're all that's available.\n\n        Return a set of :class:`Distribution` instances and a set of\n        problems.\n\n        The distributions returned should be such that they have the\n        :attr:`required` attribute set to ``True`` if they were\n        from the ``requirement`` passed to ``find()``, and they have the\n        :attr:`build_time_dependency` attribute set to ``True`` unless they\n        are post-installation dependencies of the ``requirement``.\n\n        The problems should be a tuple consisting of the string\n        ``'unsatisfied'`` and the requirement which couldn't be satisfied\n        by any distribution known to the locator.\n        "
        self.provided = {}
        self.dists = {}
        self.dists_by_name = {}
        self.reqts = {}
        meta_extras = set(meta_extras or [])
        if ':*:' in meta_extras:
            meta_extras.remove(':*:')
            meta_extras |= set([':test:', ':build:', ':dev:'])
        if isinstance(requirement, Distribution):
            dist = odist = requirement
            logger.debug('passed %s as requirement', odist)
        else:
            dist = odist = self.locator.locate(requirement, prereleases=prereleases)
            if dist is None:
                raise DistlibException('Unable to locate %r' % requirement)
            logger.debug('located %s', odist)
        dist.requested = True
        problems = set()
        todo = set([dist])
        install_dists = set([odist])
        while todo:
            dist = todo.pop()
            name = dist.key
            if name not in self.dists_by_name:
                self.add_distribution(dist)
            else:
                other = self.dists_by_name[name]
                if other != dist:
                    self.try_to_replace(dist, other, problems)
            ireqts = dist.run_requires | dist.meta_requires
            sreqts = dist.build_requires
            ereqts = set()
            if meta_extras and dist in install_dists:
                for key in ('test', 'build', 'dev'):
                    e = ':%s:' % key
                    if e in meta_extras:
                        ereqts |= getattr(dist, '%s_requires' % key)
            all_reqts = ireqts | sreqts | ereqts
            for r in all_reqts:
                providers = self.find_providers(r)
                if not providers:
                    logger.debug('No providers found for %r', r)
                    provider = self.locator.locate(r, prereleases=prereleases)
                    if provider is None and (not prereleases):
                        provider = self.locator.locate(r, prereleases=True)
                    if provider is None:
                        logger.debug('Cannot satisfy %r', r)
                        problems.add(('unsatisfied', r))
                    else:
                        (n, v) = (provider.key, provider.version)
                        if (n, v) not in self.dists:
                            todo.add(provider)
                        providers.add(provider)
                        if r in ireqts and dist in install_dists:
                            install_dists.add(provider)
                            logger.debug('Adding %s to install_dists', provider.name_and_version)
                for p in providers:
                    name = p.key
                    if name not in self.dists_by_name:
                        self.reqts.setdefault(p, set()).add(r)
                    else:
                        other = self.dists_by_name[name]
                        if other != p:
                            self.try_to_replace(p, other, problems)
        dists = set(self.dists.values())
        for dist in dists:
            dist.build_time_dependency = dist not in install_dists
            if dist.build_time_dependency:
                logger.debug('%s is a build-time dependency only.', dist.name_and_version)
        logger.debug('find done for %s', odist)
        return (dists, problems)