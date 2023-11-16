import sys
import asyncio
from urllib.parse import urlparse, urlunparse, urljoin
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from functools import partial
from typing import Set, Union, List, MutableMapping, Optional
import pyppeteer
import requests
import http.cookiejar
from pyquery import PyQuery
from fake_useragent import UserAgent
from lxml.html.clean import Cleaner
import lxml
from lxml import etree
from lxml.html import HtmlElement
from lxml.html import tostring as lxml_html_tostring
from lxml.html.soupparser import fromstring as soup_parse
from parse import search as parse_search
from parse import findall, Result
from w3lib.encoding import html_to_unicode
DEFAULT_ENCODING = 'utf-8'
DEFAULT_URL = 'https://example.org/'
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/603.3.8 (KHTML, like Gecko) Version/10.1.2 Safari/603.3.8'
DEFAULT_NEXT_SYMBOL = ['next', 'more', 'older']
cleaner = Cleaner()
cleaner.javascript = True
cleaner.style = True
useragent = None
_Find = Union[List['Element'], 'Element']
_XPath = Union[List[str], List['Element'], str, 'Element']
_Result = Union[List['Result'], 'Result']
_HTML = Union[str, bytes]
_BaseHTML = str
_UserAgent = str
_DefaultEncoding = str
_URL = str
_RawHTML = bytes
_Encoding = str
_LXML = HtmlElement
_Text = str
_Search = Result
_Containing = Union[str, List[str]]
_Links = Set[str]
_Attrs = MutableMapping
_Next = Union['HTML', List[str]]
_NextSymbol = List[str]
try:
    assert sys.version_info.major == 3
    assert sys.version_info.minor > 5
except AssertionError:
    raise RuntimeError('Requests-HTML requires Python 3.6+!')

class MaxRetries(Exception):

    def __init__(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.message = message

class BaseParser:
    """A basic HTML/Element Parser, for Humans.

    :param element: The element from which to base the parsing upon.
    :param default_encoding: Which encoding to default to.
    :param html: HTML from which to base the parsing upon (optional).
    :param url: The URL from which the HTML originated, used for ``absolute_links``.

    """

    def __init__(self, *, element, default_encoding: _DefaultEncoding=None, html: _HTML=None, url: _URL) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.element = element
        self.url = url
        self.skip_anchors = True
        self.default_encoding = default_encoding
        self._encoding = None
        self._html = html.encode(DEFAULT_ENCODING) if isinstance(html, str) else html
        self._lxml = None
        self._pq = None

    @property
    def raw_html(self) -> _RawHTML:
        if False:
            while True:
                i = 10
        'Bytes representation of the HTML content.\n        (`learn more <http://www.diveintopython3.net/strings.html>`_).\n        '
        if self._html:
            return self._html
        else:
            return etree.tostring(self.element, encoding='unicode').strip().encode(self.encoding)

    @property
    def html(self) -> _BaseHTML:
        if False:
            return 10
        'Unicode representation of the HTML content\n        (`learn more <http://www.diveintopython3.net/strings.html>`_).\n        '
        if self._html:
            return self.raw_html.decode(self.encoding, errors='replace')
        else:
            return etree.tostring(self.element, encoding='unicode').strip()

    @html.setter
    def html(self, html: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._html = html.encode(self.encoding)

    @raw_html.setter
    def raw_html(self, html: bytes) -> None:
        if False:
            while True:
                i = 10
        'Property setter for self.html.'
        self._html = html

    @property
    def encoding(self) -> _Encoding:
        if False:
            return 10
        'The encoding string to be used, extracted from the HTML and\n        :class:`HTMLResponse <HTMLResponse>` headers.\n        '
        if self._encoding:
            return self._encoding
        if self._html:
            self._encoding = html_to_unicode(self.default_encoding, self._html)[0]
            try:
                self.raw_html.decode(self.encoding, errors='replace')
            except UnicodeDecodeError:
                self._encoding = self.default_encoding
        return self._encoding if self._encoding else self.default_encoding

    @encoding.setter
    def encoding(self, enc: str) -> None:
        if False:
            i = 10
            return i + 15
        'Property setter for self.encoding.'
        self._encoding = enc

    @property
    def pq(self) -> PyQuery:
        if False:
            for i in range(10):
                print('nop')
        '`PyQuery <https://pythonhosted.org/pyquery/>`_ representation\n        of the :class:`Element <Element>` or :class:`HTML <HTML>`.\n        '
        if self._pq is None:
            self._pq = PyQuery(self.lxml)
        return self._pq

    @property
    def lxml(self) -> HtmlElement:
        if False:
            while True:
                i = 10
        '`lxml <http://lxml.de>`_ representation of the\n        :class:`Element <Element>` or :class:`HTML <HTML>`.\n        '
        if self._lxml is None:
            try:
                self._lxml = soup_parse(self.html, features='html.parser')
            except ValueError:
                self._lxml = lxml.html.fromstring(self.raw_html)
        return self._lxml

    @property
    def text(self) -> _Text:
        if False:
            for i in range(10):
                print('nop')
        'The text content of the\n        :class:`Element <Element>` or :class:`HTML <HTML>`.\n        '
        return self.pq.text()

    @property
    def full_text(self) -> _Text:
        if False:
            print('Hello World!')
        'The full text content (including links) of the\n        :class:`Element <Element>` or :class:`HTML <HTML>`.\n        '
        return self.lxml.text_content()

    def find(self, selector: str='*', *, containing: _Containing=None, clean: bool=False, first: bool=False, _encoding: str=None) -> _Find:
        if False:
            while True:
                i = 10
        "Given a CSS Selector, returns a list of\n        :class:`Element <Element>` objects or a single one.\n\n        :param selector: CSS Selector to use.\n        :param clean: Whether or not to sanitize the found HTML of ``<script>`` and ``<style>`` tags.\n        :param containing: If specified, only return elements that contain the provided text.\n        :param first: Whether or not to return just the first result.\n        :param _encoding: The encoding format.\n\n        Example CSS Selectors:\n\n        - ``a``\n        - ``a.someClass``\n        - ``a#someID``\n        - ``a[target=_blank]``\n\n        See W3School's `CSS Selectors Reference\n        <https://www.w3schools.com/cssref/css_selectors.asp>`_\n        for more details.\n\n        If ``first`` is ``True``, only returns the first\n        :class:`Element <Element>` found.\n        "
        if isinstance(containing, str):
            containing = [containing]
        encoding = _encoding or self.encoding
        elements = [Element(element=found, url=self.url, default_encoding=encoding) for found in self.pq(selector)]
        if containing:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                if any([c.lower() in element.full_text.lower() for c in containing]):
                    elements.append(element)
            elements.reverse()
        if clean:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                element.raw_html = lxml_html_tostring(cleaner.clean_html(element.lxml))
                elements.append(element)
        return _get_first_or_list(elements, first)

    def xpath(self, selector: str, *, clean: bool=False, first: bool=False, _encoding: str=None) -> _XPath:
        if False:
            while True:
                i = 10
        "Given an XPath selector, returns a list of\n        :class:`Element <Element>` objects or a single one.\n\n        :param selector: XPath Selector to use.\n        :param clean: Whether or not to sanitize the found HTML of ``<script>`` and ``<style>`` tags.\n        :param first: Whether or not to return just the first result.\n        :param _encoding: The encoding format.\n\n        If a sub-selector is specified (e.g. ``//a/@href``), a simple\n        list of results is returned.\n\n        See W3School's `XPath Examples\n        <https://www.w3schools.com/xml/xpath_examples.asp>`_\n        for more details.\n\n        If ``first`` is ``True``, only returns the first\n        :class:`Element <Element>` found.\n        "
        selected = self.lxml.xpath(selector)
        elements = [Element(element=selection, url=self.url, default_encoding=_encoding or self.encoding) if not isinstance(selection, etree._ElementUnicodeResult) else str(selection) for selection in selected]
        if clean:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                element.raw_html = lxml_html_tostring(cleaner.clean_html(element.lxml))
                elements.append(element)
        return _get_first_or_list(elements, first)

    def search(self, template: str) -> Result:
        if False:
            i = 10
            return i + 15
        'Search the :class:`Element <Element>` for the given Parse template.\n\n        :param template: The Parse template to use.\n        '
        return parse_search(template, self.html)

    def search_all(self, template: str) -> _Result:
        if False:
            i = 10
            return i + 15
        'Search the :class:`Element <Element>` (multiple times) for the given parse\n        template.\n\n        :param template: The Parse template to use.\n        '
        return [r for r in findall(template, self.html)]

    @property
    def links(self) -> _Links:
        if False:
            return 10
        'All found links on page, in as–is form.'

        def gen():
            if False:
                i = 10
                return i + 15
            for link in self.find('a'):
                try:
                    href = link.attrs['href'].strip()
                    if href and (not (href.startswith('#') and self.skip_anchors)) and (not href.startswith(('javascript:', 'mailto:'))):
                        yield href
                except KeyError:
                    pass
        return set(gen())

    def _make_absolute(self, link):
        if False:
            i = 10
            return i + 15
        'Makes a given link absolute.'
        parsed = urlparse(link)._asdict()
        if not parsed['netloc']:
            return urljoin(self.base_url, link)
        if not parsed['scheme']:
            parsed['scheme'] = urlparse(self.base_url).scheme
            parsed = (v for v in parsed.values())
            return urlunparse(parsed)
        return link

    @property
    def absolute_links(self) -> _Links:
        if False:
            i = 10
            return i + 15
        'All found links on page, in absolute form\n        (`learn more <https://www.navegabem.com/absolute-or-relative-links.html>`_).\n        '

        def gen():
            if False:
                i = 10
                return i + 15
            for link in self.links:
                yield self._make_absolute(link)
        return set(gen())

    @property
    def base_url(self) -> _URL:
        if False:
            i = 10
            return i + 15
        'The base URL for the page. Supports the ``<base>`` tag\n        (`learn more <https://www.w3schools.com/tags/tag_base.asp>`_).'
        base = self.find('base', first=True)
        if base:
            result = base.attrs.get('href', '').strip()
            if result:
                return result
        parsed = urlparse(self.url)._asdict()
        parsed['path'] = '/'.join(parsed['path'].split('/')[:-1]) + '/'
        parsed = (v for v in parsed.values())
        url = urlunparse(parsed)
        return url

class Element(BaseParser):
    """An element of HTML.

    :param element: The element from which to base the parsing upon.
    :param url: The URL from which the HTML originated, used for ``absolute_links``.
    :param default_encoding: Which encoding to default to.
    """
    __slots__ = ['element', 'url', 'skip_anchors', 'default_encoding', '_encoding', '_html', '_lxml', '_pq', '_attrs', 'session']

    def __init__(self, *, element, url: _URL, default_encoding: _DefaultEncoding=None) -> None:
        if False:
            return 10
        super(Element, self).__init__(element=element, url=url, default_encoding=default_encoding)
        self.element = element
        self.tag = element.tag
        self.lineno = element.sourceline
        self._attrs = None

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        attrs = ['{}={}'.format(attr, repr(self.attrs[attr])) for attr in self.attrs]
        return '<Element {} {}>'.format(repr(self.element.tag), ' '.join(attrs))

    @property
    def attrs(self) -> _Attrs:
        if False:
            return 10
        'Returns a dictionary of the attributes of the :class:`Element <Element>`\n        (`learn more <https://www.w3schools.com/tags/ref_attributes.asp>`_).\n        '
        if self._attrs is None:
            self._attrs = {k: v for (k, v) in self.element.items()}
            for attr in ['class', 'rel']:
                if attr in self._attrs:
                    self._attrs[attr] = tuple(self._attrs[attr].split())
        return self._attrs

class HTML(BaseParser):
    """An HTML document, ready for parsing.

    :param url: The URL from which the HTML originated, used for ``absolute_links``.
    :param html: HTML from which to base the parsing upon (optional).
    :param default_encoding: Which encoding to default to.
    """

    def __init__(self, *, session: Union['HTMLSession', 'AsyncHTMLSession']=None, url: str=DEFAULT_URL, html: _HTML, default_encoding: str=DEFAULT_ENCODING, async_: bool=False) -> None:
        if False:
            print('Hello World!')
        if isinstance(html, str):
            html = html.encode(DEFAULT_ENCODING)
        pq = PyQuery(html)
        super(HTML, self).__init__(element=pq('html') or pq.wrapAll('<html></html>')('html'), html=html, url=url, default_encoding=default_encoding)
        self.session = session or (async_ and AsyncHTMLSession()) or HTMLSession()
        self.page = None
        self.next_symbol = DEFAULT_NEXT_SYMBOL

    def __repr__(self) -> str:
        if False:
            return 10
        return f'<HTML url={self.url!r}>'

    def next(self, fetch: bool=False, next_symbol: _NextSymbol=None) -> _Next:
        if False:
            i = 10
            return i + 15
        'Attempts to find the next page, if there is one. If ``fetch``\n        is ``True`` (default), returns :class:`HTML <HTML>` object of\n        next page. If ``fetch`` is ``False``, simply returns the next URL.\n\n        '
        if next_symbol is None:
            next_symbol = DEFAULT_NEXT_SYMBOL

        def get_next():
            if False:
                while True:
                    i = 10
            candidates = self.find('a', containing=next_symbol)
            for candidate in candidates:
                if candidate.attrs.get('href'):
                    if 'next' in candidate.attrs.get('rel', []):
                        return candidate.attrs['href']
                    for _class in candidate.attrs.get('class', []):
                        if 'next' in _class:
                            return candidate.attrs['href']
                    if 'page' in candidate.attrs['href']:
                        return candidate.attrs['href']
            try:
                return candidates[-1].attrs['href']
            except IndexError:
                return None
        __next = get_next()
        if __next:
            url = self._make_absolute(__next)
        else:
            return None
        if fetch:
            return self.session.get(url)
        else:
            return url

    def __iter__(self):
        if False:
            print('Hello World!')
        next = self
        while True:
            yield next
            try:
                next = next.next(fetch=True, next_symbol=self.next_symbol).html
            except AttributeError:
                break

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.next(fetch=True, next_symbol=self.next_symbol).html

    def __aiter__(self):
        if False:
            print('Hello World!')
        return self

    async def __anext__(self):
        while True:
            url = self.next(fetch=False, next_symbol=self.next_symbol)
            if not url:
                break
            response = await self.session.get(url)
            return response.html

    def add_next_symbol(self, next_symbol):
        if False:
            print('Hello World!')
        self.next_symbol.append(next_symbol)

    async def _async_render(self, *, url: str, script: str=None, scrolldown, sleep: int, wait: float, reload, content: Optional[str], timeout: Union[float, int], keep_page: bool, cookies: list=[{}]):
        """ Handle page creation and js rendering. Internal use for render/arender methods. """
        try:
            page = await self.browser.newPage()
            await asyncio.sleep(wait)
            if cookies:
                for cookie in cookies:
                    if cookie:
                        await page.setCookie(cookie)
            if reload:
                await page.goto(url, options={'timeout': int(timeout * 1000)})
            else:
                await page.goto(f'data:text/html,{self.html}', options={'timeout': int(timeout * 1000)})
            result = None
            if script:
                result = await page.evaluate(script)
            if scrolldown:
                for _ in range(scrolldown):
                    await page._keyboard.down('PageDown')
                    await asyncio.sleep(sleep)
            else:
                await asyncio.sleep(sleep)
            if scrolldown:
                await page._keyboard.up('PageDown')
            content = await page.content()
            if not keep_page:
                await page.close()
                page = None
            return (content, result, page)
        except TimeoutError:
            await page.close()
            page = None
            return None

    def _convert_cookiejar_to_render(self, session_cookiejar):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert HTMLSession.cookies:cookiejar[] for browser.newPage().setCookie\n        '
        cookie_render = {}

        def __convert(cookiejar, key):
            if False:
                for i in range(10):
                    print('nop')
            try:
                v = eval('cookiejar.' + key)
                if not v:
                    kv = ''
                else:
                    kv = {key: v}
            except:
                kv = ''
            return kv
        keys = ['name', 'value', 'url', 'domain', 'path', 'sameSite', 'expires', 'httpOnly', 'secure']
        for key in keys:
            cookie_render.update(__convert(session_cookiejar, key))
        return cookie_render

    def _convert_cookiesjar_to_render(self):
        if False:
            while True:
                i = 10
        '\n        Convert HTMLSession.cookies for browser.newPage().setCookie\n        Return a list of dict\n        '
        cookies_render = []
        if isinstance(self.session.cookies, http.cookiejar.CookieJar):
            for cookie in self.session.cookies:
                cookies_render.append(self._convert_cookiejar_to_render(cookie))
        return cookies_render

    def render(self, retries: int=8, script: str=None, wait: float=0.2, scrolldown=False, sleep: int=0, reload: bool=True, timeout: Union[float, int]=8.0, keep_page: bool=False, cookies: list=[{}], send_cookies_session: bool=False):
        if False:
            i = 10
            return i + 15
        'Reloads the response in Chromium, and replaces HTML content\n        with an updated version, with JavaScript executed.\n\n        :param retries: The number of times to retry loading the page in Chromium.\n        :param script: JavaScript to execute upon page load (optional).\n        :param wait: The number of seconds to wait before loading the page, preventing timeouts (optional).\n        :param scrolldown: Integer, if provided, of how many times to page down.\n        :param sleep: Integer, if provided, of how many seconds to sleep after initial render.\n        :param reload: If ``False``, content will not be loaded from the browser, but will be provided from memory.\n        :param keep_page: If ``True`` will allow you to interact with the browser page through ``r.html.page``.\n\n        :param send_cookies_session: If ``True`` send ``HTMLSession.cookies`` convert.\n        :param cookies: If not ``empty`` send ``cookies``.\n\n        If ``scrolldown`` is specified, the page will scrolldown the specified\n        number of times, after sleeping the specified amount of time\n        (e.g. ``scrolldown=10, sleep=1``).\n\n        If just ``sleep`` is provided, the rendering will wait *n* seconds, before\n        returning.\n\n        If ``script`` is specified, it will execute the provided JavaScript at\n        runtime. Example:\n\n        .. code-block:: python\n\n            script = """\n                () => {\n                    return {\n                        width: document.documentElement.clientWidth,\n                        height: document.documentElement.clientHeight,\n                        deviceScaleFactor: window.devicePixelRatio,\n                    }\n                }\n            """\n\n        Returns the return value of the executed  ``script``, if any is provided:\n\n        .. code-block:: python\n\n            >>> r.html.render(script=script)\n            {\'width\': 800, \'height\': 600, \'deviceScaleFactor\': 1}\n\n        Warning: the first time you run this method, it will download\n        Chromium into your home directory (``~/.pyppeteer``).\n        '
        self.browser = self.session.browser
        content = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for i in range(retries):
            if not content:
                try:
                    (content, result, page) = self.session.loop.run_until_complete(self._async_render(url=self.url, script=script, sleep=sleep, wait=wait, content=self.html, reload=reload, scrolldown=scrolldown, timeout=timeout, keep_page=keep_page, cookies=cookies))
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries('Unable to render the page. Try increasing timeout')
        html = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING), default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html.__dict__)
        self.page = page
        return result

    async def arender(self, retries: int=8, script: str=None, wait: float=0.2, scrolldown=False, sleep: int=0, reload: bool=True, timeout: Union[float, int]=8.0, keep_page: bool=False, cookies: list=[{}], send_cookies_session: bool=False):
        """ Async version of render. Takes same parameters. """
        self.browser = await self.session.browser
        content = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for _ in range(retries):
            if not content:
                try:
                    (content, result, page) = await self._async_render(url=self.url, script=script, sleep=sleep, wait=wait, content=self.html, reload=reload, scrolldown=scrolldown, timeout=timeout, keep_page=keep_page, cookies=cookies)
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries('Unable to render the page. Try increasing timeout')
        html = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING), default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html.__dict__)
        self.page = page
        return result

class HTMLResponse(requests.Response):
    """An HTML-enabled :class:`requests.Response <requests.Response>` object.
    Effectively the same, but with an intelligent ``.html`` property added.
    """

    def __init__(self, session: Union['HTMLSession', 'AsyncHTMLSession']) -> None:
        if False:
            return 10
        super(HTMLResponse, self).__init__()
        self._html = None
        self.session = session

    @property
    def html(self) -> HTML:
        if False:
            print('Hello World!')
        if not self._html:
            self._html = HTML(session=self.session, url=self.url, html=self.content, default_encoding=self.encoding)
        return self._html

    @classmethod
    def _from_response(cls, response, session: Union['HTMLSession', 'AsyncHTMLSession']):
        if False:
            print('Hello World!')
        html_r = cls(session=session)
        html_r.__dict__.update(response.__dict__)
        return html_r

def user_agent(style=None) -> _UserAgent:
    if False:
        for i in range(10):
            print('nop')
    'Returns an apparently legit user-agent, if not requested one of a specific\n    style. Defaults to a Chrome-style User-Agent.\n    '
    global useragent
    if not useragent and style:
        useragent = UserAgent()
    return useragent[style] if style else DEFAULT_USER_AGENT

def _get_first_or_list(l, first=False):
    if False:
        for i in range(10):
            print('nop')
    if first:
        try:
            return l[0]
        except IndexError:
            return None
    else:
        return l

class BaseSession(requests.Session):
    """ A consumable session, for cookie persistence and connection pooling,
    amongst other things.
    """

    def __init__(self, mock_browser: bool=True, verify: bool=True, browser_args: list=['--no-sandbox']):
        if False:
            print('Hello World!')
        super().__init__()
        if mock_browser:
            self.headers['User-Agent'] = user_agent()
        self.hooks['response'].append(self.response_hook)
        self.verify = verify
        self.__browser_args = browser_args

    def response_hook(self, response, **kwargs) -> HTMLResponse:
        if False:
            i = 10
            return i + 15
        ' Change response encoding and replace it by a HTMLResponse. '
        if not response.encoding:
            response.encoding = DEFAULT_ENCODING
        return HTMLResponse._from_response(response, self)

    @property
    async def browser(self):
        if not hasattr(self, '_browser'):
            self._browser = await pyppeteer.launch(ignoreHTTPSErrors=not self.verify, headless=True, args=self.__browser_args)
        return self._browser

class HTMLSession(BaseSession):

    def __init__(self, **kwargs):
        if False:
            return 10
        super(HTMLSession, self).__init__(**kwargs)

    @property
    def browser(self):
        if False:
            return 10
        if not hasattr(self, '_browser'):
            self.loop = asyncio.get_event_loop()
            if self.loop.is_running():
                raise RuntimeError('Cannot use HTMLSession within an existing event loop. Use AsyncHTMLSession instead.')
            self._browser = self.loop.run_until_complete(super().browser)
        return self._browser

    def close(self):
        if False:
            while True:
                i = 10
        ' If a browser was created close it first. '
        if hasattr(self, '_browser'):
            self.loop.run_until_complete(self._browser.close())
        super().close()

class AsyncHTMLSession(BaseSession):
    """ An async consumable session. """

    def __init__(self, loop=None, workers=None, mock_browser: bool=True, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Set or create an event loop and a thread pool.\n\n            :param loop: Asyncio loop to use.\n            :param workers: Amount of threads to use for executing async calls.\n                If not pass it will default to the number of processors on the\n                machine, multiplied by 5. '
        super().__init__(*args, **kwargs)
        self.loop = loop or asyncio.get_event_loop()
        self.thread_pool = ThreadPoolExecutor(max_workers=workers)

    def request(self, *args, **kwargs):
        if False:
            print('Hello World!')
        ' Partial original request func and run it in a thread. '
        func = partial(super().request, *args, **kwargs)
        return self.loop.run_in_executor(self.thread_pool, func)

    async def close(self):
        """ If a browser was created close it first. """
        if hasattr(self, '_browser'):
            await self._browser.close()
        super().close()

    def run(self, *coros):
        if False:
            return 10
        ' Pass in all the coroutines you want to run, it will wrap each one\n            in a task, run it and wait for the result. Return a list with all\n            results, this is returned in the same order coros are passed in. '
        tasks = [asyncio.ensure_future(coro()) for coro in coros]
        (done, _) = self.loop.run_until_complete(asyncio.wait(tasks))
        return [t.result() for t in done]