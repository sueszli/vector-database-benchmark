import re
import sys
import urllib
import bs4
from .browser import Browser
from .form import Form
from .utils import LinkNotFoundError
from requests.structures import CaseInsensitiveDict

class _BrowserState:

    def __init__(self, page=None, url=None, form=None, request=None):
        if False:
            while True:
                i = 10
        self.page = page
        self.url = url
        self.form = form
        self.request = request

class StatefulBrowser(Browser):
    """An extension of :class:`Browser` that stores the browser's state
    and provides many convenient functions for interacting with HTML elements.
    It is the primary tool in MechanicalSoup for interfacing with websites.

    :param session: Attach a pre-existing requests Session instead of
        constructing a new one.
    :param soup_config: Configuration passed to BeautifulSoup to affect
        the way HTML is parsed. Defaults to ``{'features': 'lxml'}``.
        If overridden, it is highly recommended to `specify a parser
        <https://www.crummy.com/software/BeautifulSoup/bs4/doc/#specifying-the-parser-to-use>`__.
        Otherwise, BeautifulSoup will issue a warning and pick one for
        you, but the parser it chooses may be different on different
        machines.
    :param requests_adapters: Configuration passed to requests, to affect
        the way HTTP requests are performed.
    :param raise_on_404: If True, raise :class:`LinkNotFoundError`
        when visiting a page triggers a 404 Not Found error.
    :param user_agent: Set the user agent header to this value.

    All arguments are forwarded to :func:`Browser`.

    Examples ::

        browser = mechanicalsoup.StatefulBrowser(
            soup_config={'features': 'lxml'},  # Use the lxml HTML parser
            raise_on_404=True,
            user_agent='MyBot/0.1: mysite.example.com/bot_info',
        )
        browser.open(url)
        # ...
        browser.close()

    Once not used anymore, the browser can be closed
    using :func:`~Browser.close`.
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.__debug = False
        self.__verbose = 0
        self.__state = _BrowserState()
        self.get_current_page = lambda : self.page
        self.get_current_form = lambda : self.__state.form
        self.get_url = lambda : self.url

    def set_debug(self, debug):
        if False:
            for i in range(10):
                print('nop')
        'Set the debug mode (off by default).\n\n        Set to True to enable debug mode. When active, some actions\n        will launch a browser on the current page on failure to let\n        you inspect the page content.\n        '
        self.__debug = debug

    def get_debug(self):
        if False:
            i = 10
            return i + 15
        'Get the debug mode (off by default).'
        return self.__debug

    def set_verbose(self, verbose):
        if False:
            return 10
        'Set the verbosity level (an integer).\n\n        * 0 means no verbose output.\n        * 1 shows one dot per visited page (looks like a progress bar)\n        * >= 2 shows each visited URL.\n        '
        self.__verbose = verbose

    def get_verbose(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the verbosity level. See :func:`set_verbose()`.'
        return self.__verbose

    @property
    def page(self):
        if False:
            i = 10
            return i + 15
        'Get the current page as a soup object.'
        return self.__state.page

    @property
    def url(self):
        if False:
            return 10
        'Get the URL of the currently visited page.'
        return self.__state.url

    @property
    def form(self):
        if False:
            print('Hello World!')
        'Get the currently selected form as a :class:`Form` object.\n        See :func:`select_form`.\n        '
        if self.__state.form is None:
            raise AttributeError('No form has been selected yet on this page.')
        return self.__state.form

    def __setitem__(self, name, value):
        if False:
            return 10
        'Call item assignment on the currently selected form.\n        See :func:`Form.__setitem__`.\n        '
        self.form[name] = value

    def new_control(self, type, name, value, **kwargs):
        if False:
            print('Hello World!')
        'Call :func:`Form.new_control` on the currently selected form.'
        return self.form.new_control(type, name, value, **kwargs)

    def absolute_url(self, url):
        if False:
            return 10
        'Return the absolute URL made from the current URL and ``url``.\n        The current URL is only used to provide any missing components of\n        ``url``, as in the `.urljoin() method of urllib.parse\n        <https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urljoin>`__.\n        '
        return urllib.parse.urljoin(self.url, url)

    def open(self, url, *args, **kwargs):
        if False:
            print('Hello World!')
        "Open the URL and store the Browser's state in this object.\n        All arguments are forwarded to :func:`Browser.get`.\n\n        :return: Forwarded from :func:`Browser.get`.\n        "
        if self.__verbose == 1:
            sys.stdout.write('.')
            sys.stdout.flush()
        elif self.__verbose >= 2:
            print(url)
        resp = self.get(url, *args, **kwargs)
        self.__state = _BrowserState(page=resp.soup, url=resp.url, request=resp.request)
        return resp

    def open_fake_page(self, page_text, url=None, soup_config=None):
        if False:
            for i in range(10):
                print('nop')
        "Mock version of :func:`open`.\n\n        Behave as if opening a page whose text is ``page_text``, but do not\n        perform any network access. If ``url`` is set, pretend it is the page's\n        URL. Useful mainly for testing.\n        "
        soup_config = soup_config or self.soup_config
        self.__state = _BrowserState(page=bs4.BeautifulSoup(page_text, **soup_config), url=url)

    def open_relative(self, url, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Like :func:`open`, but ``url`` can be relative to the currently\n        visited page.\n        '
        return self.open(self.absolute_url(url), *args, **kwargs)

    def refresh(self):
        if False:
            for i in range(10):
                print('nop')
        'Reload the current page with the same request as originally done.\n        Any change (`select_form`, or any value filled-in in the form) made to\n        the current page before refresh is discarded.\n\n        :raise ValueError: Raised if no refreshable page is loaded, e.g., when\n            using the shallow ``Browser`` wrapper functions.\n\n        :return: Response of the request.'
        old_request = self.__state.request
        if old_request is None:
            raise ValueError('The current page is not refreshable. Either no page is opened or low-level browser methods were used to do so')
        resp = self.session.send(old_request)
        Browser.add_soup(resp, self.soup_config)
        self.__state = _BrowserState(page=resp.soup, url=resp.url, request=resp.request)
        return resp

    def select_form(self, selector='form', nr=0):
        if False:
            while True:
                i = 10
        'Select a form in the current page.\n\n        :param selector: CSS selector or a bs4.element.Tag object to identify\n            the form to select.\n            If not specified, ``selector`` defaults to "form", which is\n            useful if, e.g., there is only one form on the page.\n            For ``selector`` syntax, see the `.select() method in BeautifulSoup\n            <https://www.crummy.com/software/BeautifulSoup/bs4/doc/#css-selectors>`__.\n        :param nr: A zero-based index specifying which form among those that\n            match ``selector`` will be selected. Useful when one or more forms\n            have the same attributes as the form you want to select, and its\n            position on the page is the only way to uniquely identify it.\n            Default is the first matching form (``nr=0``).\n\n        :return: The selected form as a soup object. It can also be\n            retrieved later with the :attr:`form` attribute.\n        '

        def find_associated_elements(form_id):
            if False:
                i = 10
                return i + 15
            'Find all elements associated to a form\n                (i.e. an element with a form attribute -> ``form=form_id``)\n            '
            elements_with_owner_form = ('input', 'button', 'fieldset', 'object', 'output', 'select', 'textarea')
            found_elements = []
            for element in elements_with_owner_form:
                found_elements.extend(self.page.find_all(element, form=form_id))
            return found_elements
        if isinstance(selector, bs4.element.Tag):
            if selector.name != 'form':
                raise LinkNotFoundError
            form = selector
        else:
            found_forms = self.page.select(selector, limit=nr + 1)
            if len(found_forms) != nr + 1:
                if self.__debug:
                    print('select_form failed for', selector)
                    self.launch_browser()
                raise LinkNotFoundError()
            form = found_forms[-1]
        if form and form.has_attr('id'):
            form_id = form['id']
            new_elements = find_associated_elements(form_id)
            form.extend(new_elements)
        self.__state.form = Form(form)
        return self.form

    def _merge_referer(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Helper function to set the Referer header in kwargs passed to\n        requests, if it has not already been overridden by the user.'
        referer = self.url
        headers = CaseInsensitiveDict(kwargs.get('headers', {}))
        if referer is not None and 'Referer' not in headers:
            headers['Referer'] = referer
            kwargs['headers'] = headers
        return kwargs

    def submit_selected(self, btnName=None, update_state=True, **kwargs):
        if False:
            i = 10
            return i + 15
        'Submit the form that was selected with :func:`select_form`.\n\n        :return: Forwarded from :func:`Browser.submit`.\n\n        :param btnName: Passed to :func:`Form.choose_submit` to choose the\n            element of the current form to use for submission. If ``None``,\n            will choose the first valid submit element in the form, if one\n            exists. If ``False``, will not use any submit element; this is\n            useful for simulating AJAX requests, for example.\n\n        :param update_state: If False, the form will be submitted but the\n            browser state will remain unchanged; this is useful for forms that\n            result in a download of a file, for example.\n\n        All other arguments are forwarded to :func:`Browser.submit`.\n        '
        self.form.choose_submit(btnName)
        kwargs = self._merge_referer(**kwargs)
        resp = self.submit(self.__state.form, url=self.__state.url, **kwargs)
        if update_state:
            self.__state = _BrowserState(page=resp.soup, url=resp.url, request=resp.request)
        return resp

    def list_links(self, *args, **kwargs):
        if False:
            return 10
        'Display the list of links in the current page. Arguments are\n        forwarded to :func:`links`.\n        '
        print('Links in the current page:')
        for link in self.links(*args, **kwargs):
            print('    ', link)

    def links(self, url_regex=None, link_text=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Return links in the page, as a list of bs4.element.Tag objects.\n\n        To return links matching specific criteria, specify ``url_regex``\n        to match the *href*-attribute, or ``link_text`` to match the\n        *text*-attribute of the Tag. All other arguments are forwarded to\n        the `.find_all() method in BeautifulSoup\n        <https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find-all>`__.\n        '
        all_links = self.page.find_all('a', *args, href=True, **kwargs)
        if url_regex is not None:
            all_links = [a for a in all_links if re.search(url_regex, a['href'])]
        if link_text is not None:
            all_links = [a for a in all_links if a.text == link_text]
        return all_links

    def find_link(self, *args, **kwargs):
        if False:
            return 10
        'Find and return a link, as a bs4.element.Tag object.\n\n        The search can be refined by specifying any argument that is accepted\n        by :func:`links`. If several links match, return the first one found.\n\n        If no link is found, raise :class:`LinkNotFoundError`.\n        '
        links = self.links(*args, **kwargs)
        if len(links) == 0:
            raise LinkNotFoundError()
        else:
            return links[0]

    def _find_link_internal(self, link, args, kwargs):
        if False:
            return 10
        'Wrapper around find_link that deals with convenience special-cases:\n\n        * If ``link`` has an *href*-attribute, then return it. If not,\n          consider it as a ``url_regex`` argument.\n\n        * If searching for the link fails and debug is active, launch\n          a browser.\n        '
        if hasattr(link, 'attrs') and 'href' in link.attrs:
            return link
        if link and 'url_regex' in kwargs:
            raise ValueError('link parameter cannot be treated as url_regex because url_regex is already present in keyword arguments')
        elif link:
            kwargs['url_regex'] = link
        try:
            return self.find_link(*args, **kwargs)
        except LinkNotFoundError:
            if self.get_debug():
                print('find_link failed for', kwargs)
                self.list_links()
                self.launch_browser()
            raise

    def follow_link(self, link=None, *bs4_args, bs4_kwargs={}, requests_kwargs={}, **kwargs):
        if False:
            while True:
                i = 10
        "Follow a link.\n\n        If ``link`` is a bs4.element.Tag (i.e. from a previous call to\n        :func:`links` or :func:`find_link`), then follow the link.\n\n        If ``link`` doesn't have a *href*-attribute or is None, treat\n        ``link`` as a url_regex and look it up with :func:`find_link`.\n        ``bs4_kwargs`` are forwarded to :func:`find_link`.\n        For backward compatibility, any excess keyword arguments\n        (aka ``**kwargs``)\n        are also forwarded to :func:`find_link`.\n\n        If the link is not found, raise :class:`LinkNotFoundError`.\n        Before raising, if debug is activated, list available links in the\n        page and launch a browser.\n\n        ``requests_kwargs`` are forwarded to :func:`open_relative`.\n\n        :return: Forwarded from :func:`open_relative`.\n        "
        link = self._find_link_internal(link, bs4_args, {**bs4_kwargs, **kwargs})
        requests_kwargs = self._merge_referer(**requests_kwargs)
        return self.open_relative(link['href'], **requests_kwargs)

    def download_link(self, link=None, file=None, *bs4_args, bs4_kwargs={}, requests_kwargs={}, **kwargs):
        if False:
            while True:
                i = 10
        'Downloads the contents of a link to a file. This function behaves\n        similarly to :func:`follow_link`, but the browser state will\n        not change when calling this function.\n\n        :param file: Filesystem path where the page contents will be\n            downloaded. If the file already exists, it will be overwritten.\n\n        Other arguments are the same as :func:`follow_link` (``link``\n        can either be a bs4.element.Tag or a URL regex.\n        ``bs4_kwargs`` arguments are forwarded to :func:`find_link`,\n        as are any excess keyword arguments (aka ``**kwargs``) for backwards\n        compatibility).\n\n        :return: `requests.Response\n            <http://docs.python-requests.org/en/master/api/#requests.Response>`__\n            object.\n        '
        link = self._find_link_internal(link, bs4_args, {**bs4_kwargs, **kwargs})
        url = self.absolute_url(link['href'])
        requests_kwargs = self._merge_referer(**requests_kwargs)
        response = self.session.get(url, **requests_kwargs)
        if self.raise_on_404 and response.status_code == 404:
            raise LinkNotFoundError()
        if file is not None:
            with open(file, 'wb') as f:
                f.write(response.content)
        return response

    def launch_browser(self, soup=None):
        if False:
            return 10
        'Launch a browser to display a page, for debugging purposes.\n\n        :param: soup: Page contents to display, supplied as a bs4 soup object.\n            Defaults to the current page of the ``StatefulBrowser`` instance.\n        '
        if soup is None:
            soup = self.page
        super().launch_browser(soup)